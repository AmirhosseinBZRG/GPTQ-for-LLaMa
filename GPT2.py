import huggingface_hub
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import scipy
from scipy import stats
from torch.profiler import profile, record_function, ProfilerActivity

# Import necessary utilities
from utils import find_layers, DEV, set_seed, get_wikitext2, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
from transformers import AutoModelForCausalLM

def get_gpt2(model_name):
    huggingface_hub.login("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")
    access_token = "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"

    def skip(*args, **kwargs):
        pass  # Skip weight initialization
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    try:
        # Load the GPT-2 model with float16 precision using AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, ignore_mismatched_sizes=True)
        model.seqlen = 1024  # Set the sequence length for GPT-2
    except Exception as e:
        print(f"Error loading model: {e}")
        return None  # or handle as appropriate
    return model

@torch.no_grad()
def gpt2_sequential(model, dataloader, dev):
    print('Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h  # Access GPT-2 layers
    model.transformer.wte = model.transformer.wte.to(dev)  # Move embedding to device
    model.transformer.ln_f = model.transformer.ln_f.to(dev)  # Move final layer norm to device
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.n_embd), dtype=dtype, device=dev)  # Adjust for GPT-2 embedding size
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])  # Wrap the first layer
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module
    model.transformer.wte = model.transformer.wte.cpu()
    model.transformer.ln_f = model.transformer.ln_f.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    
    print('Ready.')
    quantizers = {}
    observer = Observer()  # Ensure Observer is defined or imported appropriately

    for i in range(len(layers)):
        print(f'Quantizing layer {i + 1}/{len(layers)}..')
        layer = layers[i].to(dev)
        full = find_layers(layer)

        # Define sequential processing based on true_sequential flag
        if args.true_sequential:
            sequential = [['attn.c_attn', 'attn.c_proj'], ['mlp.c_fc', 'mlp.c_proj']]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name], observe=args.observe)
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            # Add batch processing
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

    model.config.use_cache = use_cache
    return quantizers

@torch.no_grad()
def gpt2_eval(model, testenc, dev):
    print('Evaluating ...')
    model = model.to(dev)
    testenc = testenc.input_ids.to(dev)
    nsamples = testenc.numel() // model.seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.wte = model.transformer.wte.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.n_embd), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    model.transformer.wte = model.transformer.wte.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    # Profiling setup
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
        with record_function("model_inference"):
            for i in range(len(layers)):
                layer = layers[i].to(dev)
                for j in range(nsamples):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                layers[i] = layer.cpu()
                del layer
                torch.cuda.empty_cache()
                inps, outs = outs, inps

    # Print profiler results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Calculate perplexity
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        lm_logits = model(hidden_states).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl
