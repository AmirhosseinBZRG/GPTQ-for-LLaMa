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
def gpt2_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model



def gpt2_multigpu(model, gpus, gpu_dist):
    model.transformer.wte = model.transformer.wte.to(gpus[0])
    
    if hasattr(model.transformer, 'ln_f') and model.transformer.ln_f:
        model.transformer.ln_f = model.transformer.ln_f.to(gpus[0])
    
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])
    
    cache = {'mask': None, 'position_ids': None}
    
    class MoveModule(nn.Module):
        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache = invalidate_cache
        
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
                kwargs['attention_mask'] = cache['mask']
            if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
                cache['position_ids'] = kwargs['position_ids'].to(self.dev)
                kwargs['position_ids'] = cache['position_ids']
            return self.module(*inp, **kwargs)
    
    layers = model.transformer.h
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) - 1 else gpus[(i - 1) // pergpu]), i == 0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0] - 1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus += [i] * gpu_dist[i]
        remaining_assignments = len(layers) - len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus += [-1] * remaining_assignments
        assigned_gpus += [0]
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i == 0)
    
    model.gpus = gpus

if __name__ == '__main__':
    import argparse
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='GPT-2 model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'hellaswag'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='Evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='Test generation.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic.')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval.')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe', action='store_true', help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. When this feature is enabled, `--save` or `--save_safetensors` would be disabled.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')

    args = parser.parse_args()

    # Parse layers distribution argument
    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    # Load model or initialize it
    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model)
        model.eval()

    # Load data
    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.config.n_positions)

    # Perform sequential quantization if applicable
    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = gpt2_sequential(model, dataloader, DEV)
        print(f"Quantization time: {time.time() - tick:.2f} seconds")

    # Benchmarking
    if args.benchmark:
        gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            gpt2_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        input_ids = next(iter(dataloader))[0][:, :args.benchmark]
        benchmark(model, input_ids, check=args.check)

    # Evaluation
    if args.eval:
        datasets = ['wikitext2', 'hellaswag']
        for dataset in datasets:
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.config.n_positions)
            print(f"Evaluating on dataset: {dataset}")
            gpt2_eval(model, testloader, DEV)

    # Test generation
    if args.test_generation:
        gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            gpt2_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
        with torch.no_grad():
            generated_ids = model.generate(input_ids)

    # Export quantization parameters if specified
    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    # Save model if specified
    if not args.observe and args.save:
        gpt2_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save)

    if not args.observe and args.save_safetensors:
        gpt2_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)


