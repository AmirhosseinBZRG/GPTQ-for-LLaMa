import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_hellaswag(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    dataset = load_dataset('hellaswag')
    traindata = dataset['train']
    valdata = dataset['validation']

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    # Set padding token to EOS token
    tokenizer.pad_token = tokenizer.eos_token

    # Function to process each sample
    def process_sample(sample):
        ctx = sample['ctx']
        ending = sample['endings'][0]  # Take the first ending
        return f"{ctx} {ending}"

    # Process and tokenize the data
    trainenc = tokenizer.batch_encode_plus(
        [process_sample(sample) for sample in traindata],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=seqlen
    )
    
    testenc = tokenizer.batch_encode_plus(
        [process_sample(sample) for sample in valdata],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=seqlen
    )

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - 1)
        inp = trainenc.input_ids[i]
        tar = inp.clone()
        tar[:-1] = -100
        trainloader.append((inp.unsqueeze(0), tar.unsqueeze(0)))

    return trainloader, testenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'hellaswag' in name:
        return get_hellaswag(nsamples, seed, seqlen, model)

