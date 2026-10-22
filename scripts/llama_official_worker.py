#!/usr/bin/env python3
"""Our Llama capture/codec worker; never imports or patches baseline methods."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = 'unsloth/Llama-3.2-3B'  # the unquantized checkpoint in AQUA's README
REVISION = 'd4446454d87d51aa42e1fb174f25acc5f8762331'


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''): h.update(part)
    return h.hexdigest()


def cuda():
    import torch
    if os.environ.get('TRITON_INTERPRET') or not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU or Triton interpreter fallback')
    torch.cuda.set_device(0)
    torch.ones(1, device='cuda').add_(1); torch.cuda.synchronize()
    return torch


def model_manifest(path):
    data = json.loads(Path(path).read_text())
    if (data['model'] != MODEL or data['revision'] != REVISION or
            data['config']['model_type'] != 'llama' or data['config']['num_hidden_layers'] != 28 or
            data['config'].get('quantization_config')):
        raise ValueError('Wrong model, revision, architecture or quantized weights')
    return data


def dataset_tokens(tokenizer, split):
    from datasets import load_dataset
    # Same joining/tokenization as the official AQUA perplexity entry point.
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    text = '\n\n'.join(dataset['text'])
    ids = tokenizer(text, return_tensors='pt')['input_ids'][0]
    return ids, dict(split=split, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        tokens=ids.numel(), token_ids_sha256=hashlib.sha256(ids.numpy().astype('<i8').tobytes()).hexdigest(),
        dataset_fingerprint=dataset._fingerprint)


def preflight(args):
    torch = cuda()
    from transformers import AutoTokenizer
    data = model_manifest(args.manifest)
    tokenizer = AutoTokenizer.from_pretrained(data['snapshot'], local_files_only=True)
    _, identity = dataset_tokens(tokenizer, 'test')
    software = {}
    for name in ['torch', 'transformers', 'triton', 'datasets', 'numpy', 'pyarrow', 'accelerate']:
        try: software[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: software[name] = None
    write(args.output, dict(gpu_measured=True, device=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0), cuda=torch.version.cuda,
        total_device_bytes=torch.cuda.get_device_properties(0).total_memory,
        software=software, test_data=identity, model_revision=REVISION))


def capture(args):
    torch = cuda()
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from qwen_kv_bench.model_runtime import _pairs
    data = model_manifest(args.manifest); out = args.output; out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(data['snapshot'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(data['snapshot'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError('Original unquantized BF16 parameters required')
    if model.config.rope_scaling.get('rope_type') != 'llama3':
        raise ValueError('Expected the pinned Llama3 scaled RoPE configuration')
    rotary = model.model.rotary_emb
    if float(getattr(rotary, 'attention_scaling', 1.)) != 1.:
        raise ValueError('Unsupported RoPE attention scaling')
    inv = rotary.inv_freq.detach().float().cpu().numpy().copy()
    manifest = dict(format='llama32_original_captures_v1', model=MODEL, revision=REVISION,
        config=model.config.to_dict(), layers=28, rope_inv_freq=inv.tolist(), entries=[], datasets={},
        scope='all 28 original post-RoPE HF KV layers; our codec study, not paper calibration')
    for split, count, tokens in [('train', 4, 1024), ('validation', 1, 1024), ('test', 1, 8192)]:
        ids, identity = dataset_tokens(tokenizer, split); manifest['datasets'][split] = identity
        if len(ids) < count * tokens: raise ValueError('Too few source tokens')
        for case in range(count):
            name = f'{split}_{case:03d}'; print(f'{name}: original BF16 prefill, all 28 layers', flush=True)
            with torch.inference_mode():
                result = model(ids[case*tokens:(case+1)*tokens][None].to('cuda'), use_cache=True, logits_to_keep=1)
            pairs = _pairs(result.past_key_values)
            if len(pairs) != 28: raise ValueError('Full 28-layer cache required')
            arrays = {}
            for kind, index in [('K_bits', 0), ('V_bits', 1)]:
                if any(pair[index].dtype != torch.bfloat16 for pair in pairs): raise ValueError('Non-BF16 KV')
                arrays[kind] = np.stack([pair[index].detach().transpose(1, 2).contiguous()
                    .view(torch.uint16).cpu().numpy()[0].reshape(tokens, -1).copy() for pair in pairs])
            files = {}; hashes = {}
            for kind, a in arrays.items():
                path = out / (name + '_' + kind + '.npy'); np.save(path, a, allow_pickle=False)
                files[kind] = path.name; hashes[kind] = sha(path)
            manifest['entries'].append(dict(name=name, split=split, tokens=tokens, files=files, hashes=hashes))
            write(out/'manifest.json', manifest)
            del result, pairs, arrays; gc.collect(); torch.cuda.empty_cache()
    del model; gc.collect(); torch.cuda.empty_cache()
    trains = [load_capture(out, manifest, e) for e in manifest['entries'] if e['split'] == 'train']
    from qwen_kv_bench import transfer_reference as ref
    flat = np.concatenate([ref.words(a).ravel() for a in trains])
    torch.save(torch.from_numpy(flat).view(torch.bfloat16), out/'splitzip_train.pt')
    del flat, trains
    test = load_capture(out, manifest, next(e for e in manifest['entries'] if e['split'] == 'test'))
    manifest['benchmark_inputs'] = []
    for start in [0, 8192-1024]:
        words = ref.words(test)[:, start:start+1024].copy(); name = f'test_{start:05d}.pt'
        torch.save(torch.from_numpy(words.ravel()).view(torch.bfloat16), out/name)
        manifest['benchmark_inputs'].append(dict(path=name, start=start, tokens=1024,
            shape=list(words.shape), values=words.size, original_word_sha256=hashlib.sha256(words.astype('<u2').tobytes()).hexdigest(),
            file_sha256=sha(out/name)))
    manifest['calibration_input'] = dict(path='splitzip_train.pt', file_sha256=sha(out/'splitzip_train.pt'))
    write(out/'manifest.json', manifest)


def load_capture(directory, manifest, entry):
    import numpy as np
    result = {}
    for name, filename in entry['files'].items():
        path = directory/filename
        if sha(path) != entry['hashes'][name]: raise ValueError('Capture checksum mismatch')
        result[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    return dict(result, Hk=8, hd=128, base=500000., rope_inv_freq=manifest['rope_inv_freq'])


def ours(args):
    torch = cuda()
    import numpy as np
    from qwen_kv_bench import transfer_reference as ref, transfer_probability as prob, matched_transfer as matched
    from scripts.bench_qwen_matched_gpu import measure, sha_words
    out = args.output; out.mkdir(parents=True, exist_ok=True); directory = args.captures
    m = json.loads((directory/'manifest.json').read_text())
    if m['revision'] != REVISION or m['layers'] != 28: raise ValueError('Wrong captured model')
    sources = [matched.adjacent_capture(load_capture(directory,m,e)) for e in m['entries'] if e['split']=='train']
    scale = matched.adjacent_capture(load_capture(directory,m,next(e for e in m['entries'] if e['split']=='validation')))
    print('Offline Block32 A fit: four TRAIN contexts, validation scale, all 28 layers', flush=True)
    begin = time.perf_counter()
    base = ref.fit(sources, scale, mode='block', block_size=32, max_positions=8192)
    ref.save_model(base, out/'block32_base.npz'); state = prob.empty_state(base,'A')
    prob.save_state(state,out/'block32_state.npz'); fitting_seconds = time.perf_counter()-begin
    del sources, scale; gc.collect()
    test = load_capture(directory,m,next(e for e in m['entries'] if e['split']=='test'))
    result = dict(format='llama32_ours_gpu_v1', gpu_measured=True, model=MODEL, revision=REVISION,
        fitting_seconds=fitting_seconds, layers=28, rows=[],
        calibration_manifest_sha256=sha(directory/'manifest.json'),
        base_file_sha256=sha(out/'block32_base.npz'),state_file_sha256=sha(out/'block32_state.npz'),
        state_sha256=state.state_sha256,base_statistics=ref.model_statistics(base),
        scope='our GPU transfer codec; native CUDA intervals and synchronized wall times; not serving throughput')
    for name in ['raw_bf16','block32_A']:
        adapter = matched.Raw('cuda:0') if name=='raw_bf16' else matched.Predictive(name,base,state,'cuda:0')
        for case in m['benchmark_inputs']:
            start = case['start']; words = ref.words(test)[:,start:start+case['tokens']].copy()
            if sha_words(words) != case['original_word_sha256']: raise ValueError('Different benchmark words')
            print(f'{name}: tokens {start}:{start+case["tokens"]}, all 28 layers', flush=True)
            source = torch.from_numpy(words).to('cuda')
            packet, enc = measure(lambda:adapter.encode(source,start), adapter.check_encode, args, torch, torch.device('cuda'), words.nbytes)
            def check(value):
                adapter.check_decode()
                if not torch.equal(value,source): raise AssertionError('Original BF16 words differ')
            decoded, dec = measure(lambda:adapter.decode(packet), check, args, torch, torch.device('cuda'), words.nbytes)
            wire, account = matched.export_record(adapter,packet,{'model':MODEL,'revision':REVISION})
            restored = adapter.decode(matched.import_record(adapter,wire,{'model':MODEL,'revision':REVISION})); check(restored)
            result['rows'].append(dict(method=name,start=start,tokens=case['tokens'],values=words.size,
                original_word_sha256=case['original_word_sha256'],bytes=account,encode=enc,decode=dec,exact=True,
                persistent_device_bytes=adapter.statistics().get('persistent_device_bytes')))
            write(out/'results.json',result)
            del source, packet, decoded, restored, wire; gc.collect()
        del adapter; gc.collect(); torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['preflight','capture','ours']); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--captures',type=Path)
    p.add_argument('--warmup',type=int,default=1); p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.warmup<0 or args.repeats<1: p.error('Invalid timing counts')
    if args.phase=='ours' and args.captures is None: p.error('--captures is required')
    {'preflight':preflight,'capture':capture,'ours':ours}[args.phase](args)


if __name__=='__main__': main()
