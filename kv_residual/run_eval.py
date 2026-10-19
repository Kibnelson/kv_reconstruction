"""Indexed-byte and exact-recovery evaluation of frozen residual KV codecs.

CPU reference timings are diagnostics, not a serving benchmark. Every codec
decodes its serialized cache; source arrays are used only for verification.
The complete shared learned state is charged separately for every case in the
default aggregate (and amortized alternatives are reported explicitly).
"""
import argparse
import hashlib
import json
import pathlib
import platform
import time
import zlib
import numpy as np
from kv_research import container, learned_codec
from kv_stacked import huffman_codec
from kv_stacked.run_eval import aggregate, matrices, equal


def factories(config):
    """Return per-record encoders, decoders, and full serialized state bytes."""
    encoders, decoders, states = {}, {}, {}
    models = {}
    for name, spec in config['candidates'].items():
        kind, options = spec['kind'], dict(spec.get('options', {}))
        states[name] = 0
        if kind == 'huffman':
            encoders[name] = lambda w, l, k: huffman_codec.encode_matrix(w)
            decoders[name] = lambda b, l, k: huffman_codec.decode_matrix(b)
        elif kind == 'huffman_lower_deflate':
            from kv_coset_fields import combined, controls
            strategy = options.pop('strategy', 'deflate')
            if options:
                raise ValueError('unknown lower Deflate options: ' + str(options))
            def enc(w, l, k, strategy=strategy):
                return combined.encode_matrix(w,
                    lambda lo, e: controls.encode_lower(lo, e, strategy), 2,
                    lambda b, e: controls.decode_lower(b, e, strategy))
            def dec(b, l, k, strategy=strategy):
                return combined.decode_matrix(b,
                    lambda lo, e: controls.decode_lower(lo, e, strategy), 2)
            encoders[name], decoders[name] = enc, dec
        elif kind == 'direct':
            from kv_residual import direct
            encoders[name] = lambda w, l, k, options=options: direct.encode_matrix(w, **options)
            decoders[name] = lambda b, l, k: direct.decode_matrix(b)
        elif kind in ('original_learned', 'original_learned_deflate', 'learned', 'learned_e8'):
            path = pathlib.Path(spec.get('model', config.get('model', 'kv_research/models/learned64.bin')))
            identity = str(path.resolve())
            if identity not in models:
                models[identity] = learned_codec.load_model(path)
            model = models[identity]
            states[name] = path.stat().st_size
            if kind == 'original_learned':
                encoders[name] = lambda w, l, k, model=model, options=options: learned_codec.encode_matrix(w, model, f'layer_{l:02d}_{k}', **options)
                decoders[name] = lambda b, l, k, model=model: learned_codec.decode_matrix(b, model)
            elif kind == 'original_learned_deflate':
                from kv_stacked import wrapper
                level = options.pop('level', 6)
                encoders[name] = lambda w, l, k, model=model, options=options, level=level: wrapper.wrap(
                    learned_codec.encode_matrix(w, model, f'layer_{l:02d}_{k}', **options), level)
                decoders[name] = lambda b, l, k, model=model: learned_codec.decode_matrix(wrapper.unwrap(b), model)
            else:
                if kind == 'learned':
                    from kv_residual import learned as module
                else:
                    from kv_residual import learned_e8 as module
                # Force the actual learned branch; codec diagnostics expose its
                # internal correction modes. Do not substitute strong controls.
                options.setdefault('force', True)
                encoders[name] = lambda w, l, k, model=model, options=options, module=module: module.encode_matrix(w, model, f'layer_{l:02d}_{k}', **options)
                decoders[name] = lambda b, l, k, model=model, module=module: module.decode_matrix(b, model)
        else:
            raise ValueError('unknown candidate kind ' + kind)
    return encoders, decoders, states


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--data-dir')
    p.add_argument('--only-index', type=int)
    p.add_argument('--generation-candidate', action='append', default=[])
    p.add_argument('--save-sample', action='store_true')
    p.add_argument('--model-dir', default='.smoke_model')
    return p.parse_args()


def main():
    args = parse_args()
    config_path, manifest_path = pathlib.Path(args.config), pathlib.Path(args.manifest)
    config = json.loads(config_path.read_text())
    if config.get('tile_tokens', 16) != 16:
        raise ValueError('this evaluation protocol requires 16-token tiles')
    manifest = json.loads(manifest_path.read_text())
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else manifest_path.parent
    encoders, decoders, states = factories(config)
    names = list(config['candidates'])
    if any(name not in names for name in args.generation_candidate):
        raise ValueError('generation candidate absent from config')
    output = pathlib.Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    result = dict(config=config, capture_manifest=manifest, cases=[], state_bytes=states,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__,
                     zlib=zlib.ZLIB_RUNTIME_VERSION, device='cpu'),
        scope='Exact KV storage and CPU reference reconstruction, independent 16-token records. Forced learned branches are measured directly. Full decoder state is charged per cache; no GPU, bounded attention-buffer, admission, or concurrent serving claim.')
    selected = list(enumerate(manifest['cases']))
    if args.only_index is not None:
        if not 0 <= args.only_index < len(selected):
            raise ValueError('case index out of range')
        selected = [selected[args.only_index]]
    lm = tok = None
    saved_e8 = False
    for index, case in selected:
        print('case ' + case['name'], flush=True)
        with np.load(data_dir / case['file']) as f:
            raw = matrices({k: f[k].copy() for k in f.files})
        payload = b''.join(w.astype('<u2').tobytes() for pair in raw for w in pair)
        if len(payload) != case['raw_payload_bytes'] or hashlib.sha256(payload).hexdigest() != case['raw_sha256']:
            raise AssertionError('capture raw bytes/SHA256 mismatch')
        row = {k: v for k, v in case.items() if k != 'input_token_ids'}
        row['codecs'] = {}
        for name in names:
            started = time.perf_counter()
            encoded = container.encode_cache(raw, encoders[name], tile_tokens=16)
            encode_seconds = time.perf_counter() - started
            started = time.perf_counter()
            recovered = container.decode_cache(encoded.data, decoders[name])
            decode_seconds = time.perf_counter() - started
            if not equal(raw, recovered):
                raise AssertionError(name + ': exact recovery failed')
            damaged = bytearray(encoded.data)
            shape = container.shape(damaged)
            damaged[shape['payload_start']] ^= 255
            tile_index = shape['ntiles'] - 1
            actual = container.decode_tile(damaged, shape['layers'] - 1, tile_index, decoders[name])
            tile_start = tile_index * shape['tile_tokens']
            if not all(np.array_equal(a, b[tile_start:]) for a, b in zip(actual, raw[-1])):
                raise AssertionError(name + ': later tile depends on damaged earlier payload')
            stats = {**encoded.stats, 'roundtrip_exact': True,
                'independent_later_tile': True,
                'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds,
                'encoded_sha256': hashlib.sha256(encoded.data).hexdigest(),
                'decoder_state_bytes': states[name],
                'saving_with_state_percent': 100 * (1 - (len(encoded.data) + states[name]) / len(payload)),
                'savings_including_state_percent': {str(n): 100 * (1 - (len(encoded.data) + states[name] / n) / len(payload)) for n in (1, 4, 16)}}
            if args.save_sample and index == 0 and name in args.generation_candidate:
                sample_name = name + '_sample.bin'
                (output.parent / sample_name).write_bytes(encoded.data)
                stats['sample_file'] = sample_name
                sample_spec = config['candidates'][name]
                if sample_spec['kind'] == 'learned_e8' and sample_spec.get('options', {}).get('family', 'e8') == 'e8' and not saved_e8:
                    (output.parent / 'e8_sample.bin').write_bytes(encoded.data)
                    saved_e8 = True
            if name in args.generation_candidate and index == 0:
                if lm is None:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    torch.set_num_threads(4)
                    torch.set_num_interop_threads(1)
                    lm = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                        dtype=torch.bfloat16, attn_implementation='sdpa').eval()
                    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
                from kv_research.run_eval import generation_check
                stats['generation'] = generation_check(lm, tok, raw, encoded.data,
                    decoders[name], case['first_token'], steps=8)
            row['codecs'][name] = stats
            print(f"  {name}: {stats['saving_with_state_percent']:.4f}% saved incl. state; exact", flush=True)
            del recovered, encoded
        if args.save_sample and index == 0:
            np.savez(output.parent / 'sample_raw.npz', **{f'layer_{l:02d}_{k}': w for l, pair in enumerate(raw) for k, w in zip(('K', 'V'), pair)})
        result['cases'].append(row)
        result['aggregate'] = aggregate(result['cases'], states, names)
        output.write_text(json.dumps(result, indent=2) + '\n')
    print('complete ' + str(output), flush=True)


if __name__ == '__main__':
    main()
