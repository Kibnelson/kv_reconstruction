"""Actual-byte binary-residual evaluation with matched frozen controls.

The source KV is used only to verify decoded bytes. Whole-record Deflate
controls wrap the identical cached inner bytes, rather than fitting/re-encoding
a second candidate. Timing is CPU diagnostic, not a serving benchmark.
"""
import argparse
import hashlib
import json
import pathlib
import platform
import struct
import time
import zlib
import numpy as np
from kv_research import container, learned_codec
from kv_residual.run_eval import factories as control_factories
from kv_stacked import wrapper
from kv_stacked.run_eval import aggregate, matrices, equal


def factories(config):
    """Return encoders, decoders, serialized state charges, and state metadata."""
    base_config = {**config, 'candidates': {name: spec for name, spec in config['candidates'].items()
        if spec['kind'] not in ('binary', 'record_deflate')}}
    encoders, decoders, states = control_factories(base_config)
    details = {name: {'serialized_bytes': size, 'runtime_rotation_array_bytes': 0}
        for name, size in states.items()}
    models, transforms = {}, {}
    for name, spec in config['candidates'].items():
        if spec['kind'] != 'binary':
            continue
        from kv_binary import binary_codec
        model_path = pathlib.Path(spec.get('model', config.get('model', 'kv_research/models/learned64.bin')))
        state_path = pathlib.Path(spec['state'])
        model_key, transform_key = str(model_path.resolve()), str(state_path.resolve())
        if model_key not in models:
            models[model_key] = learned_codec.load_model(model_path)
        if transform_key not in transforms:
            transforms[transform_key] = binary_codec.load_state(state_path)
        model, transform = models[model_key], transforms[transform_key]
        if 'dimension' in spec and int(spec['dimension']) != transform.dimension:
            raise ValueError('configured binary dimension differs from loaded state')
        if 'transform' in spec and spec['transform'] != transform.transform:
            raise ValueError('configured transform differs from loaded state')
        options = dict(spec.get('options', {}))
        options.setdefault('assignment', 'radius')
        options.setdefault('force', True)
        encoders[name] = lambda w, l, k, model=model, transform=transform, options=options: binary_codec.encode_matrix(w, model, transform, f'layer_{l:02d}_{k}', **options)
        decoders[name] = lambda b, l, k, model=model, transform=transform: binary_codec.decode_matrix(b, model, transform)
        model_bytes, transform_bytes = model_path.stat().st_size, state_path.stat().st_size
        states[name] = model_bytes + transform_bytes
        dimension = int(transform.dimension)
        if dimension not in (8, 32, 64):
            raise ValueError('binary dimension metadata must be 8, 32, or 64')
        details[name] = dict(serialized_bytes=states[name], dictionary_bytes=model_bytes,
            transform_bytes=transform_bytes, runtime_rotation_array_bytes=transform.runtime_rotation_array_bytes,
            model_path=str(model_path), state_path=str(state_path), dimension=dimension,
            transform=spec.get('transform', getattr(transform, 'transform', 'unknown')),
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            transform_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest())
    unresolved = {name for name, spec in config['candidates'].items() if spec['kind'] == 'record_deflate'}
    while unresolved:
        progressed = False
        for name in list(unresolved):
            spec = config['candidates'][name]
            parent = spec['base']
            if parent not in decoders:
                continue
            decoder = decoders[parent]
            decoders[name] = lambda b, l, k, decoder=decoder: decoder(wrapper.unwrap(b), l, k)
            encoders[name] = None  # encode_candidate wraps exact cached bytes.
            states[name] = states[parent]
            details[name] = {**details[parent], 'wrapped_base': parent}
            unresolved.remove(name)
            progressed = True
        if not progressed:
            raise ValueError('missing/cyclic wrapped candidate dependency')
    return encoders, decoders, states, details


def encode_candidate(name, raw, config, encoders, cache, layer_map=None):
    """Return (encoded,pipeline_seconds,stage_seconds,dependency_seconds,extra)."""
    if name in cache:
        return cache[name]
    spec = config['candidates'][name]
    dependency_seconds, extra = 0., {}
    if spec['kind'] == 'record_deflate':
        base = spec['base']
        source, dependency_seconds, _, _, _ = encode_candidate(base, raw, config, encoders, cache, layer_map)
        cursor = [0]
        def encoder(w, l, k):
            index = cursor[0]
            cursor[0] += 1
            start, end = struct.unpack_from('<QQ', source.data, container.HEADER.size + index * 8)
            inner = source.data[start:end]
            encoded = wrapper.wrap(inner, spec.get('level', 6))
            if wrapper.unwrap(encoded.data) != inner:
                raise AssertionError('wrapper changed its exact cached inner record')
            return encoded
        extra = dict(wrapper_inner_byte_identity=True,
                     inner_container_sha256=hashlib.sha256(source.data).hexdigest(),
                     inner_diagnostics=source.stats.get('diagnostics', {}),
                     inner_candidate=base)
    else:
        original = encoders[name]
        encoder = original if layer_map is None else lambda w, l, k: original(w, layer_map[l], k)
    started = time.perf_counter()
    encoded = container.encode_cache(raw, encoder, tile_tokens=16)
    stage_seconds = time.perf_counter() - started
    if spec['kind'] == 'record_deflate':
        expected_records = container.shape(source.data)['nrecords']
        if cursor[0] != expected_records:
            raise AssertionError('wrapper did not visit every exact inner record')
        extra['wrapped_record_count'] = cursor[0]
    value = encoded, stage_seconds + dependency_seconds, stage_seconds, dependency_seconds, extra
    cache[name] = value
    return value


def verify_cache(raw, encoded, decoder):
    started = time.perf_counter()
    recovered = container.decode_cache(encoded.data, decoder)
    seconds = time.perf_counter() - started
    if not equal(raw, recovered):
        raise AssertionError('binary/control exact recovery failed')
    damaged = bytearray(encoded.data)
    shape = container.shape(damaged)
    damaged[shape['payload_start']] ^= 255
    index = shape['ntiles'] - 1
    later = container.decode_tile(damaged, shape['layers'] - 1, index, decoder)
    start = index * shape['tile_tokens']
    if not all(np.array_equal(a, b[start:]) for a, b in zip(later, raw[-1])):
        raise AssertionError('later tile depends on unrelated earlier payload')
    return seconds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--data-dir')
    p.add_argument('--only-index', type=int)
    p.add_argument('--generation-candidate', action='append', default=[])
    p.add_argument('--save-sample', action='store_true')
    p.add_argument('--model-dir', default='.smoke_model')
    args = p.parse_args()
    config_path, manifest_path = pathlib.Path(args.config), pathlib.Path(args.manifest)
    config = json.loads(config_path.read_text())
    if config.get('tile_tokens', 16) != 16:
        raise ValueError('this protocol requires 16-token records')
    manifest = json.loads(manifest_path.read_text())
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else manifest_path.parent
    encoders, decoders, states, state_details = factories(config)
    names = list(config['candidates'])
    if any(name not in names for name in args.generation_candidate):
        raise ValueError('generation candidate absent from config')
    output = pathlib.Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    result = dict(config=config, capture_manifest=manifest, cases=[], state_bytes=states,
        state_details=state_details, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, zlib=zlib.ZLIB_RUNTIME_VERSION, device='cpu'),
        scope='Exact CPU KV representation, independent16-token records; full learned dictionary plus serialized transform state charged. Dense runtime rotation arrays disclosed separately. No GPU, bounded attention-buffer, capacity-admission, or concurrent-serving result.')
    selected = list(enumerate(manifest['cases']))
    if args.only_index is not None:
        if not 0 <= args.only_index < len(selected):
            raise ValueError('case index outside manifest')
        selected = [selected[args.only_index]]
    lm = tok = None
    for index, case in selected:
        print('case ' + case['name'], flush=True)
        with np.load(data_dir / case['file']) as loaded:
            raw = matrices({key: loaded[key].copy() for key in loaded.files})
        raw_bytes = b''.join(w.astype('<u2').tobytes() for pair in raw for w in pair)
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        if len(raw_bytes) != case['raw_payload_bytes'] or raw_sha != case['raw_sha256']:
            raise AssertionError('capture raw length/SHA256 mismatch')
        row = {key: value for key, value in case.items() if key != 'input_token_ids'}
        row['codecs'] = {}
        cache, saved = {}, {}
        for name in names:
            encoded, seconds, stage_seconds, dependency_seconds, extra = encode_candidate(name, raw, config, encoders, cache)
            decode_seconds = verify_cache(raw, encoded, decoders[name])
            stats = {**encoded.stats, **extra, 'roundtrip_exact': True, 'independent_later_tile': True,
                'encode_seconds': seconds, 'encode_stage_seconds': stage_seconds,
                'dependency_encode_seconds': dependency_seconds, 'decode_seconds': decode_seconds,
                'encoded_sha256': hashlib.sha256(encoded.data).hexdigest(), 'decoder_state_bytes': states[name],
                'saving_with_state_percent': 100 * (1 - (len(encoded.data) + states[name]) / len(raw_bytes)),
                'savings_including_state_percent': {str(n): 100 * (1 - (len(encoded.data) + states[name] / n) / len(raw_bytes)) for n in (1, 4, 16)}}
            if args.save_sample and index == 0 and name in args.generation_candidate:
                filename = name + '_sample.bin'
                (output.parent / filename).write_bytes(encoded.data)
                saved[name] = dict(encoded_file=filename, encoded_sha256=stats['encoded_sha256'],
                    raw_file='sample_raw.npz', raw_sha256=raw_sha, kind=config['candidates'][name]['kind'],
                    raw_payload_bytes=len(raw_bytes),
                    options=config['candidates'][name].get('options', {}),
                    model_path=state_details[name].get('model_path'), state_path=state_details[name].get('state_path'),
                    model_sha256=state_details[name].get('model_sha256'),
                    state_sha256=state_details[name].get('transform_sha256'),
                    decoder_state_bytes=states[name], diagnostics=stats.get('diagnostics', {}),
                    inner_diagnostics=stats.get('inner_diagnostics', {}),
                    wrapped_record_count=stats.get('wrapped_record_count'))
                stats['sample_file'] = filename
            if index == 0 and name in args.generation_candidate:
                if lm is None:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    torch.set_num_threads(4)
                    torch.set_num_interop_threads(1)
                    lm = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                        dtype=torch.bfloat16, attn_implementation='sdpa').eval()
                    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
                from kv_research.run_eval import generation_check
                stats['generation'] = generation_check(lm, tok, raw, encoded.data, decoders[name], case['first_token'], steps=8)
            row['codecs'][name] = stats
            print(f"  {name}: {stats['saving_with_state_percent']:.4f}% saved incl. state; exact", flush=True)
        if saved:
            raw_path = output.parent / 'sample_raw.npz'
            np.savez(raw_path, **{f'layer_{l:02d}_{kind}': w for l, pair in enumerate(raw) for kind, w in zip(('K', 'V'), pair)})
            file_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            for sample in saved.values():
                sample['raw_file_sha256'] = file_sha
            (output.parent / 'sample_metadata.json').write_text(json.dumps(dict(config=config, samples=saved), indent=2) + '\n')
        result['cases'].append(row)
        result['aggregate'] = aggregate(result['cases'], states, names)
        output.write_text(json.dumps(result, indent=2) + '\n')
    print('complete ' + str(output), flush=True)


if __name__ == '__main__':
    main()
