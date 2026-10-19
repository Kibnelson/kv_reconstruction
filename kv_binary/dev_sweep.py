"""Development-only actual-record sweep, then full sensor-cache shortlist.

No final data is read. Selection extrapolates real sampled record byte lengths
to the full1024-token development cache and adds the ACTUAL entire dictionary
and transform state. Full-cache selection is subsequently verified directly.
"""
import argparse
import hashlib
import json
import pathlib
import time
import numpy as np
from kv_binary.run_eval import factories, encode_candidate, verify_cache
from kv_research import container, learned_codec
from kv_stacked.run_eval import matrices

MODEL = 'kv_research/models/learned64.bin'
MANIFEST = '.kv_research_data/dev_manifest.json'


def controls():
    return {
        'exponent_huffman_lower_deflate': {'kind': 'huffman_lower_deflate'},
        'learned_radius': {'kind': 'learned', 'options': {'correction': 'best_entropy', 'assignment': 'radius', 'force': True}},
        'learned_radius_deflate': {'kind': 'record_deflate', 'base': 'learned_radius', 'level': 6},
        'learned_l1': {'kind': 'learned', 'options': {'correction': 'best_entropy', 'assignment': 'l1', 'force': True}},
        'e8_radius_step10': {'kind': 'learned_e8', 'options': {'step_log2': 10, 'group_size': 3072,
            'correction': 'best', 'family': 'e8', 'assignment': 'radius', 'force': True}},
        'e8_radius_step10_deflate': {'kind': 'record_deflate', 'base': 'e8_radius_step10', 'level': 6},
    }


def build_grid(output_dir, seed=104729, dimensions=(8, 32, 64), precisions=('f32',)):
    from kv_binary import binary_codec
    model = learned_codec.load_model(MODEL)
    candidates = controls()
    state_rows = []
    models = pathlib.Path('kv_binary/models')
    models.mkdir(exist_ok=True, parents=True)
    for transform in ('identity', 'dense', 'hadamard'):
        for dimension in dimensions:
            path = models / f'{transform}_d{dimension}.bin'
            state = binary_codec.make_state(model, dimension=dimension, transform=transform, seed=seed)
            binary_codec.save_state(state, path)
            for precision in precisions:
                suffix = '' if precisions == ('f32',) else '_' + precision
                name = f'binary_{transform}_d{dimension}' + suffix
                candidates[name] = dict(kind='binary', state=str(path), transform=transform, dimension=dimension,
                                       options=dict(assignment='radius', force=True, amplitude_precision=precision))
                candidates[name + '_deflate'] = dict(kind='record_deflate', base=name, level=6)
            state_rows.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   serialized_bytes=path.stat().st_size, runtime_rotation_array_bytes=dimension * dimension * 8))
    config = dict(tile_tokens=16, model=MODEL, candidates=candidates,
                  development_only=True, transform_seed=seed)
    (output_dir / 'grid_config.json').write_text(json.dumps(config, indent=2) + '\n')
    return config, state_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default=MANIFEST)
    p.add_argument('--output-dir', default='kv_binary/development')
    p.add_argument('--full-output', default='kv_binary/results/dev.json')
    p.add_argument('--seed', type=int, default=104729)
    p.add_argument('--dimensions', default='8,32,64')
    p.add_argument('--precisions', default='f32')
    p.add_argument('--skip-full', action='store_true')
    args = p.parse_args()
    path = pathlib.Path(args.manifest)
    manifest = json.loads(path.read_text())
    if manifest.get('split') != 'dev' or len(manifest['cases']) != 1:
        raise ValueError('this sweep is restricted to one development manifest case')
    case = manifest['cases'][0]
    if case['name'] != 'dev_sensor_pipeline_1024' or case['tokens'] != 1024:
        raise ValueError('only the prespecified sensor1024 development case is permitted')
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    dimensions = tuple(int(value) for value in args.dimensions.split(','))
    precisions = tuple(args.precisions.split(','))
    if not dimensions or any(d not in (8, 32, 64) for d in dimensions) or len(set(dimensions)) != len(dimensions):
        raise ValueError('invalid or duplicate dimensions')
    if not precisions or any(p not in ('f32', 'bf16') for p in precisions) or len(set(precisions)) != len(precisions):
        raise ValueError('invalid or duplicate amplitude precisions')
    config, state_rows = build_grid(output_dir, args.seed, dimensions, precisions)
    encoders, decoders, states, state_details = factories(config)
    with np.load(path.parent / case['file']) as loaded:
        full = matrices({key: loaded[key].copy() for key in loaded.files})
    full_raw = b''.join(w.astype('<u2').tobytes() for pair in full for w in pair)
    if len(full_raw) != case['raw_payload_bytes'] or hashlib.sha256(full_raw).hexdigest() != case['raw_sha256']:
        raise AssertionError('development capture identity mismatch')
    layer_map, tile_indices = [0, 15, 29], [0, 32, 63]
    sample = [tuple(np.concatenate([w[t * 16:t * 16 + 16] for t in tile_indices]) for w in full[layer]) for layer in layer_map]
    sample_records, full_records = len(layer_map) * 2 * len(tile_indices), len(full) * 2 * 64
    full_overhead = container.HEADER.size + 8 * (full_records + 1)
    sample_overhead = container.HEADER.size + 8 * (sample_records + 1)
    sample_raw_bytes = sum(w.nbytes for pair in sample for w in pair)
    record_dir = output_dir / 'records'
    record_dir.mkdir(exist_ok=True, parents=True)
    sample_path = record_dir / 'sampled_raw.npz'
    np.savez(sample_path, **{f'layer_{local:02d}_{kind}': w
        for local, pair in enumerate(sample) for kind, w in zip(('K', 'V'), pair)})
    rows, cache = {}, {}
    started = time.perf_counter()
    for name in config['candidates']:
        encoded, seconds, stage, dependency, extra = encode_candidate(name, sample, config, encoders, cache, layer_map)
        decoder = lambda b, l, k, name=name: decoders[name](b, layer_map[l], k)
        decode_seconds = verify_cache(sample, encoded, decoder)
        record_path = record_dir / (name + '.bin')
        record_path.write_bytes(encoded.data)
        record_bytes = len(encoded.data) - sample_overhead
        estimate = full_overhead + record_bytes * full_records / sample_records + states[name]
        row = dict(serialized_bytes=len(encoded.data), sampled_record_bytes=record_bytes,
            sample_raw_bytes=sample_raw_bytes, decoder_state_bytes=states[name],
            sample_payload_saving_percent=100 * (1 - len(encoded.data) / sample_raw_bytes),
            estimated_full_serialized_plus_state_bytes=estimate,
            estimated_full_saving_with_state_percent=100 * (1 - estimate / len(full_raw)),
            roundtrip_exact=True, independent_later_tile=True,
            encode_seconds=seconds, encode_stage_seconds=stage, dependency_encode_seconds=dependency,
            decode_seconds=decode_seconds, component_bytes=encoded.stats['component_bytes'],
            encoded_file=str(record_path), encoded_sha256=hashlib.sha256(encoded.data).hexdigest(),
            diagnostics=encoded.stats.get('diagnostics', {}), **extra)
        rows[name] = row
        print(f"grid {name}: sample payload {row['sample_payload_saving_percent']:.3f}%, estimated full+state {row['estimated_full_saving_with_state_percent']:.3f}%, exact", flush=True)
    chosen = {}
    for transform in ('identity', 'dense', 'hadamard'):
        candidates = [name for name, spec in config['candidates'].items()
            if spec['kind'] == 'binary' and spec['transform'] == transform]
        best = min(candidates, key=lambda name: min(rows[name]['estimated_full_serialized_plus_state_bytes'], rows[name + '_deflate']['estimated_full_serialized_plus_state_bytes']))
        chosen[transform] = best
    shortlist = controls()
    full_names = {}
    for name in chosen.values():
        spec = config['candidates'][name]
        full_name = f"binary_{spec['transform']}_d{spec['dimension']}"
        shortlist[full_name] = spec
        shortlist[full_name + '_deflate'] = dict(kind='record_deflate', base=full_name, level=6)
        full_names[spec['transform']] = full_name
    selected_config = {**config, 'candidates': shortlist}
    selected_config_path = pathlib.Path('kv_binary/dev_config.json')
    selected_config_path.write_text(json.dumps(selected_config, indent=2) + '\n')
    result = dict(scope='Development-only actual independent record grid; estimates guide full-cache shortlist and are not final measured cache savings.',
        source_manifest=str(path), source_raw_sha256=case['raw_sha256'], layer_map=layer_map,
        sampled_raw_file=str(sample_path), sampled_raw_file_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        original_tile_indices=tile_indices, sampled_records=sample_records, full_records=full_records,
        full_raw_bytes=len(full_raw), full_outer_index_bytes=full_overhead,
        state_files=state_rows, state_details=state_details, results=rows, selected_by_transform=chosen,
        selected_full_candidate_names=full_names, dimensions=list(dimensions), amplitude_precisions=list(precisions),
        payload_ranking=sorted(rows, key=lambda name: rows[name]['sampled_record_bytes']),
        complete_estimated_ranking=sorted(rows, key=lambda name: rows[name]['estimated_full_serialized_plus_state_bytes']),
        elapsed_seconds=time.perf_counter() - started)
    (output_dir / 'grid_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('selected ' + json.dumps(chosen), flush=True)
    if not args.skip_full:
        import subprocess, sys
        subprocess.run([sys.executable, '-u', '-m', 'kv_binary.run_eval', '--config', str(selected_config_path),
            '--manifest', str(path), '--output', args.full_output], check=True)


if __name__ == '__main__':
    main()
