"""Merge unchanged development controls with the final binary precision rerun.

This only reuses records from the same prespecified development capture. Every
final-evaluation case still executes all candidates afresh after freezing.
"""
import argparse
import hashlib
import json
import pathlib
from kv_binary.run_eval import factories
from kv_stacked.run_eval import aggregate


CONTROL_SOURCES = (
    'kv_coset_fields/combined.py', 'kv_coset_fields/controls.py',
    'kv_research/container.py', 'kv_research/e8_codec.py',
    'kv_research/learned_codec.py', 'kv_research/models/learned64.bin',
    'kv_residual/entropy.py', 'kv_residual/learned.py',
    'kv_residual/learned_e8.py', 'kv_residual/run_eval.py',
    'kv_stacked/huffman_codec.py', 'kv_stacked/wrapper.py', 'kv_stacked/run_eval.py',
)


def binary(name, specs):
    spec = specs[name]
    return spec['kind'] == 'binary' or (spec['kind'] == 'record_deflate' and binary(spec['base'], specs))


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', default='kv_binary/results/dev_f32.json')
    p.add_argument('--binary', default='kv_binary/results/dev_precision_selected.json')
    p.add_argument('--config', default='kv_binary/dev_config.json')
    p.add_argument('--output', default='kv_binary/results/dev.json')
    p.add_argument('--old-freeze', default='kv_residual/selection.json')
    args = p.parse_args()
    old = json.loads(pathlib.Path(args.original).read_text())
    new = json.loads(pathlib.Path(args.binary).read_text())
    config = json.loads(pathlib.Path(args.config).read_text())
    if old['capture_manifest'] != new['capture_manifest'] or old['capture_manifest'].get('split') != 'dev':
        raise AssertionError('development capture manifest differs')
    if len(old['cases']) != 1 or len(new['cases']) != 1:
        raise AssertionError('only the one prespecified development case may be merged')
    old_case, new_case = old['cases'][0], new['cases'][0]
    for key in ('name', 'raw_sha256', 'raw_payload_bytes'):
        if old_case[key] != new_case[key]:
            raise AssertionError('development case identity differs: ' + key)
    if old_case['name'] != 'dev_sensor_pipeline_1024':
        raise AssertionError('unexpected development fixture')
    _, _, states, state_details = factories(config)
    source_freeze = json.loads(pathlib.Path(args.old_freeze).read_text())['source_and_model_sha256']
    source_hashes = {}
    for path in CONTROL_SOURCES:
        digest = sha(path)
        if digest != source_freeze[path]:
            raise AssertionError('unchanged control dependency differs from prior freeze: ' + path)
        source_hashes[path] = digest
    controls, binaries, codecs = [], [], {}
    for name, spec in config['candidates'].items():
        use_binary = binary(name, config['candidates'])
        source = new if use_binary else old
        source_case = new_case if use_binary else old_case
        if source['config']['candidates'][name] != spec or source['state_bytes'][name] != states[name]:
            raise AssertionError('candidate specification/state charge changed: ' + name)
        stats = source_case['codecs'][name]
        if not stats['roundtrip_exact'] or not stats['independent_later_tile']:
            raise AssertionError('source verification failed')
        if stats['raw_payload_bytes'] != old_case['raw_payload_bytes']:
            raise AssertionError('source raw footprint differs')
        (binaries if use_binary else controls).append(name)
        codecs[name] = stats
    if len(controls) != 6 or len(binaries) != 6:
        raise AssertionError('expected six unchanged controls plus six selected binary arms')
    result = {**new, 'config': config, 'config_sha256': sha(args.config),
        'state_bytes': states, 'state_details': state_details,
        'cases': [{**old_case, 'codecs': codecs}]}
    result['aggregate'] = aggregate(result['cases'], states, list(config['candidates']))
    result['reused_control_provenance'] = dict(
        note='Only unchanged controls on identical development raw bytes were reused. All final cases must execute every candidate anew.',
        control_names=controls, binary_names=binaries,
        original_evidence_file=args.original, original_evidence_sha256=sha(args.original),
        binary_rerun_evidence_file=args.binary, binary_rerun_evidence_sha256=sha(args.binary),
        prior_control_freeze=args.old_freeze, prior_control_freeze_sha256=sha(args.old_freeze),
        unchanged_control_source_sha256=source_hashes)
    output = pathlib.Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('merged verified development controls and binary precision rerun: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
