"""Freeze the requested codec comparisons after development, before final KV."""
import datetime
import hashlib
import json
import pathlib

def main():
    root=pathlib.Path('kv_residual');out=root/'results'
    assert not pathlib.Path('.kv_residual_data/final_manifest.json').exists()
    main_dev=json.loads((out/'dev.json').read_text())
    residual_dev=json.loads((out/'residual_dev.json').read_text())
    assert all(v['all_exact'] for run in (main_dev,residual_dev) for v in run['aggregate'].values())
    config={'tile_tokens':16,'model':'kv_research/models/learned64.bin','candidates':{}}
    for run in (main_dev,residual_dev):config['candidates'].update(run['config']['candidates'])
    assert len(config['candidates'])==12
    selection_e8=json.loads((root/'e8_selection.json').read_text())
    for name,spec in (('learned_e8_entropy','e8'),('learned_scalar_entropy','scalar'),('learned_e8_coset','e8_coset')):
        for key in ('step_log2','family','correction'):
            assert config['candidates'][name]['options'][key]==selection_e8[spec][key]
    score=lambda name:main_dev['aggregate'][name]['savings_including_state_percent']['1']
    learned_choice=max(('learned_entropy_radius','learned_entropy_l1'),key=score)
    (root/'final_config.json').write_text(json.dumps(config,indent=2)+'\n')
    sources=list(root.glob('*.py'))+[root/'final_config.json',root/'e8_selection.json']
    dependencies=['kv_research/models/learned64.bin','kv_research/learned_codec.py','kv_research/e8_codec.py',
      'kv_research/container.py','kv_research/native_decode.py','kv_research/native_decode.cpp','kv_research/model_utils.py',
      'kv_research/run_eval.py','kv_research/fixtures.py','kv_smoke/codec.py','kv_smoke/prompts.py','kv_smoke/download_model.py',
      'kv_stacked/run_eval.py','kv_stacked/huffman_codec.py','kv_stacked/field_codec.py','kv_stacked/wrapper.py',
      'kv_coset_fields/combined.py','kv_coset_fields/controls.py','.smoke_model/download_manifest.json']
    sources.extend(pathlib.Path(p) for p in dependencies)
    evidence=[out/'dev.json',out/'residual_dev.json',out/'dev_e8_sweep.json',out/'unit_tests.txt']
    sha=lambda paths:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    selection=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      generation_candidates=[learned_choice,'learned_e8_entropy'],best_learned_correction=learned_choice,
      reason='Fixed learned64 dictionary. Keep matched radius controls and L1 assignment separately; E8/scalar scales selected from the same development-only grid. Preserve forced actual learned/E8 records even if larger.',
      fresh_final_manifest_existed=False,source_and_model_sha256=sha(sources),development_evidence_sha256=sha(evidence))
    (root/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps({'generation_candidates':selection['generation_candidates'],'candidates':len(config['candidates']),'frozen_files':len(sources)},indent=2))

if __name__=='__main__':main()
