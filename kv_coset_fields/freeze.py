"""Record development-only choices before capturing this experiment's final KV."""
import datetime
import hashlib
import json
import pathlib

def main():
    root=pathlib.Path('kv_coset_fields');out=root/'results'
    assert not pathlib.Path('.kv_coset_fields_data/final_manifest.json').exists()
    dev=json.loads((out/'dev.json').read_text());forced=json.loads((out/'forced_dev.json').read_text())
    assert all(v['all_exact'] for v in dev['aggregate'].values())
    assert all(v['all_exact'] for v in forced['aggregate'].values())
    score=lambda name:dev['aggregate'][name]['savings_including_state_percent']['1']
    uncond=max(('learned16','learned64'),key=score)
    cond=max(('learned16_conditional','learned64_conditional'),key=score)
    names=['exponent_huffman','lower_deflate','split_deflate','temporal_previous','temporal_exponent_aware',uncond,cond]
    config=dict(tile_tokens=16,candidates={name:dev['config']['candidates'][name] for name in names})
    forced_name='learned16_forced_cosets'
    config['candidates'][forced_name]=forced['config']['candidates'][forced_name]
    (root/'final_config.json').write_text(json.dumps(config,indent=2)+'\n')
    sources=list(root.glob('*.py'))+list((root/'models').glob('*.bin'))+[root/'final_config.json']
    dependencies=['kv_stacked/huffman_codec.py','kv_stacked/wrapper.py','kv_stacked/field_codec.py','kv_stacked/run_eval.py',
      'kv_research/container.py','kv_research/learned_codec.py','kv_research/native_decode.py','kv_research/native_decode.cpp',
      'kv_research/model_utils.py','kv_research/run_eval.py','kv_research/e8_codec.py','kv_research/fixtures.py',
      'kv_smoke/codec.py','kv_smoke/prompts.py','kv_smoke/download_model.py','.smoke_model/download_manifest.json']
    sources.extend(pathlib.Path(p) for p in dependencies)
    evidence=[out/'dev.json',out/'forced_dev.json',out/'controls_dev.json',out/'unit_tests.txt',
      root/'dev_temporal.json',root/'models/learned_lower_training_report.json']
    sha=lambda paths:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    selection=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      generation_candidate=forced_name,best_development_candidate=max(names,key=score),
      learned_choices={'unconditional':uncond,'exponent_conditional':cond},
      reason='Keep both temporal hypotheses and entropy controls; select the best N=1 learned model in each conditioning family on development. Forced cosets are an expansion-allowed mechanism diagnostic, not the selected compressor.',
      fresh_final_manifest_existed=False,frozen_sha256=sha(sources),evidence_sha256=sha(evidence))
    (root/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps({k:selection[k] for k in ('best_development_candidate','learned_choices','generation_candidate')},indent=2))

if __name__=='__main__':main()
