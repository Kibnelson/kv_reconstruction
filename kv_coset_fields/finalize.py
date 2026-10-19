"""Validate frozen results and package a reproducible combination experiment."""
from collections import Counter
import datetime
import hashlib
import json
import pathlib
import shutil
import zipfile

def main():
    root=pathlib.Path(__file__).resolve().parent;project=root.parent;out=root/'results'
    selection=json.loads((root/'selection.json').read_text())
    config=json.loads((root/'final_config.json').read_text())
    result=json.loads((out/'final.json').read_text())
    attribution=json.loads((out/'posthoc_attribution.json').read_text())
    assert attribution['source_final_sha256']==hashlib.sha256((out/'final.json').read_bytes()).hexdigest()
    assert result['config']==config and len(result['cases'])==6
    assert len({c['name'] for c in result['cases']})==6
    for p,digest in {**selection['frozen_sha256'],**selection['evidence_sha256']}.items():
        assert hashlib.sha256((project/p).read_bytes()).hexdigest()==digest,'changed frozen source/evidence '+p
    assert not selection['fresh_final_manifest_existed']
    names=list(config['candidates']);forced=selection['generation_candidate']
    assert len(names)==8
    records=0;counts={name:Counter() for name in names};widths=Counter()
    for case,source in zip(result['cases'],result['capture_manifest']['cases']):
        assert case['name']==source['name'] and case['raw_sha256']==source['raw_sha256']
        assert case['raw_payload_bytes']==source['raw_payload_bytes'] and set(case['codecs'])==set(names)
        expected=60*((case['tokens']+config['tile_tokens']-1)//config['tile_tokens']);records+=expected
        for name,stats in case['codecs'].items():
            assert stats['roundtrip_exact'] and stats['independent_later_tile']
            assert sum(stats['component_bytes'].values())==stats['serialized_bytes']
            if name!='exponent_huffman':assert stats['diagnostics']['audit']['records_verified']==expected
            counts[name].update(stats.get('diagnostics',{}).get('record_modes',{}))
        actual=case['codecs'][forced]
        assert actual['diagnostics']['record_modes']=={'combined_lower':expected}
        assert actual['component_bytes'].get('matrix_lower_raw_lower_vectors',0)==0
        widths.update(actual['diagnostics']['lower_candidate_correction_width_histogram'])
    assert records==28800 and sum(widths.values())==11059200
    generation=result['cases'][0]['codecs'][forced]['generation']
    assert generation['steps']==8 and generation['all_step_logits_bitwise_equal'] and generation['max_abs_logit_difference']==0
    unitlog=(out/'unit_tests.txt').read_text();assert 'Ran 20 tests' in unitlog and '\nOK\n' in unitlog
    first=result['cases'][0];sample=(out/'forced_sample.bin').read_bytes()
    assert hashlib.sha256(sample).hexdigest()==first['codecs'][forced]['encoded_sha256']
    shutil.copyfile(project/'.kv_coset_fields_data'/first['file'],out/'sample_raw.npz')
    model=root/'models/lower16.bin'
    info={k:first[k] for k in ('name','tokens','raw_payload_bytes','raw_sha256')}
    info.update(encoded_sha256=first['codecs'][forced]['encoded_sha256'],model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                candidate=forced,all_records_use_actual_cosets=True)
    (out/'sample_metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    shutil.copyfile(project/'.kv_coset_fields_data/final_manifest.json',out/'fresh_capture_manifest.json')
    shutil.copyfile(project/'.smoke_model/download_manifest.json',out/'checkpoint_manifest.json')
    shutil.copyfile(project/'.kv_research_data/train_manifest.json',out/'training_capture_manifest.json')
    shutil.copyfile(project/'.kv_research_data/dev_manifest.json',out/'development_capture_manifest.json')
    aggregate=result['aggregate'];saving=lambda name:aggregate[name]['savings_including_state_percent']['1']
    baseline=saving('exponent_huffman');best=max(names,key=saving);seven=100*widths['7']/sum(widths.values())
    mean_width=sum(int(k)*v for k,v in widths.items())/sum(widths.values())
    descriptions={'exponent_huffman':'Exact exponent Huffman; sign/fraction literal',
      'lower_deflate':'Same exponent coding + lower-byte Deflate control',
      'split_deflate':'Same exponent coding + packed sign/fraction Deflate control',
      'temporal_previous':'Previous-token predictor + modular lower-byte cosets',
      'temporal_exponent_aware':'Exponent-aware previous-token predictor + cosets',
      'learned16':'16 learned fraction prototypes + exact signs/cosets',
      'learned16_conditional':'16 prototypes conditioned on known exponent + cosets',
      'learned64':'64 learned fraction prototypes + exact signs/cosets',
      'learned64_conditional':'64 prototypes conditioned on known exponent + cosets',
      'learned16_forced_cosets':'Forced 16-prototype cosets; expansion allowed diagnostic'}
    lines=['# Test of exponent Huffman plus predictor/coset reconstruction','',
      f"The baseline saved **{baseline:.3f}%**. The strongest final candidate was **{best}**, saving **{saving(best):.3f}%** including required state. The difference is **{saving(best)-baseline:.3f} percentage points**.",'',
      f"The candidate selected on development before fresh capture was **{selection['best_development_candidate']}**; its final saving was **{saving(selection['best_development_candidate']):.3f}%**. Maximum-over-final comparisons above are descriptive; no weights or settings were changed after final capture.",'',
      '## What was actually combined','',
      'BF16 has one sign bit, eight exponent bits and seven fraction bits. Every candidate stores the original exponent stream using the same exact local canonical Huffman code. The remaining lower byte is sign in bit 7 plus the seven fraction bits. Temporal candidates predict this byte from earlier decoded values within the same tile. Learned candidates predict eight fraction symbols using a frozen dictionary, optionally shifted by a frozen mean table indexed by the already-decoded exponent. Signs remain exact.','',
      'A simple modular coset label is s = x mod 2^b. The decoder chooses the representative nearest its prediction on the appropriate symbol circle: modulo 256 for lower bytes or modulo 128 for fractions. The encoder uses the smallest reconstructible width or a certified circular radius. Width zero reproduces the prediction; full symbol width retains every original symbol bit. These are coordinatewise modular codes, not advanced cross-coordinate error-correcting syndromes.','',
      'Adaptive versions retain the original exponent-Huffman record byte-for-byte whenever the complete combined record is not smaller. Each trial lower record is still decoded and checked before that decision. Every learned model file is charged even if every record falls back. The forced diagnostic stores all prototype IDs, width metadata, exact signs and modular residues despite expansion, exercising actual reconstruction on real KV.','',
      '## Fresh final results','',
      'Settings and all predictor weights were frozen before capturing six fresh cases: puppet-theatre prose, JavaScript run-length encoding and chess-league records at 512 and 2,048 tokens. Length variants share filler, so they are correlated synthetic coverage rather than representative serving or task-quality benchmarks. The checkpoint is pretrained native-BF16 SmolLM2-135M. All methods use the same independent 16-token records and temporal index.','',
      '| Candidate | Saving vs raw BF16, state included | Selected new lower records / total | Model bytes per cache |',
      '|---|---:|---:|---:|']
    for name in names:
        accepted=counts[name]['combined_lower']
        value='baseline' if name=='exponent_huffman' else f'{accepted:,} / {records:,}'
        lines.append(f"| {name} | {saving(name):.3f}% | {value} | {result['state_bytes'][name]:,} |")
    lines+=['','Names:']+[f'- **{n}:** {descriptions[n]}.' for n in names]
    lines+=['',
      'Savings are total raw bytes minus total stored bytes, divided by total raw bytes. Every record header, local table, checksum, packed padding, predictor ID, width, residue, stream boundary and cache index is included. Learned state is charged separately to each evaluated cache at N=1; N=4/16 JSON scenarios are arithmetic sharing only. Records never use history from another record.','',
      '## What the coset test establishes','',
      f"The forced representation saved **{saving(forced):.3f}%**, versus **{baseline:.3f}%** for the baseline. It used actual cosets in all **{records:,}** records and **{sum(widths.values()):,}** eight-value vectors.",'',
      f"**{seven:.2f}%** of forced vectors required width 7. Those residues contain all seven original fraction bits, so prediction infers no missing fraction information there. Other vectors use fewer correction bits, but predictor IDs, width metadata, vector selection, exact signs, local headers and external models must also be paid. The complete-byte results determine whether the combination helps; correctness alone does not establish compression benefit.",'',
      f"The mean forced fraction residue width was **{mean_width:.4f} bits/value**, versus seven literal fraction bits. Before headers, the 16-prototype format also pays four prototype-ID bits, three width bits, one selection-bitmap bit and eight sign bits per eight-value vector. Its average is therefore **{mean_width+2:.4f} bits/value** for the lower fields, versus eight literal bits. Rare narrower residues do not offset that metadata in this experiment.",'',
      '| Stored forced correction width | Eight-value vectors |','|---|---:|']
    for width,count in sorted(widths.items(),key=lambda kv:int(kv[0])):lines.append(f'| {width} | {count:,} |')
    lines+=['',
      'The tested adaptive coset configurations can keep the baseline locally, while learned variants still pay global model state. An entropy-only gain must not be attributed to cosets. These results concern these small predictors on this checkpoint; they do not prove that every conditional predictor, lattice, learned representation or cross-dimensional code must fail.','',
      '## Development and validation','',
      '- Training used only three prior 1,024-token captures (railway, Python graph traversal and catalogue). Four learned dictionaries were fit: 16/64 prototypes with and without exponent conditioning. Development used the separate sensor prompt; the best N=1 model in each conditioning family was chosen before final capture.',
      '- Five temporal predictors were checked on development records. The main final comparison retains the plain previous-token and exponent-aware hypotheses. Lower-byte Huffman was also rejected on development; Deflate and the split-layout control were kept.',
      f'- All {len(names)*len(result["cases"])} fresh full-cache roundtrips were bit-exact. Every nonbaseline trial record, including rejected candidates, passed its lower-field roundtrip audit. Later tiles remained independently decodable after damage to an unrelated first payload.',
      '- Twenty unit tests passed, including all BF16 patterns, all lower symbols, circular wraparound, tails, noncontiguous arrays, malformed streams, checksums, model identity, forced expansion, rejection auditing and complete byte accounting.',
      '- The forced-coset candidate passed eight successive model decode steps with identical logits, generated tokens and appended KV. This check restores the complete prefix on CPU and does not implement compact-resident attention.',
      '- The bundled real sample uses actual forced cosets in every record and is verifiable with NumPy alone.','',
      '## Attribution of the small entropy-only gain','',
      f"Posthoc inspection of the frozen final byte counts found that **{100*attribution['methods']['lower_deflate']['layer0_V_share_of_net_saving']:.2f}%** of lower-byte Deflate savings came from layer 0 V. Later layers contributed **{attribution['methods']['lower_deflate']['later_layers_saved_bytes']:,} bytes** of that method\'s saving. This is a descriptive per-layer attribution, not a causal ablation or new parameter selection. See posthoc_attribution.json.",'',
      '## Performance and scope limits','',
      'This is a CPU storage/recovery experiment. It does not establish cheap GPU decoding, lower TPOT/TTFT, bounded GPU workspace, higher concurrency, or PCIe/SSD transfer benefits. Python/reference decoder allocations and attention memory are not represented by serialized byte counts. Three independent CPU jobs overlap and contend; encode wall time additionally includes trial decode audits. These diagnostic wall times are not clean production codec or serving benchmarks.','',
      'The model check reuses a fully reconstructed cache. Reconstruction buffering, GPU execution, and admission under long-context concurrency remain separate implementation work. Further predictor tuning should use development data and another fresh final split.','',
      '## Reproduce','',
      'See ../README.md. final.json includes full per-case bytes, state charges, diagnostic trial and accepted mode counts, hashes and model checks. selection.json contains source/model/evidence hashes fixed before capture. fresh_capture_manifest.json records exact token IDs and raw KV hashes. Run python -m kv_coset_fields.verify_sample to reconstruct the bundled actual-coset sample without model weights.','']
    (out/'review.md').write_text('\n'.join(lines))
    dependencies=['kv_research/__init__.py','kv_research/container.py','kv_research/learned_codec.py','kv_research/native_decode.py','kv_research/native_decode.cpp',
      'kv_research/model_utils.py','kv_research/run_eval.py','kv_research/e8_codec.py','kv_research/fixtures.py','kv_research/capture.py',
      'kv_smoke/codec.py','kv_smoke/prompts.py','kv_smoke/download_model.py','kv_smoke/requirements.txt',
      'kv_stacked/__init__.py','kv_stacked/huffman_codec.py','kv_stacked/wrapper.py','kv_stacked/field_codec.py','kv_stacked/run_eval.py']
    files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='package_manifest.json']
    files.extend(project/p for p in dependencies)
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      files=[dict(path=str(p.relative_to(project)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)])
    mpath=root/'package_manifest.json';mpath.write_text(json.dumps(manifest,indent=2)+'\n');files.append(mpath)
    archive=project/'kv_exponent_coset_combination_tests.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(project)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,baseline_saving=baseline,
      best=best,best_saving=saving(best),forced_coset_saving=saving(forced),width7_percent=seven),indent=2))

if __name__=='__main__':main()
