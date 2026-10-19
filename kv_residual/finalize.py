"""Gate the frozen fresh experiment and package measured code and evidence."""
from collections import Counter
import datetime
import hashlib
import json
import pathlib
import shutil
import zipfile

def main():
    root=pathlib.Path(__file__).resolve().parent;project=root.parent;out=root/'results'
    selected=json.loads((root/'selection.json').read_text())
    result=json.loads((out/'final.json').read_text())
    extension=json.loads((out/'wrapped_extension.json').read_text())
    extension_selection=json.loads((root/'wrapped_extension_selection.json').read_text())
    assert extension['all_nine_exact'] and extension['source_config_model_unchanged_after_freeze']
    assert extension['selection_sha256']==hashlib.sha256((root/'wrapped_extension_selection.json').read_bytes()).hexdigest()
    assert extension_selection['config_sha256']==hashlib.sha256((root/'wrapped_extension_config.json').read_bytes()).hexdigest()
    assert extension['manifest_sha256']==hashlib.sha256((root/'wrapped_extension_manifest.json').read_bytes()).hexdigest()
    assert len(extension['codecs'])==9
    for p,digest in extension_selection['source_sha256'].items():
        assert hashlib.sha256((project/p).read_bytes()).hexdigest()==digest,'changed extension source '+p
    for n,s in extension['codecs'].items():
        assert s['roundtrip_exact'] and s['independent_later_tile']
        assert sum(s['component_bytes'].values())==s['serialized_bytes']
        assert s['shared_model_bytes']==(0 if n=='huffman_lower_deflate' else 70095)
        if n.endswith('_wrapped'):assert s['wrapped_inner_records_verified']==7680
    config=json.loads((root/'final_config.json').read_text())
    assert result['config']==config and len(result['cases'])==6
    assert len({c['name'] for c in result['cases']})==6 and len(config['candidates'])==12
    for p,digest in {**selected['source_and_model_sha256'],**selected['development_evidence_sha256']}.items():
        assert hashlib.sha256((project/p).read_bytes()).hexdigest()==digest,'changed frozen source/evidence '+p
    assert not selected['fresh_final_manifest_existed']
    assert 'Ran 20 tests' in (out/'unit_tests.txt').read_text() and '\nOK\n' in (out/'unit_tests.txt').read_text()
    names=list(config['candidates']);mechanisms={n:Counter() for n in names};backends={n:Counter() for n in names};parts={n:Counter() for n in names}
    total_records=0
    for case,source in zip(result['cases'],result['capture_manifest']['cases']):
        assert case['name']==source['name'] and case['raw_sha256']==source['raw_sha256'] and case['raw_payload_bytes']==source['raw_payload_bytes']
        assert set(case['codecs'])==set(names)
        expected_records=60*((case['tokens']+15)//16);expected_vectors=case['raw_payload_bytes']//16
        total_records+=expected_records
        for name,stats in case['codecs'].items():
            assert stats['roundtrip_exact'] and stats['independent_later_tile']
            assert sum(stats['component_bytes'].values())==stats['serialized_bytes']
            d=stats.get('diagnostics',{});m=d.get('mechanism_counts',{})
            mechanisms[name].update(m);parts[name].update(stats['component_bytes'])
            backends[name].update(d.get('correction_backend_histogram',d.get('accepted_correction_mode_histogram',{})))
            spec=config['candidates'][name]
            if spec['kind']=='learned':assert m['stored_learned_vectors']==expected_vectors
            if spec['kind']=='learned_e8':
                assert m['learned_vectors']==expected_vectors
                assert stats['component_bytes'].get('matrix_raw_words',0)==0
                if spec['options']['family']=='e8':assert m['e8_vectors']==expected_vectors and m['e8_groups']==expected_records
                else:assert m['scalar_groups']==expected_records
    assert total_records==28800
    for name in selected['generation_candidates']:
        gen=result['cases'][0]['codecs'][name]['generation']
        assert gen['steps']==8 and gen['all_step_logits_bitwise_equal'] and gen['max_abs_logit_difference']==0
    e8=mechanisms['learned_e8_entropy'];assert e8['e8_vectors']==11059200 and e8['nonzero_e8_vectors']>0 and e8['half_integer_e8_vectors']>0
    first=result['cases'][0]
    model=project/config['model'];samples=[]
    for name in selected['generation_candidates']:
        stats=first['codecs'][name];file=stats['sample_file'];blob=(out/file).read_bytes()
        assert hashlib.sha256(blob).hexdigest()==stats['encoded_sha256']
        samples.append(dict(candidate=name,kind=config['candidates'][name]['kind'],file=file,encoded_sha256=stats['encoded_sha256']))
    with __import__('numpy').load(out/'sample_raw.npz') as raw:
        words=b''.join(raw[f'layer_{l:02d}_{k}'].astype('<u2').tobytes() for l in range(30) for k in ('K','V'))
    assert len(words)==first['raw_payload_bytes'] and hashlib.sha256(words).hexdigest()==first['raw_sha256']
    info={k:first[k] for k in ('name','tokens','raw_payload_bytes','raw_sha256')}
    info.update(model_file=config['model'],model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),samples=samples)
    (out/'sample_metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    shutil.copyfile(project/'.kv_residual_data/final_manifest.json',out/'fresh_capture_manifest.json')
    shutil.copyfile(project/'.smoke_model/download_manifest.json',out/'checkpoint_manifest.json')
    shutil.copyfile(project/'.kv_research_data/train_manifest.json',out/'training_capture_manifest.json')
    shutil.copyfile(project/'.kv_research_data/dev_manifest.json',out/'development_capture_manifest.json')
    aggregate=result['aggregate'];save=lambda n:aggregate[n]['savings_including_state_percent']['1']
    best_learned=selected['best_learned_correction'];best=max(names,key=save)
    radius_coset=save('learned_coset_radius');radius_entropy=save('learned_entropy_radius')
    coord=parts['learned_e8_entropy']['matrix_e8_coordinates_and_coset_flags']
    correction_bytes=lambda n:sum(v for k,v in parts[n].items() if k.startswith('matrix_correction_'))
    remaining_saved=correction_bytes('learned_entropy_radius')-correction_bytes('learned_e8_entropy')
    correction_change=('smaller by ' if remaining_saved>=0 else 'larger by ')+f'{abs(remaining_saved):,} bytes'
    backend_summary=', '.join(f'{k}: {v:,} records' for k,v in sorted(backends[best_learned].items()))
    nonzero=100*e8['nonzero_e8_vectors']/e8['e8_vectors'];half=100*e8['half_integer_e8_vectors']/e8['e8_vectors']
    labels={'exponent_huffman':'Exact exponent Huffman baseline',
      'exponent_huffman_lower_deflate':'Strong baseline + lower-byte Deflate',
      'direct_entropy':'Direct magnitude/sign coding; no learned state',
      'original_learned':'Previous learned dictionary + cosets, old framing',
      'original_learned_deflate':'Previous complete learned records → Deflate',
      'learned_coset_radius':'New framing; same radius predictor + cosets',
      'learned_entropy_radius':'Same radius predictor + improved correction coding',
      'learned_coset_l1':'L1 prototype assignment + coset control',
      'learned_entropy_l1':'L1 prototype assignment + improved corrections',
      'learned_e8_entropy':'Radius predictor → E8 residual → improved corrections',
      'learned_scalar_entropy':'Radius predictor → scalar split → improved corrections',
      'learned_e8_coset':'Radius predictor → E8 residual → exact cosets'}
    lines=['# Learned prediction with better corrections and E8 residuals','',
      f"Both requested combinations were implemented and tested with exact recovery. The development-selected learned correction candidate saved **{save(best_learned):.3f}%**; learned plus E8 residual prediction saved **{save('learned_e8_entropy'):.3f}%**. The strongest final result was **{best}**, at **{save(best):.3f}%**. All percentages include required model state.",'',
      '## Exact flow','',
      'Let m be the original 15-bit BF16 magnitude symbol and c the frozen learned prototype selected for its eight-value vector. Store the sign separately. The basic improved method stores r=m−c with a lossless signed-integer coder and recovers m=c+r. This full magnitude includes exponent and fraction; no separate exponent stream is sent in these learned arms.','',
      'The E8 method computes q=nearest_E8((m−c)/S), stores q and S, forms p=clip(c+S·q,0,32767), then stores the exact remaining correction e=m−p. Decode computes m=p+e. The selected S is 1,024 symbol units (step_log2=10). Twice-coordinates are integers, so scale application is exact. This lattice operates in reversible BF16 symbol space, not floating-point KV distortion space.','',
      'A matched scalar control uses ordinary coordinate rounding at the same scale and the same correction backend. Every E8 coordinate, parity/coset flag, scale, prototype ID, sign, remaining correction, checksum and header counts. E8 does not remove the information represented by the correction; its description must earn its own byte cost.','',
      '## Fresh final comparison','',
      'Three new topics (bakery prose, Ruby leap-year code and swimming records) were captured at 512 and 2,048 tokens only after freezing code, dictionary, development evidence and settings. These are six correlated synthetic coverage cases on pinned native-BF16 SmolLM2-135M, not representative serving benchmarks. All methods use independently decodable 16-token layer/K-or-V records and the same temporal index.','',
      '| Candidate | Description | Saving including full state |','|---|---|---:|']
    for n in names:lines.append(f'| {n} | {labels[n]} | {save(n):.3f}% |')
    lines+=['',
      'Ratios are weighted by raw byte count. Every learned candidate pays the complete70,095-byte frozen dictionary separately for every evaluated cache at N=1. N=4/16 JSON scenarios are arithmetic sharing, not measured concurrency. Direct and exponent controls have no external learned state. Fixed LLM weights are common to every arm and are not KV payload.','',
      '## Attribute each improvement','',
      f"- **Framing and fallback policy:** previous learned cosets saved {save('original_learned'):.3f}%; the new format with the same radius-selected prototypes and cosets saved {radius_coset:.3f}%. The old format pays a header for every64-value internal group and permits group-level raw fallback. The new format amortizes correction framing across the same16-token access record and forces learned records. This comparison combines format and fallback changes; it is not an isolated predictor or entropy improvement.",
      f"- **Correction coding, fixed predictor:** radius-selected prototypes plus improved corrections saved {radius_entropy:.3f}%, versus {radius_coset:.3f}% with cosets: **{radius_entropy-radius_coset:.3f} percentage points** from the matched coding change.",
      f"- **Prototype assignment:** L1 selection plus improved corrections saved {save('learned_entropy_l1'):.3f}%, versus {radius_entropy:.3f}% under radius selection. Both reuse the same frozen dictionary; this additional difference changes which prototype IDs are stored.",
      f"- **Prediction contribution:** direct magnitude/sign coding with the same entropy backend saved {save('direct_entropy'):.3f}%. Compare that with the complete learned result, including its IDs and model.",
      f"- **E8 contribution:** learned+E8 with improved corrections saved {save('learned_e8_entropy'):.3f}%; its matched scalar split saved {save('learned_scalar_entropy'):.3f}%. Neither should be credited with savings already achieved by learned-only correction coding.",'',
      'Improved correction coding chooses the smallest complete stream from ZigZag–Rice, ZigZag byte-plane Deflate, centered packed/Deflate, ZigZag packed and signed literal alternatives. Rice parameters and all other local adaptation are stored; there is no hidden per-request model. The plain learned coset controls use the same predictions and store exact modular labels. Width15 means the full magnitude, fixing the half-modulus boundary and preserving every BF16 word.','',
      f'Stored correction backends for the selected learned candidate: {backend_summary}.','',
      '## What E8 actually stored','',
      f"The selected E8 entropy arm stored actual lattice descriptions in **{e8['e8_groups']:,}** records and **{e8['e8_vectors']:,}** vectors. **{nonzero:.3f}%** of points were nonzero and **{half:.3f}%** used the half-integer coset. The coarse selected scale therefore predicts the origin for many vectors; actual branch execution alone does not establish useful lattice prediction.",'',
      f"Stored E8 coordinates/flags cost **{coord:,} bytes**. Relative to matched learned-only radius coding, the remaining correction stream was **{correction_change}**. The complete totals above also include residual metadata and framing. This reveals whether the smaller remainder pays for describing the lattice point.",'',
      'E8 coordinates are fixed-bit packed within each access record, not entropy-coded. The tested scale grid was4/7/10, not exhaustive. These prototypes do not reproduce a published lattice quantizer or establish an optimal E8 design.','',
      '## Separate fresh packing refinement','',
      'A development-only probe found substantial redundancy in the complete serialized E8 records. Whole-record Deflate can compress coordinates, metadata and corrections, so this does not isolate coordinate coding alone. We then froze a separate extension before capturing one additional original 2,048-token urban-beekeeping prompt. The same four inner methods were tested with and without independent level-6 record wrapping, plus the strong baseline. The following single-case numbers include the complete index and model state; they are not pooled into the six-case averages above.','',
      '| Additional fresh-case candidate | Saving including state |','|---|---:|']
    for n,s in extension['codecs'].items():lines.append(f"| {n} | {s['savings_with_model_n1_percent']:.3f}% |")
    lines+=['',
      'The wrapper operates on complete inner matrix records and carries its full13-byte header even on literal fallback. Every wrapped record was verified to reproduce its byte-identical inner record, then the entire KV and independent later tile were checked. This separates packaging overhead from residual prediction. See wrapped_extension_config.json, wrapped_extension_selection.json, wrapped_extension_manifest.json and results/wrapped_extension.json.','',
      '## Correctness and protocol','',
      '- All72 fresh cache roundtrips recovered every original BF16 bit. Later tiles decoded after an unrelated first payload was damaged.',
      '- Twenty unit tests passed, including all BF16 patterns, full signed16 and int32 extremes, tails, noncontiguous arrays, E8 membership/parity, true serialized prototype-ID matching, corruption/checksums, raw fallback accounting, and the step14 half-modulus regression.',
      '- Both selected learned correction and learned+E8 candidates passed eight successive model steps each with identical logits, generated tokens and appended KV. The E8 path stored real IDs, coordinates and exact corrections; it was not replaced by a baseline.',
      '- Main learned arms force actual predictor/correction records. Main E8/scalar arms force actual residual descriptions for finite groups. All final groups are verified to use those mechanisms. Unit-only nonfinite groups may use explicit raw preservation.',
      '- The frozen dictionary was not retrained. Development chose radius/L1 assignment and correction formats on the separate sensor prompt. E8/scalar grids used only development layers0/15/29, first256tokens; selected configurations were then checked on the full development cache.',
      '- Two real samples allow NumPy-only verification of both actual reconstruction pipelines. The separate fresh packing extension passed nine additional full-cache roundtrips, with every wrapped inner record byte-checked.','',
      '## Limits','',
      'These are CPU exact-storage/reference implementations. Full-prefix model checks do not establish bounded GPU reconstruction, cheap GPU decoding, improved TPOT/TTFT, PCIe/SSD benefits, higher admission capacity or concurrent serving. Runtime workspaces and attention allocations are outside serialized byte counts. Three CPU test jobs overlap and share resources; baseline lower-byte Deflate encode timing includes trial audits. Timing fields are diagnostic rather than controlled production benchmarks.','',
      'Both requested design families are now represented by measured exact prototypes, alongside the earlier exponent/lower-field work. This is coverage of those concrete approaches, not every possible learned predictor, correction model or lattice construction. New tuning should use development data and another fresh final split.','',
      '## Reproduce','',
      'Read ../README.md. final.json contains full per-case bytes, state charges, mechanisms, coding choices, hashes and generation checks. selection.json freezes source/model/evidence before final capture. Run python -m kv_residual.verify_sample for both included real representations without language-model weights.','']
    # Keep readable spaces around numeric values in the generated prose.
    replacements={'complete70,095':'complete 70,095','every64-value':'every 64-value','same16-token':'same 16-token',
      'was4/7/10':'was 4/7/10','All72':'All 72','step14':'step 14','layers0/15/29':'layers 0/15/29','first256tokens':'first 256 tokens',
      'full13-byte':'full 13-byte','Width15':'Width 15'}
    report='\n'.join(lines)
    for a,b in replacements.items():report=report.replace(a,b)
    (out/'review.md').write_text(report)
    dependencies=['kv_research/__init__.py','kv_research/container.py','kv_research/learned_codec.py','kv_research/e8_codec.py',
      'kv_research/native_decode.py','kv_research/native_decode.cpp','kv_research/model_utils.py','kv_research/run_eval.py',
      'kv_research/fixtures.py','kv_research/capture.py','kv_research/models/learned64.bin',
      'kv_smoke/codec.py','kv_smoke/prompts.py','kv_smoke/download_model.py','kv_smoke/requirements.txt',
      'kv_stacked/__init__.py','kv_stacked/run_eval.py','kv_stacked/huffman_codec.py','kv_stacked/field_codec.py','kv_stacked/wrapper.py',
      'kv_coset_fields/__init__.py','kv_coset_fields/combined.py','kv_coset_fields/controls.py']
    files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('package_manifest.json','e8_sample.bin')]
    files.extend(project/p for p in dependencies)
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      files=[dict(path=str(p.relative_to(project)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)])
    mpath=root/'package_manifest.json';mpath.write_text(json.dumps(manifest,indent=2)+'\n');files.append(mpath)
    archive=project/'kv_learned_corrections_e8_residual_tests.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(project)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    assert archive.stat().st_size<50*1024**2
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,best_learned=best_learned,
      learned_saving=save(best_learned),e8_saving=save('learned_e8_entropy'),best_overall=best,best_saving=save(best),e8_nonzero_percent=nonzero),indent=2))

if __name__=='__main__':main()
