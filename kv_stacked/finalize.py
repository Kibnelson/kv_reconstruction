"""Verify the frozen experiment, write review report and package deliverable."""
import datetime,hashlib,json,pathlib,re,shutil,zipfile

def main():
    root=pathlib.Path(__file__).resolve().parent;project=root.parent;out=root/'results'
    selected=json.loads((root/'selection.json').read_text())
    r=json.loads((out/'final.json').read_text());dev=json.loads((out/'dev.json').read_text())
    config=json.loads((root/'final_config.json').read_text())
    assert r['config']==config and len(r['cases'])==6
    assert len({c['name'] for c in r['cases']})==6
    for p,digest in {**selected['frozen_sha256'],**selected['evidence_sha256']}.items():
        assert hashlib.sha256((project/p).read_bytes()).hexdigest()==digest,'changed frozen evidence/source '+p
    names=config['candidates'];winner=selected['winner'];assert winner=='field_huffman'
    for case,source in zip(r['cases'],r['capture_manifest']['cases']):
        assert case['name']==source['name'] and case['raw_sha256']==source['raw_sha256'] and case['raw_payload_bytes']==source['raw_payload_bytes']
        assert set(case['codecs'])==set(names)
        assert all(v['roundtrip_exact'] and v['independent_later_tile'] for v in case['codecs'].values())
    generation=r['cases'][0]['codecs'][winner]['generation']
    assert generation['steps']==8 and generation['all_step_logits_bitwise_equal'] and generation['max_abs_logit_difference']==0
    unitlog=(out/'unit_tests.txt').read_text();assert 'Ran 12 tests' in unitlog and '\nOK\n' in unitlog
    first=r['cases'][0];blob=(out/'winner_sample.bin').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==first['codecs'][winner]['encoded_sha256']
    shutil.copyfile(project/'.kv_stacked_data'/first['file'],out/'sample_raw.npz')
    (out/'sample_metadata.json').write_text(json.dumps({k:first[k] for k in ('name','tokens','raw_payload_bytes','raw_sha256')}|{'encoded_sha256':first['codecs'][winner]['encoded_sha256']},indent=2)+'\n')
    shutil.copyfile(project/'.kv_stacked_data/final_manifest.json',out/'fresh_capture_manifest.json')
    shutil.copyfile(project/'.smoke_model/download_manifest.json',out/'checkpoint_manifest.json')
    rows=r['aggregate'];saving=lambda name:rows[name]['savings_including_state_percent']['1']
    b=saving('exponent');literal=saving('exponent_deflate');packed=saving('field_packed');best=saving(winner)
    lines=['# Building on exact exponent packing','',
      '**Measured scope:** exact BF16 KV storage and recovery on pretrained SmolLM2-135M, CPU. All principal comparisons retain independently decodable 16-token records. No GPU or serving claim follows.','',
      f"The original exponent codec saved **{b:.2f}%**. Literally adding per-record Deflate increased saving to **{literal:.2f}%**, a **{literal-b:.2f} percentage-point** gain. This is a genuine second coding stage on the original serialized output.",'',
      f"The strongest tested format saved **{best:.2f}%**: preserve original sign/fraction bits and use local canonical Huffman codes for exponent symbols. Relative to the original exponent codec, the **{best-b:.2f} percentage-point** gain includes both leaner framing and better exponent coding. It is not all attributable to stacking a generic compressor.",'',
      '## Fresh final comparison','',
      'Three fresh topics (coastal archive prose, Go ring buffer code and seed-vault records) were evaluated at 512 and 2,048 tokens after a development-only freeze. Length variants are correlated; these synthetic fixtures are coverage checks, not representative serving benchmarks. All bytes, local coding tables, checksums, wrapper headers and temporal indexes are included. Learned methods additionally pay the complete 70,095-byte dictionary per cache at N=1.','',
      '| Candidate | Description | Saving including state | Exact |','|---|---|---:|---|']
    labels={
      'exponent':'Original exponent-packed matrix records',
      'exponent_deflate':'Original exponent records → independent Deflate',
      'raw_deflate':'Raw BF16 → same Deflate wrapper',
      'byte_shuffle_deflate':'Reversible byte shuffle → Deflate',
      'field_packed':'Lean fields + group64 exponent packing',
      'field_packed_deflate':'Same lean fields; packed exponent stream → Deflate',
      'field_best':'Lean fields; best local exponent representation',
      'field_best_deflate':'Field-best complete record → Deflate',
      'learned':'Frozen learned dictionary + exact corrections',
      'learned_deflate':'Same learned record → Deflate',
      'field_huffman':'Sign/fraction verbatim + Huffman exponent stream',
      'field_huffman_deflate':'Complete Huffman record → Deflate'}
    for name in names:lines.append(f"| {name} | {labels[name]} | {saving(name):.2f}% | {rows[name]['all_exact']} |")
    lines+=['','Ratios are weighted by total raw bytes. Decoder-state charges are applied separately to each evaluated case; unrelated runs are not treated as concurrent caches. N=4/16 arithmetic sharing scenarios are available in final.json, not measured concurrent execution.','',
      '## Attribute each improvement correctly','',
      f"1. **Literal stacking:** exponent records → Deflate adds {literal-b:.2f} percentage points while preserving the original inner byte format.",
      f"2. **Leaner packaging:** the field-packed control saves {packed:.2f}%, versus {b:.2f}%. It removes redundant per-group framing while retaining bases, widths, shapes, checksums and temporal indexes.",
      f"3. **Packed-exponent entropy stage:** the identical lean layout with Deflate on its packed exponent stream saves {saving('field_packed_deflate'):.2f}%, versus {packed:.2f}% before that stage.",
      f"4. **Better exponent model:** canonical Huffman coding saves {best:.2f}% in the lean field layout. It assigns short exact codes to frequent original exponent symbols; the complete sparse code-length table is stored inside every record.",
      f"5. **Stop when stacking stops paying:** another Deflate wrapper on the Huffman record saves {saving('field_huffman_deflate'):.2f}%, versus {best:.2f}% without it. Added coding is useful only when remaining redundancy exceeds its overhead.",'',
      'The strongest format is a refinement of the exponent-field representation, not a lossy approximation and not a neural KV predictor. It preserves one sign bit and seven fraction bits per value verbatim, then entropy-codes the exact eight-bit exponent symbols. All original BF16 bits are reassembled. The standalone percentages of previous codecs are never added together.','',
      '## Development and larger record experiment','',
      'Development used the prior separate mixed sensor prompt, 1,024 tokens. Configuration, source and model hashes were frozen before capturing fresh final KV. Original exponent Huffman beat temporal-delta and channel-delta variants; rejected trials are recorded in huffman_dev_sweep.json. Wrapper level 6 and all strategy choices were fixed before fresh capture.','',
      'A development-only granularity probe compared 16-, 64- and 256-token records. Huffman improved from 31.730% to 32.635% saving between 16 and 256, while the returned K+V word tile increased from 12 KiB to 192 KiB. These are returned tensor bytes, not total decoder or GPU workspace. Larger records change random-access granularity, so the primary comparison retains 16-token records. See granularity_dev.json.','',
      '## Correctness and execution limits','',
      f"- All {len(r['cases'])*len(names)} fresh cache roundtrips recovered the original BF16 words exactly and passed independent later-tile checks after an unrelated first payload was damaged.",
      '- Twelve unit tests passed, covering every BF16 word, all exceptional-value payloads, zero/tail shapes, coding tables, wrapper fallback, byte shuffling, checksums, corruption and byte accounting.',
      '- The development-selected Huffman winner passed eight successive model decode steps with identical logits, generated tokens and appended KV.',
      '- The model check restores the full prefix on CPU. It does not demonstrate compact-resident attention with bounded GPU reconstruction buffers.',
      '- Huffman decoding here is a Python prefix-tree implementation. Its storage result does not establish cheap GPU decoding; a native parallel implementation and actual attention integration remain to be measured.',
      '- Deflate and Huffman state/tables inside the stored records count toward encoded bytes. Runtime heap objects, decompressor workspaces and attention allocations are not measured by these serialized sizes.',
      '- Final cases ran as three independent CPU test jobs. Summed stage wall times overlap and are affected by contention. Wrapped encode times include measured source-stage time, with source records reused during comparison; these are diagnostic CPU timings, not serving benchmarks.','',
      '## What to improve next','',
      'This establishes a stronger exact-storage baseline for further learned or E8 work. Potential additional stages must target remaining uncertainty rather than recode already-removed redundancy. A cheap conditional probability model for exponent symbols, compact shared tables, or reversible prediction of sign/fraction information can be tested next. Shared-state bytes, restart metadata and decode cost must remain in the comparison.',
      'The sign/fraction stream remains verbatim and is eight bits per original value. Its remaining compressibility is a measurement question; empirical entropy from a small sample is not a universal lower bound. Further tuning requires fresh final data.','',
      '## Reproduce and inspect','',
      'Read ../README.md. final.json contains per-case complete bytes, local components, hashes and model checks. selection.json records frozen source/state hashes; fresh_capture_manifest.json records exact token IDs and raw KV hashes. sample_raw.npz and winner_sample.bin allow verification without language-model weights: python -m kv_stacked.verify_sample.','']
    (out/'review.md').write_text('\n'.join(lines))
    dependencies=['kv_research/__init__.py','kv_research/container.py','kv_research/learned_codec.py','kv_research/native_decode.py','kv_research/native_decode.cpp',
      'kv_research/model_utils.py','kv_research/run_eval.py','kv_research/e8_codec.py','kv_research/fixtures.py','kv_research/models/learned64.bin',
      'kv_research/training_report.json','kv_smoke/codec.py','kv_smoke/prompts.py','kv_smoke/download_model.py','kv_smoke/requirements.txt']
    files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.so' and p.name!='package_manifest.json']
    files +=[project/p for p in dependencies]
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
       files=[dict(path=str(p.relative_to(project)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)])
    mpath=root/'package_manifest.json';mpath.write_text(json.dumps(manifest,indent=2)+'\n');files.append(mpath)
    archive=project/'kv_exponent_stacking_tests.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(project)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,
        original_exponent_saving=b,literal_stack_saving=literal,best_saving=best,winner=winner),indent=2))

if __name__=='__main__':main()
