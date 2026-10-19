"""Build measured report and reproducible follow-up archive after final tests."""
import datetime,hashlib,json,pathlib,re,shutil,zipfile

def main():
    root=pathlib.Path(__file__).resolve().parent; project=root.parent
    results=root/'results'
    final=json.loads((results/'final.json').read_text())
    dev=json.loads((results/'dev.json').read_text())
    extra=json.loads((results/'dev_extra.json').read_text())
    freeze=json.loads((root/'selection.json').read_text())
    assert len(final['cases'])==6
    assert final['candidate_config']==json.loads((root/'final_config.json').read_text())
    for path,digest in freeze['source_and_model_sha256'].items():
        assert hashlib.sha256((project/path).read_bytes()).hexdigest()==digest,'changed frozen source/state '+path
    for path,digest in freeze['dev_result_sha256'].items():
        assert hashlib.sha256((project/path).read_bytes()).hexdigest()==digest,'changed development evidence '+path
    training=json.loads((root/'training_report.json').read_text())
    reproduced=training['reproduction_check']
    assert reproduced['all4models_sha256_identical']
    for count,digest in reproduced['model_ids'].items():
        assert hashlib.sha256((root/'models'/f'learned{count}.bin').read_bytes()).hexdigest()==digest
    assert hashlib.sha256((root/'models/fixed64.bin').read_bytes()).hexdigest()==reproduced['fixed64_model_id']
    unitlog=(results/'unit_tests.txt').read_text()
    unitcount=int(re.search(r'Ran (\d+) tests in',unitlog).group(1))
    assert unitcount==18 and '\nOK\n' in unitlog and 'FAILED' not in unitlog
    assert all(row['all_roundtrips_exact'] for row in final['aggregate'].values())
    sample_case=final['cases'][0]
    shutil.copyfile(project/'.kv_research_data'/sample_case['file'],results/'sample_raw.npz')
    sample_specs=[]
    for spec in final['candidate_config']['candidates']:
        path=results/(spec['name']+'_sample.bin')
        if spec['family'] in ('e8','learned'):
            assert path.exists(),'missing sample '+spec['name']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==sample_case['codecs'][spec['name']]['encoded_sha256']
            sample_specs.append({**spec,'encoded_sha256':sample_case['codecs'][spec['name']]['encoded_sha256']})
            if spec.get('generation',True):
                g=sample_case['codecs'][spec['name']]['generation']
                assert g['steps']==8 and g['all_step_logits_bitwise_equal'] and g['max_abs_logit_difference']==0
    (results/'sample_metadata.json').write_text(json.dumps({'case':sample_case['name'],'candidates':sample_specs},indent=2)+'\n')
    for split in ('train','dev','test'):
        shutil.copyfile(project/'.kv_research_data'/(split+'_manifest.json'),results/(split+'_capture_manifest.json'))
    shutil.copyfile(project/'.smoke_model/download_manifest.json',results/'checkpoint_manifest.json')
    rows=final['aggregate']; selected=freeze['selected_learned_name']
    learned=rows[selected]; baseline=rows['exponent']; fixed=rows['fixed64']; e8=rows['e8_symbol']
    gain=learned['savings_including_state_percent']['1']-baseline['savings_including_state_percent']['1']
    fixed_gain=learned['savings_including_state_percent']['1']-fixed['savings_including_state_percent']['1']
    lines=['# Exact KV reconstruction: learned and E8 follow-up','',
      '**Measured scope:** CPU reference codecs on pretrained SmolLM2-135M BF16 KV. These results concern exact representation size and reconstruction correctness; GPU serving and capacity-limited model execution are not implemented.','',
      f"The selected trained dictionary saved **{learned['savings_including_state_percent']['1']:.2f}%** including all indexed records and one complete serialized decoder model per cache. Exact exponent packing saved **{baseline['savings_including_state_percent']['1']:.2f}%**: a **{gain:.2f} percentage-point** improvement on these fixtures.",'',
      f"Relative to the same-layout fixed dictionary, the selected learned model improved saving by **{fixed_gain:.2f} percentage points** after the same state-accounting rule. This supports learned prototype placement over this particular fixed construction, not superiority over all analytical or learned codecs.",'',
      '## What was changed and tested','',
      '- Genuine nearest-E8 projection was implemented using the two D8 cosets. A float-magnitude version and a BF16 magnitude-symbol version were tested during development. This coordinate-based prototype is not the finite E8P12 paper implementation.',
      '- Vector dictionaries were genuinely trained using only three separate 1,024-token training prompts. There are 60 dictionaries, one per layer and K/V, pooled across heads; each entry predicts eight BF16 magnitude symbols. Encoding selects by maximum symbol error. This is a trained lookup decoder, not a neural autoencoder or AQUA-KV reproduction.',
      '- Exact signs and modular corrections preserve original BF16 words. Correction widths vary per eight values; 64-value framing groups retain raw fallback. Every required historical token remains represented.',
      '- Development selected the E8 step/scale and dictionary size. Serialized models and source/configuration hashes were frozen before final KV capture. See selection.json and PROTOCOL.md.',
      '- Final evaluation used three untouched synthetic topics at 512 and 2,048 tokens: prose, Rust code, and structured lookup. These are six correlated prompt/length cases, not six independent representative workloads.','',
      '## Final complete sizes','',
      'Savings use raw BF16 payload as denominator. Payload includes all framing and temporal indexes; state includes the complete model file, even training statistics. N=4/16 are arithmetic sharing scenarios, not measured concurrency.','',
      '| Candidate | Encoded MiB | Shared state bytes | Payload saving | Saving, N=1 | Saving, N=16 | Exact |',
      '|---|---:|---:|---:|---:|---:|---|']
    for name,row in rows.items():
        lines.append(f"| {name} | {row['serialized_bytes']/2**20:.3f} | {row['decoder_state']['serialized_bytes']:,} | {row['payload_space_saving_percent']:.2f}% | {row['savings_including_state_percent']['1']:.2f}% | {row['savings_including_state_percent']['16']:.2f}% | {row['all_roundtrips_exact']} |")
    raw_values=learned['raw_payload_bytes']/2
    e8_coordinates=8*e8['component_bytes']['matrix_e8_coordinates_and_coset_flags']/raw_values
    learned_indices=8*learned['component_bytes']['matrix_code_indices']/raw_values
    e8_corrections=8*e8['component_bytes']['matrix_exact_corrections']/raw_values
    learned_corrections=8*learned['component_bytes']['matrix_exact_corrections']/raw_values
    lines+=['','State is charged separately for each evaluated case before aggregation; unrelated runs are not treated as a single concurrent deployment. Aggregates are byteweighted. The indexed raw control exposes format overhead.','',
      '## Why the learned record is smaller','',
      '| Payload component | E8 symbol: bits per original value | Learned dictionary: bits per original value |','|---|---:|---:|',
      f'| Coordinates or dictionary indexes | {e8_coordinates:.3f} | {learned_indices:.3f} |',
      f'| Exact corrections | {e8_corrections:.3f} | {learned_corrections:.3f} |','',
      'These rows are components, not total rates; signs, widths, scales, fallbacks, framing, indexes and shared state are counted in the complete-size table above. E8 obtains smaller corrections, but its explicit lattice coordinates cost more than the learned codebook indexes. The learned decoder amortizes richer patterns in its fixed dictionary. A more accurate prediction therefore need not yield a smaller exact representation. Corrections still dominate the learned payload, leaving substantial work for any stronger codec.','',
      '## Development results and iteration','',
      '| Candidate | Full dev payload saving | Dev saving including one decoder |','|---|---:|---:|']
    for source in (dev,extra):
        for name,row in source['aggregate'].items():
            lines.append(f"| {name} | {row['payload_space_saving_percent']:.2f}% | {row['savings_including_state_percent']['1']:.2f}% |")
    lines+=['',
      'Development showed ordinary float E8 was a poor match for exact BF16 correction cost. Moving E8 to magnitude-symbol space improved storage but remained behind exponent packing. Learned prototype placement and per-eight-value corrections improved the bytes frontier. The fixed control uses exactly the learned record layout and correction grouping, helping separate this implementation change from training.',
      'The fixed dictionary is one plausible source-independent digital grid; its current file repeats dictionaries by layer and is conservatively charged. Its state could be generated procedurally in a different implementation. Larger learned dictionaries were evaluated on development and charged their actual larger files. No configuration was changed after observing final results.','',
      '## Correctness and model integration','']
    count=0
    for case in final['cases']:
        for name,row in case['codecs'].items():
            assert row['roundtrip_exact'] and row['later_tile_independence_passed'];count+=1
            if 'generation' in row:
                g=row['generation']; assert g['all_step_logits_bitwise_equal'] and g['steps']==8
                lines.append(f"- {name}, {case['name']}: eight successive decode steps had identical logits and appended KV; maximum logit difference {g['max_abs_logit_difference']:.0f}.")
    lines += [f'- All {count} final cache roundtrips and later-tile independence checks passed.',
      '- Eighteen unit tests passed: 15 reference codec/container tests plus three native decoder tests. These cover E8 membership/nearest projection/parity packing, complete BF16 patterns, exact corrections/edge cases, model identity/serialization, indexed access, and native width/format agreement.',
      '- All four trained/fixed dictionary files were regenerated byte-identically from the captured training data. See training_report.json.',
      '- The bundled raw sample and encoded records can be checked without model weights or PyTorch: python -m kv_research.verify_sample.','',
      '## Reference timings','',
      '| Candidate | Sum of encode wall seconds | Sum of decode wall seconds |','|---|---:|---:|']
    for name,row in rows.items():lines.append(f"| {name} | {row['encode_seconds_total']:.3f} | {row['decode_seconds_total']:.3f} |")
    lines+=['',
      'These are CPU Python/NumPy elapsed times with no statistical performance claim. Learned timing includes repeated model serialization/hashing for identity checks. Training time, model prefill, generation and codec timings are recorded separately. Development jobs overlapped; final evaluation used three independent test-case workers after development. CPU contention affects timings. This is not concurrent LLM serving; no GPU or transfer throughput is inferred.','',
      '## What remains unresolved','',
      'Exact recovery and the reported size improvement are confirmed only for these cases and this dictionary decoder. Independent final prompts provide a small generalization check, not broad task-quality or model-family evidence.',
      'The real-model check restores the whole prefix on CPU. It does not show attention consuming compact resident records with only a bounded reconstruction buffer. Runtime codebook arrays, Python/allocator overhead and transient workspace differ from serialized state; parameter-array bytes are separately recorded. N=1/4/16 does not measure concurrent requests or admission latency.',
      'Next: compare stronger exact baselines and byte-optimized controls; reduce repeated identity/header overhead; implement integer lookup/correction in a native tiled attention path; then measure GPU workspace, TPOT, transfers and admission under long-context concurrency. Use fresh final data for subsequent codec tuning. Dictionary learning, E8 and predictor-plus-correction coding are established primitives; this experiment does not establish novelty.','',
      '## Provenance','',
      '- Checkpoint and raw capture hashes: *_capture_manifest.json and checkpoint_manifest.json.',
      '- Configuration freeze and source/model hashes: ../selection.json.',
      '- Training and exact reproduction: ../training_report.json and ../train_models.py.',
      '- E8 projection reference cited in source: https://neilsloane.com/doc/Me83.pdf, Section VI.',
      '- Prior simple-predictor result was 5.70% saving on different fixtures; current raw-byte comparisons use the new evaluation cases.','']
    native_path=results/'native_benchmark.json'
    if native_path.exists():
        native=json.loads(native_path.read_text())
        assert native['runtime']['final_jobs_completed']
        assert native['source'].startswith('final frozen')
        assert native['hashes']['raw_word_sha256']==sample_case['raw_sha256']
        assert native['hashes']['encoded_sha256']==sample_case['codecs'][selected]['encoded_sha256']
        assert hashlib.sha256((root/'native_decode.cpp').read_bytes()).hexdigest()==native['hashes']['native_cpp_sha256']
        assert hashlib.sha256((root/'native_decode.py').read_bytes()).hexdigest()==native['hashes']['native_wrapper_sha256']
        native_generation=json.loads((results/'native_generation.json').read_text())
        assert native_generation['steps']==8 and native_generation['all_step_logits_bitwise_equal'] and native_generation['max_abs_logit_difference']==0
        section=['## Native CPU learned-record decoder','',
          'A compiled integer unpack/lookup decoder was added without changing the frozen model or stored record format. It snapshots and hashes decoder state once. This is an execution implementation change, not a new fitted candidate.',
          '',f"The complete {native['raw_payload_bytes']/2**20:.2f} MiB raw KV sample was reconstructed after warmup, with three timed passes per implementation and bitwise checks after every pass. Timing began after all final test workers finished.",'',
          '| Decoder | Median reconstruction ms | Raw MiB/s | Exact |','|---|---:|---:|---|']
        for name,row in native['measurements'].items():
            assert row['all_bf16_words_exact'] and row['complete_equality_checks']==4
            section.append(f"| {name} | {row['median_seconds']*1000:.3f} | {row['raw_MiB_per_second']:.2f} | True |")
        section+=['',
          'The native decoder also passed eight successive model decode steps with identical logits, generated tokens and appended KV. Compilation, library loading, one-time model snapshot/hash, encoding, source reads and equality checks are outside the timing loop. Container traversal, output allocation and concatenation are included. The cached-identity Python control helps separate hashing overhead from unpacking/lookup overhead.',
          '',
          'These local CPU measurements confirm a faster implementation of the same materialization rule. They do not establish GPU performance or a serving latency win. Every path restores the whole prefix into CPU arrays; bounded GPU reconstruction and attention remain unimplemented. The native binary is excluded from the package and is rebuilt locally from source.',
          '']
        position=lines.index('## What remains unresolved')
        lines[position:position]=section
    (results/'results.md').write_text('\n'.join(lines))
    archive=project/'kv_reconstruction_learned_e8_tests.zip'
    files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='package_manifest.json' and p.suffix!='.so']
    for name in ('codec.py','prompts.py','download_model.py','requirements.txt'):
        files.append(project/'kv_smoke'/name)
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'files':[{'path':str(p.relative_to(project)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]}
    manifest_path=results/'package_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');files.append(manifest_path)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(project)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,selected=selected,
      learned_saving_percent=learned['savings_including_state_percent']['1'],exponent_saving_percent=baseline['savings_including_state_percent']['1'],gain_percentage_points=gain),indent=2))

if __name__=='__main__':main()
