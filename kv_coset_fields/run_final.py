"""Frozen final comparison, independent CPU jobs with strict merge gates."""
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from kv_stacked.run_eval import aggregate

def main():
    root=pathlib.Path('kv_coset_fields');out=root/'results'
    selection=json.loads((root/'selection.json').read_text())
    for p,digest in selection['frozen_sha256'].items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==digest,p
    config=json.loads((root/'final_config.json').read_text())
    manifest=json.loads(pathlib.Path('.kv_coset_fields_data/final_manifest.json').read_text())
    assert len(manifest['cases'])==6 and len({c['name'] for c in manifest['cases']})==6
    started=time.perf_counter();outputs=[]
    def run(i,case):
        output=out/f'final_case_{i}.json';log=out/f'final_case_{i}.log'
        cmd=[sys.executable,'-u','-m','kv_coset_fields.run_eval','--config',str(root/'final_config.json'),
            '--manifest','.kv_coset_fields_data/final_manifest.json','--only-index',str(i),'--output',str(output)]
        if i==0:cmd+=['--generation-winner',selection['generation_candidate'],'--save-sample']
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('completed '+case['name'],flush=True)
        return i,output
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,i,c) for i,c in enumerate(manifest['cases'])]
        for f in concurrent.futures.as_completed(futures):outputs.append(f.result())
    parts=[json.loads(p.read_text()) for _,p in sorted(outputs)];names=list(config['candidates'])
    for case,part in zip(manifest['cases'],parts):
        assert part['config']==config and part['state_bytes']==parts[0]['state_bytes']
        assert len(part['cases'])==1 and part['cases'][0]['name']==case['name']
        row=part['cases'][0];assert set(row['codecs'])==set(names)
        assert row['raw_sha256']==case['raw_sha256'] and row['raw_payload_bytes']==case['raw_payload_bytes']
        expected=60*((case['tokens']+config['tile_tokens']-1)//config['tile_tokens'])
        for name,stats in row['codecs'].items():
            assert stats['roundtrip_exact'] and stats['independent_later_tile']
            if config['candidates'][name]['kind']!='baseline':
                assert stats['diagnostics']['audit']['records_verified']==expected
    final=parts[0];final['cases']=[p['cases'][0] for p in parts]
    final['aggregate']=aggregate(final['cases'],final['state_bytes'],names)
    final['test_execution']=dict(workers=3,elapsed_seconds=time.perf_counter()-started,
        note='Independent CPU jobs, not serving concurrency; stage wall times overlap and encode includes trial-decoder audits.')
    (out/'final.json').write_text(json.dumps(final,indent=2)+'\n')
    print('all final cases complete',flush=True)

if __name__=='__main__':main()
