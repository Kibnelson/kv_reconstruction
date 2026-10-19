"""Frozen final CPU test jobs, three independent workers, strict result merge."""
import concurrent.futures,hashlib,json,os,pathlib,subprocess,sys,time
from kv_stacked.run_eval import aggregate

def main():
    root=pathlib.Path('kv_stacked');out=root/'results';out.mkdir(exist_ok=True,parents=True)
    selection=json.loads((root/'selection.json').read_text())
    for p,digest in selection['frozen_sha256'].items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==digest,p
    config=json.loads((root/'final_config.json').read_text())
    manifest=json.loads(pathlib.Path('.kv_stacked_data/final_manifest.json').read_text())
    assert len(manifest['cases'])==6
    started=time.perf_counter();outputs=[]
    def run(i,case):
        output=out/f'final_case_{i}.json';log=out/f'final_case_{i}.log'
        cmd=[sys.executable,'-u','-m','kv_stacked.run_eval','--config',str(root/'final_config.json'),
            '--manifest','.kv_stacked_data/final_manifest.json','--only-index',str(i),'--output',str(output)]
        if i==0:cmd+=['--generation-winner',selection['winner'],'--save-sample']
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('completed '+case['name'],flush=True)
        return i,output
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,i,c) for i,c in enumerate(manifest['cases'])]
        for f in concurrent.futures.as_completed(futures):outputs.append(f.result())
    parts=[json.loads(p.read_text()) for _,p in sorted(outputs)]
    names=config['candidates']
    assert len(parts)==6
    for case,part in zip(manifest['cases'],parts):
        assert part['config']==config and part['state_bytes']==parts[0]['state_bytes']
        assert len(part['cases'])==1 and part['cases'][0]['name']==case['name']
        assert set(part['cases'][0]['codecs'])==set(names)
        assert all(c['roundtrip_exact'] and c['independent_later_tile'] for c in part['cases'][0]['codecs'].values())
    final=parts[0];final['cases']=[p['cases'][0] for p in parts]
    final['aggregate']=aggregate(final['cases'],final['state_bytes'],names)
    final['test_execution']={'workers':3,'elapsed_seconds':time.perf_counter()-started,
        'note':'Independent CPU test cases run concurrently; summed worker wall times overlap and are not serial or GPU serving benchmarks.'}
    (out/'final.json').write_text(json.dumps(final,indent=2)+'\n')
    print('all final cases complete',flush=True)

if __name__=='__main__':main()
