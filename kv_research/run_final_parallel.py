"""Run independent final cases as CPU test jobs; never a serving benchmark."""
import argparse,concurrent.futures,hashlib,json,os,pathlib,subprocess,sys,time
from kv_research.run_eval import aggregate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    data=pathlib.Path('.kv_research_data').resolve()
    results=pathlib.Path('kv_research/results');results.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((data/'test_manifest.json').read_text())
    freeze=json.loads(pathlib.Path('kv_research/selection.json').read_text())
    for name,digest in freeze['source_and_model_sha256'].items():
        assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest
    jobs=pathlib.Path('.kv_research_jobs');jobs.mkdir(exist_ok=True)
    outputs=[]
    started=time.perf_counter()
    def run(index,case):
        job=jobs/str(index);job.mkdir(exist_ok=True)
        target=job/case['file']
        if not target.exists():target.symlink_to(data/case['file'])
        (job/'test_manifest.json').write_text(json.dumps({**manifest,'cases':[case]},indent=2)+'\n')
        output=results/f'final_case_{index}.json'
        cmd=[sys.executable,'-u','-m','kv_research.run_eval','--split','test','--data-dir',str(job),
            '--config','kv_research/final_config.json','--output',str(output)]
        if index==0:cmd+=['--generation','--save-sample']
        log=results/f'final_case_{index}.log'
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with log.open('w') as stream:
            subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,check=True)
        print('completed '+case['name'],flush=True)
        return index,output
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(run,i,c) for i,c in enumerate(manifest['cases'])]
        for future in concurrent.futures.as_completed(pending):outputs.append(future.result())
    partial=[json.loads(path.read_text()) for _,path in sorted(outputs)]
    assert len(partial)==len(manifest['cases'])==6
    assert len({part['cases'][0]['name'] for part in partial})==6
    expected={spec['name'] for spec in partial[0]['candidate_config']['candidates']}
    for case,part in zip(manifest['cases'],partial):
        assert len(part['cases'])==1 and part['cases'][0]['name']==case['name']
        assert part['candidate_config']==partial[0]['candidate_config']
        assert part['decoder_states']==partial[0]['decoder_states']
        assert set(part['cases'][0]['codecs'])==expected
        assert all(row['roundtrip_exact'] and row['later_tile_independence_passed'] for row in part['cases'][0]['codecs'].values())
    final=partial[0]
    final['capture_manifest']=manifest
    final['cases']=[part['cases'][0] for part in partial]
    final['aggregate']=aggregate(final['cases'],final['candidate_config']['candidates'],final['decoder_states'])
    final['test_job_execution']={'workers':args.workers,'elapsed_seconds':time.perf_counter()-started,
        'orchestration_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'note':'Independent CPU correctness/storage test jobs ran concurrently. Summed worker wall times overlap and share CPU resources; they are neither total elapsed runtime nor a clean serial comparison. This is not concurrent LLM serving.'}
    (results/'final.json').write_text(json.dumps(final,indent=2)+'\n')
    print('completed all final cases',flush=True)

if __name__=='__main__':main()
