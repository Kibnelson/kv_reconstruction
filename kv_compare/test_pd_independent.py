"""Login launcher checks with a standalone fixture repo and a recording card."""
from __future__ import annotations

import json
import os
import pwd
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
BRANCH='experiment/t-pd-matched-campaign-v1'

CARD='''#!/usr/bin/env bash
python3 - "$@" <<'PY'
import json,os,sys
from pathlib import Path
record=dict(stage=sys.argv[1],pin=sys.argv[2],tag=sys.argv[3],env={k:v for k,v in os.environ.items() if k.startswith('KVPD_') or k in ('RUNS','CODECS','TRITON_INTERPRET','PD_MATRIX_ROUNDS')})
with Path(os.environ['RECORD']).open('a') as f:f.write(json.dumps(record)+'\\n')
print('recording fixture stage '+sys.argv[1],flush=True)
if sys.argv[1]==os.environ.get('FAIL_STAGE'):sys.exit(7)
PY
'''


class IndependentLoginTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.repo=self.root/'repo';self.repo.mkdir()
        (self.repo/'scripts').mkdir()
        shutil.copyfile(ROOT/'scripts/run_t_pd_independent.sh',self.repo/'scripts/run_t_pd_independent.sh')
        (self.repo/'scripts/pd_card.sh').write_text(CARD)
        (self.repo/'benchmark_results').mkdir()
        (self.repo/'benchmark_results'/'old_bundle.txt').write_text('historical results')
        self.git('init','-q','-b',BRANCH)
        self.git('config','user.name','fixture');self.git('config','user.email','fixture@example.invalid')
        self.git('add','.');self.git('commit','-qm','fixture')
        self.origin=self.root/'origin.git'
        subprocess.run(['git','clone','-q','--bare',str(self.repo),str(self.origin)],check=True)
        self.git('remote','add','origin',str(self.origin))
        self.pin=self.git('rev-parse','HEAD')
        self.record=self.root/'calls.jsonl';self.states=self.root/'states'
        uv=self.root/'uv';uv.write_text('#!/bin/sh\nexit 0\n');uv.chmod(0o755)
        self.env={k:v for k,v in os.environ.items() if not k.startswith(('KVPD_','T_PD_','SLURM_'))}
        self.env.update(RECORD=str(self.record),T_PD_STATE_HOME=str(self.states),T_PD_UV=str(uv),
            KVPD_BRANCH='pd-baselines',KVPD_ROOT='/tmp/inherited-team-root',
            KVPD_SETUP_CMD='inherited hook must not run',RUNS='inherited workload',CODECS='inherited-codec',
            TRITON_INTERPRET='1',PD_MATRIX_ROUNDS='9')

    def git(self,*args):
        return subprocess.run(['git','-C',str(self.repo),*args],check=True,capture_output=True,text=True).stdout.strip()

    def launch(self,repo=None,**extra):
        return subprocess.run(['bash',str((repo or self.repo)/'scripts/run_t_pd_independent.sh'),'smoke'],
            env=dict(self.env,**extra),capture_output=True,text=True,timeout=30)

    def calls(self):
        return [json.loads(x) for x in self.record.read_text().splitlines()] if self.record.exists() else []

    def test_own_namespace_and_saved_state(self):
        result=self.launch()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        calls=self.calls()
        self.assertEqual([x['stage'] for x in calls],['create','run','cleanup','verify-cleanup','review','publish'])
        for c in calls:
            e=c['env']
            self.assertEqual(c['pin'],self.pin)
            self.assertEqual(e['KVPD_BRANCH'],BRANCH)
            username=pwd.getpwuid(os.getuid()).pw_name
            self.assertEqual(e['KVPD_ROOT'],f"/tmp/{username}/kvpd-t-matched-{username}")
            self.assertNotIn('KVPD_SETUP_CMD',e)
            self.assertNotIn('RUNS',e);self.assertNotIn('CODECS',e)
            self.assertEqual((e['TRITON_INTERPRET'],e['PD_MATRIX_ROUNDS'],e['KVPD_SPECS']),('0','1','1'))
        state=next(self.states.glob('campaigns/*/state.sh'))
        self.assertEqual(state.stat().st_mode & 0o777,0o600)
        self.assertEqual(state.parent.stat().st_mode & 0o777,0o700)
        self.assertIn(str(self.repo),state.read_text())
        self.assertEqual((state.parent/'status.txt').read_text().strip(),'closeout-complete')
        self.assertEqual(self.git('rev-parse','HEAD'),self.pin)
        self.assertEqual(self.git('status','--porcelain'),'')

    def test_failure_keeps_our_log_and_state(self):
        result=self.launch(FAIL_STAGE='run')
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        self.assertEqual([c['stage'] for c in self.calls()],['create','run'])
        state=next(self.states.glob('campaigns/*/state.sh'))
        self.assertIn('stage=run exit=7',(state.parent/'status.txt').read_text())
        self.assertIn('recording fixture stage run',(state.parent/'login.log').read_text())

    def test_linked_worktree_is_rejected(self):
        linked=self.root/'linked'
        self.git('worktree','add','-q','--detach',str(linked),self.pin)
        result=self.launch(repo=linked)
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertIn('standalone git clone',result.stderr)
        self.assertFalse(self.calls())

    def test_dirty_clone_is_rejected(self):
        (self.repo/'untracked').write_text('keep this')
        result=self.launch()
        self.assertNotEqual(result.returncode,0)
        self.assertEqual((self.repo/'untracked').read_text(),'keep this')
        self.assertFalse(self.calls())

    def test_nonhead_pin_is_rejected(self):
        self.git('commit','--allow-empty','-qm','unpublished')
        result=self.launch()
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.calls())

    def test_existing_gpu_allocation_is_rejected(self):
        result=self.launch(SLURM_JOB_ID='fixture')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.calls())

    def test_documented_sparse_clone_starts_independently(self):
        clone=self.root/'independent'
        subprocess.run(['git','clone','-q','--filter=blob:none','--single-branch','--no-checkout',
            '--branch',BRANCH,str(self.origin),str(clone)],check=True,capture_output=True)
        for args in [('sparse-checkout','init','--no-cone'),
                ('sparse-checkout','set','--no-cone','/*','!/benchmark_results/',
                    '!/benchmark_fixtures/team_transfer_smol_replay_v1/'),('switch','--detach','HEAD')]:
            subprocess.run(['git','-C',str(clone),*args],check=True,capture_output=True)
        for key,value in [('user.name','fixture'),('user.email','fixture@example.invalid')]:
            subprocess.run(['git','-C',str(clone),'config',key,value],check=True)
        self.assertFalse((clone/'benchmark_results').exists())
        result=self.launch(repo=clone)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(len(self.calls()),6)
        self.assertTrue((clone/'.git').is_dir())
        self.assertFalse((clone/'.git/objects/info/alternates').exists())


if __name__=='__main__':unittest.main()
