"""Software-only checks for matched T frames, review gates and frozen matrices."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def tiny_bundle():
    from kv_compare import fused
    from kv_compare.head_spread import Policy
    from kv_compare.head_spread_wire import policy_record
    from kv_compare.pd.matched import SCHEMA
    gen=torch.Generator().manual_seed(37)
    L,W,hd,C=2,64,16,32
    k=dict(Aq=torch.randint(-2,3,(L,W,W),dtype=torch.int8,generator=gen),
        sa=torch.full((L,W),.002),b=torch.randn((L,W),generator=gen)*.03,
        sp=torch.full((L,W),.3),isx=torch.full((L,W),12.))
    k['Aq'][0].zero_();k['inv']=(fused.QS/(2*math.log(2)*k['sp'].double())).float()
    groups=dict(k,groups=4,Acq=torch.zeros_like(k['Aq']))
    phase=torch.arange(640)[:,None]*torch.exp(-torch.arange(hd//2).float())[None,:]
    policy=Policy(2,.75,0,np.tile(np.eye(4,dtype=np.float32)[None],(L,1,1)),np.ones((L,4),np.float32))
    return dict(meta=dict(L=L,W=W,C=C,hd=hd),cos=phase.cos().contiguous(),sin=phase.sin().contiguous(),
        t_matched=dict(format=SCHEMA,dense=k,grouped=groups,policy=policy_record(policy),sha256='ab'*32))


class MatchedFrames(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET')=='1','explicit CPU interpreter required')
    def test_offset_tail_fresh_receiver_and_dense_stream_parity(self):
        import kv_compare.pd.matched as module
        old,pad=module.TB,module.PAD;module.TB,module.PAD=8,32
        self.addCleanup(setattr,module,'TB',old)
        self.addCleanup(setattr,module,'PAD',pad)
        from kv_compare.pd.matched import MatchedT,NAMES
        state=tiny_bundle()
        gen=torch.Generator().manual_seed(923)
        words=torch.randn((2,17,64),generator=gen).bfloat16().view(torch.int16)
        words[0,0,:5]=torch.tensor([0,-32768,32640,32705,-128],dtype=torch.int16)
        outputs={}
        for name in NAMES:
            with self.subTest(codec=name):
                print('CPU frame check:',name,flush=True)
                parts,meta=MatchedT(state,name,'cpu').encode(words,pos0=37)
                self.assertEqual(meta['Tpad'],32)
                with tempfile.TemporaryDirectory() as td:
                    path=Path(td)/'frame.pt'
                    torch.save(dict(parts=parts,meta=meta,state=state),path)
                    fresh=torch.load(path,map_location='cpu',weights_only=False)
                receiver=MatchedT(fresh['state'],name,'cpu')
                out=torch.full_like(words,12345)
                receiver.decode(fresh['parts'],fresh['meta'],lambda l,w:out[l].copy_(w))
                self.assertTrue(torch.equal(words,out))
                bad=dict(meta,policy_sha256='00'*32)
                with self.assertRaisesRegex(ValueError,'shared state/policy'):
                    receiver.decode(parts,bad,lambda l,w:None)
                outputs[name]=parts
        for k in outputs['t-fast']:
            self.assertTrue(torch.equal(outputs['t-fast'][k],outputs['t-dense-cached'][k]),k)

    def test_rope_and_input_geometry_limits(self):
        from kv_compare.pd.matched import MatchedT
        codec=MatchedT(tiny_bundle(),'t-fast','cpu')
        for T,pos in [(0,0),(17,-1),(513,100)]:
            with self.assertRaises(ValueError):codec.codec(T,pos)

    def test_dev_selection_global_fallback_and_tie(self):
        from kv_compare.pd.matched import policy_choice
        baseline=dict(frame_bytes=100,policy_bytes=0,alpha=0.,kind=0)
        learned=dict(frame_bytes=100,policy_bytes=30464,alpha=.75,kind=2)
        self.assertEqual(policy_choice([learned,baseline]),baseline)
        learned=dict(learned,frame_bytes=99)
        self.assertEqual(policy_choice([learned,baseline]),learned)


class CampaignContracts(unittest.TestCase):
    def runs(self,matrix,**extra):
        env=dict(os.environ,MATRIX=matrix,PRINT_RUNS='1',**extra)
        env.pop('RUNS',None)
        r=subprocess.run(['bash','scripts/pd_matrix.sh'],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        return r.stdout.splitlines()

    def test_smoke_both_layouts_and_matched_controls(self):
        from kv_compare.pd.matched import NAMES
        rows=self.runs('smoke')
        self.assertEqual(len(rows),8)
        for layout in ('1p3d','2p2d'):
            rr=[r for r in rows if 'LAYOUT='+layout in r and 'TRANSPORT=ring' in r and 't-fast' in r]
            self.assertTrue(rr)
            self.assertTrue(all(all(n in r for n in NAMES) for r in rr))
        self.assertTrue(all('MATCHED_ROUND=0' in r for r in rows))

    def test_full_repeats_frozen_once(self):
        rows=self.runs('ring2p2d',PD_MATRIX_ROUNDS='3')
        self.assertEqual(len(rows),6)
        self.assertTrue(all(sum('MATCHED_ROUND='+str(i) in r for r in rows)==2 for i in range(3)))
        env=dict(os.environ,MATRIX='ring2p2d',PRINT_RUNS='1',PD_MATRIX_ROUNDS='1',RUNS=';'.join(rows))
        r=subprocess.run(['bash','scripts/pd_matrix.sh'],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        self.assertEqual(r.stdout.splitlines(),rows)

    def test_failed_or_missing_methods_do_not_validate(self):
        spec=importlib.util.spec_from_file_location('matched_review',ROOT/'scripts/pd_matched_review.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'matrix').mkdir();(root/'setup').mkdir()
            (root/'campaign.json').write_text(json.dumps(dict(matrix='smoke',runs=['LAYOUT=1p3d CODECS=raw,t-fast,t-compact'])))
            (root/'run.json').write_text(json.dumps(dict(status='setup failed')))
            report=mod.audit(root)
            self.assertFalse(report['readiness_passed'])
            self.assertFalse(report['declared_campaign_coverage_passed'])
            self.assertFalse(report['full_repeated_comparison_validated'])
            self.assertTrue(report['smoke_only'])
            self.assertTrue(report['issues'])

    def test_complete_fixture_passes_only_smoke_and_missing_row_fails(self):
        from kv_compare.pd.matched import NAMES
        spec=importlib.util.spec_from_file_location('matched_review_positive',ROOT/'scripts/pd_matched_review.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        names=['raw',*NAMES]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'setup').mkdir();(root/'matrix'/'pd_matrix_fixture').mkdir(parents=True)
            (root/'node_specs').mkdir()
            (root/'node_specs'/'specs.json').write_text(json.dumps(dict(system={'fixture':True},bandwidth={'patterns':[{'fixture':True}]})))
            resultdir=root/'matrix'/'run';resultdir.mkdir()
            declared='LAYOUT=1p3d CODECS='+','.join(names)+' MATCHED_ROUND=0'
            (root/'campaign.json').write_text(json.dumps(dict(matrix='smoke',runs=[declared])))
            (root/'run.json').write_text(json.dumps(dict(status='completed',specs_exit=0)))
            (root/'setup'/'fixture_pd_matched_checks.json').write_text(json.dumps(dict(ok=True,gpu_executed=True,fresh_receivers=[{}]*4)))
            (root/'matrix'/'pd_matrix_fixture'/'matrix.json').write_text(json.dumps(dict(runs=[declared],failed=[],folders=['run'])))
            env=dict(layout='1p3d',connector='tkv',transport='ring',eager=False,matched_round=0,
                model_revision='fixture',prompt_sha256={'fixture':'a'*64},fresh_PD_prefix_cache_disabled=True,requested_codecs=names,engine_group='matched')
            (resultdir/'env.json').write_text(json.dumps(env))
            checks={n:dict(ok=True,prompts=1,transfers=dict(exact=1,installs=1)) for n in names}
            parameters=dict(codecs=names,links=[10],rates=['inf'])
            sweep={n:dict(codec=n,exit=0,link_gbps=10,rate='inf',requested_num_prompts=2,
                bench=dict(completed=2,request_throughput=2,request_goodput=2,median_ttft_ms=1,p99_ttft_ms=1),transfers=dict(installs=2)) for n in names}
            (resultdir/'results.json').write_text(json.dumps(dict(check=checks,sweep=sweep,benchmark_parameters=parameters)))
            r=mod.audit(root)
            self.assertTrue(r['declared_campaign_coverage_passed'],r['issues'])
            self.assertFalse(r['full_repeated_comparison_validated'])
            del sweep['t-fast']
            (resultdir/'results.json').write_text(json.dumps(dict(check=checks,sweep=sweep,benchmark_parameters=parameters)))
            self.assertFalse(mod.audit(root)['declared_campaign_coverage_passed'])


if __name__=='__main__':unittest.main()
