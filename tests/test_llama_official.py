import json
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from qwen_kv_bench import transfer_reference as ref
from scripts import run_llama_official_a4500 as runner
from tests.test_qwen_matched_gpu import captures


class ReferenceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('constriction'),'constriction required')
    def test_scaled_rope_exact_stream_roundtrip_at_nonzero_position(self):
        from qwen_kv_bench import transfer_probability as prob
        train,scale,test=captures()
        for a in (train,scale,test): a['rope_inv_freq']=np.array([.5,.125,.03125,.0078125])
        test['positions']=np.arange(7,7+test['K_bits'].shape[1])
        base=ref.fit([train],scale,mode='block',block_size=8,max_positions=128)
        state=prob.empty_state(base,'A')
        packet,_=prob.encode(base,state,test)
        recovered,_=prob.decode(base,state,packet)
        np.testing.assert_array_equal(recovered,ref.words(test))
        legacy,_=ref.encode(base,ref.words(test),start=7)
        restored,_=ref.decode(base,legacy)
        np.testing.assert_array_equal(restored,ref.words(test))

    def test_real_frequencies_control_float_and_frozen_integer_rotations(self):
        train,scale,test=captures()
        freq=np.array([.5,.125,.03125,.0078125])
        for a in (train,scale,test): a['rope_inv_freq']=freq
        geometry=ref._geometry(train)
        value=np.zeros((2,8));value[:,0]=1
        rotated=ref._rope_float(value,geometry,np.array([1,7]))
        np.testing.assert_allclose(rotated[:,0],np.cos(np.array([1,7])*.5))
        np.testing.assert_allclose(rotated[:,1],np.sin(np.array([1,7])*.5))
        base=ref.fit([train],scale,mode='block',block_size=8,max_positions=128)
        np.testing.assert_array_equal(base.cosine[7],np.rint(np.cos(7*freq)*32768).astype(np.int64))
        np.testing.assert_array_equal(base.sine[7],np.rint(np.sin(7*freq)*32768).astype(np.int64))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'base.npz';ref.save_model(base,path);loaded=ref.load_model(path)
            self.assertEqual(base.state_sha256,loaded.state_sha256)
            self.assertEqual(loaded.geometry['rope_inv_freq'],freq.tolist())

    def test_missing_frequencies_preserve_legacy_geometry_and_reject_bad_capture(self):
        train,_,_=captures();old=ref._geometry(train)
        self.assertNotIn('rope_inv_freq',old)
        expected=old['base']**(-np.arange(0,8,2,dtype=np.float64)/8)
        np.testing.assert_array_equal(ref._rope_frequencies(old),expected)
        for invalid in ([1,2],[1,2,3,float('nan')],[1,2,3,0]):
            with self.assertRaisesRegex(ValueError,'inverse frequencies'):
                ref._geometry(dict(train,rope_inv_freq=invalid))


class ProtocolTests(unittest.TestCase):
    def test_synthetic_smoke_does_not_qualify_as_completed_model_comparison(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(runner,'SOURCE_FILES',[]):
            run=runner.Run(Path(tmp)/'out',['exans'])
            run.stage('exans',lambda:dict(status='smoke_only',scope='synthetic correctness only'))
            self.assertEqual(run.result['stages']['exans']['status'],'smoke_only')
            self.assertFalse(all(run.result['stages'].get(s,{}).get('status')=='completed' for s in run.result['requested']))
            self.assertFalse(run.result['gpu_measured'])

    def test_mismatched_tokenization_and_quantized_model_are_rejected(self):
        a=dict(text_sha256='text',token_ids_sha256='tokens',tokens=10)
        runner.same_tokens(a,dict(a))
        with self.assertRaisesRegex(ValueError,'token IDs differ'):
            runner.same_tokens(a,dict(a,token_ids_sha256='different'))
        cfg=dict(model_type='llama',num_hidden_layers=28,num_key_value_heads=8,head_dim=128,torch_dtype='bfloat16')
        runner.check_model_config(cfg)
        with self.assertRaisesRegex(ValueError,'unquantized'):
            runner.check_model_config(dict(cfg,quantization_config={'bits':4}))

    def test_upstream_modification_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'author.py';path.write_text('original\n')
            subprocess.run(['git','init','-q',root],check=True)
            subprocess.run(['git','add','author.py'],cwd=root,check=True)
            subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                'commit','-qm','author source'],cwd=root,check=True)
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
            runner.check_sources(root,head)
            path.write_text('modified\n')
            with self.assertRaisesRegex(RuntimeError,'modified'):runner.check_sources(root,head)

    def test_failed_correctness_cannot_become_completed_and_dependencies_stop(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner,'SOURCE_FILES',[]):
            run=runner.Run(Path(tmp)/'out',['ours'])
            def fail(): raise AssertionError('different BF16 words')
            run.stage('ours',fail)
            attempted=[];run.stage('dependent',lambda:attempted.append(True),['ours'])
            self.assertEqual(run.result['stages']['ours']['status'],'failed')
            self.assertEqual(run.result['stages']['dependent']['status'],'blocked')
            self.assertFalse(attempted);self.assertFalse(run.result['gpu_measured'])

    def test_resume_rejects_changed_completed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(runner,'SOURCE_FILES',[]):
            run=runner.Run(Path(tmp)/'out',['ours']);path=run.out/'evidence.json';path.write_text('original')
            run.stage('proof',lambda:dict(artifacts=run.artifacts(path)))
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):run.stage('proof',lambda:None)

    def test_report_allowlist_excludes_model_kv_tokens_and_predictors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'run';src.mkdir()
            (src/'results.json').write_text('{}');(src/'summary.md').write_text('software check')
            for name in ['model.safetensors','aqua_predictors.pt','raw_kv.npy','tokens.json','encoded.t']:
                (src/name).write_bytes(b'must not publish')
            dest=root/'report';runner.package(src,dest)
            self.assertEqual({p.name for p in dest.iterdir()},{'results.json','summary.md','package_manifest.json'})

    def test_official_aqua_command_retains_full_calibration_and_buffers(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=runner.Run.__new__(runner.Run);run.out=Path(tmp);run.sources={'aqua':Path(tmp)}
            run.python={'aqua':'author-python'};run.snapshot=Path(tmp)/'snapshot';run.env={};run.result={}
            identity=dict(test_data=dict(text_sha256='t',token_ids_sha256='i',tokens=100))
            for name in ['aqua_preflight.json','splitzip_preflight.json']:
                (run.out/name).write_text(json.dumps(identity))
            commands=[]
            def command(name,argv,cwd=None):
                commands.append((name,list(map(str,argv))))
                log=run.out/(name+'.log');log.write_text('WikiText-2 perplexity: 7.0\n');return log
            run.command=command;run.verify_env=lambda name:None
            with mock.patch.object(runner,'check_sources'):
                result=run.aqua()
            train=commands[0][1]
            self.assertIn('train_predictors.py',Path(train[1]).name)
            self.assertEqual(train[train.index('--total_nsamples')+1],'256')
            self.assertEqual(train[train.index('--valid_nsamples')+1],'32')
            self.assertEqual(train[train.index('--model_seqlen')+1],'8192')
            self.assertNotIn('--not_quantize_first_layer',train)
            for _,cmd in commands[1:]:
                self.assertEqual(cmd[cmd.index('--prefix_size')+1],'4')
                self.assertEqual(cmd[cmd.index('--recent_buffer_size')+1],'128')
            self.assertEqual(result['raw_perplexity'],7.)


if __name__=='__main__':unittest.main()
