"""Runtime-loader recovery and explicit backup calibration provenance checks."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock
from kv_compare.profile_recovery import LEGACY_CODE,digest,identity,recover_partial,runtime_environment


class RecoveryTests(unittest.TestCase):
    def test_native_loader_changes_do_not_change_aqua_runtime(self):
        from scripts.run_llama_profile_baselines import Suite
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'lib64').mkdir()
            suite=Suite.__new__(Suite); suite.out=root; suite.save=lambda:None; suite.result={}
            suite.env=dict(os.environ,CUDA_HOME=str(root),LD_LIBRARY_PATH='/unchanged-aqua')
            suite.aqua_visible_device='0'
            observed=[]
            def launch(argv,path): observed.append(os.environ['LD_LIBRARY_PATH'])
            with mock.patch('scripts.run_llama_profile_baselines.call',side_effect=launch):
                suite.command('exans_author_smoke',['fixture'])
                suite.command('aqua_reference',['fixture'])
            self.assertEqual(observed[0],str(root/'lib64')+os.pathsep+'/unchanged-aqua')
            self.assertEqual(observed[1],'/unchanged-aqua')
    def test_cuda_and_nvcomp_runtime_paths_preserve_existing_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp)/'cuda'; (home/'lib64').mkdir(parents=True)
            sdk=Path(tmp)/'sdk'; sdk.mkdir()
            old=dict(PATH='/usr/bin',LD_LIBRARY_PATH='/existing',CUDA_VISIBLE_DEVICES='GPU-fixture')
            new=runtime_environment(old,home,[sdk])
            self.assertEqual(new['LD_LIBRARY_PATH'].split(os.pathsep),[str(home/'lib64'),str(sdk),'/existing'])
            self.assertEqual(old['LD_LIBRARY_PATH'],'/existing')
            self.assertEqual(new['CUDA_VISIBLE_DEVICES'],'GPU-fixture')

    def test_narrow_legacy_resume_preserves_stage_evidence_and_archives_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'profile.json'; p.write_text('fixture')
            recipe=dict(code=LEGACY_CODE,pins={'aqua':['author/repo','pin']},stages=['profile','exans','aqua'],document='test_text',kappas=[0,1])
            old=dict(identity=identity(recipe),recipe=recipe,status='partial',stages={'profile':dict(status='completed',artifacts={'profile.json':digest(p)}),'aqua_train':dict(status='blocked')})
            (root/'results.json').write_text(json.dumps(old))
            newrecipe=dict(recipe,code={'fixed runner':'fixture'},aqua_calibration='original')
            newrecipe['pins']={'aqua':('author/repo','pin')}
            result=recover_partial(old,newrecipe,root)
            self.assertEqual(result['stages'],old['stages'])
            self.assertEqual(p.read_text(),'fixture')
            self.assertEqual(result['identity'],identity(newrecipe))
            archived=root/result['resume_history'][-1]['archived_result']
            self.assertEqual(json.loads(archived.read_text()),old)
            changed=copy.deepcopy(newrecipe); changed['kappas']=[.5]
            with self.assertRaisesRegex(ValueError,'protocol changed'): recover_partial(old,changed,root)
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'evidence changed'): recover_partial(old,newrecipe,root)

    def test_backup_choice_is_allowed_only_before_training_and_preserves_native_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            recipe=dict(code={'current source':'fixture'},stages=['profile','exans','aqua'],aqua_calibration='original')
            old=dict(identity=identity(recipe),recipe=recipe,status='partial',stages={'exans':dict(status='completed',artifacts={})})
            (root/'results.json').write_text(json.dumps(old))
            desired=dict(recipe,aqua_calibration='backup')
            self.assertEqual(recover_partial(old,desired,root)['stages']['exans']['status'],'completed')
            old['stages']['aqua_train']=dict(status='completed',artifacts={})
            with self.assertRaisesRegex(ValueError,'completed baseline training'): recover_partial(old,desired,root)


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
class CalibrationTests(unittest.TestCase):
    def test_sampler_matches_the_released_aqua_function_on_the_same_records(self):
        import torch
        from scripts.prepare_aqua_calibration import sample_records
        source=os.environ.get('AQUA_TEST_SOURCE')
        if not source: self.skipTest('pinned author source required')
        tree=ast.parse((Path(source)/'aquakv/datautils.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_red_pajama')
        data=[dict(text='a'*n) for n in [2,20,13,31,4,17]]
        class Tokenizer:
            def __call__(self,text,**kw): return types.SimpleNamespace(input_ids=torch.arange(len(text))[None])
        space=dict(load_dataset=lambda *a,**kw:data,random=random,trange=lambda n,**kw:range(n))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<unchanged author sampler>','exec'),space)
        random.seed(0); expected=space['get_red_pajama'](12,8,Tokenizer())
        actual,selection=sample_records(data,Tokenizer(),nsamples=12,seqlen=8,seed=0)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))
        self.assertEqual(len(selection),12)

    def test_backup_shard_hashes_token_file_geometry_reuse_and_tamper_rejection(self):
        import torch
        from transformers import AutoTokenizer
        from scripts.prepare_aqua_calibration import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source/aquakv'; source.mkdir(parents=True)
            (source/'datautils.py').write_text('fixture author source identity')
            shard=root/'shard.parquet'; shard.write_bytes(b'fixture bytes')
            manifest=root/'manifest.json'; manifest.write_text(json.dumps(dict(repo='fixture',revision='pin',rows=3,
                files=[dict(path='data/shard.parquet',bytes=shard.stat().st_size,sha256=digest(shard))])))
            class Data:
                column_names=['text','meta']; _fingerprint='fixture'
                def __len__(self): return 3
                def __getitem__(self,i): return dict(text='fixture '+str(i))
            class Tokenizer:
                def __call__(self,text,**kwargs): return types.SimpleNamespace(input_ids=torch.arange(8195)[None])
            fake=types.ModuleType('datasets'); fake.load_dataset=lambda *a,**kw:Data()
            output=root/'tokens.pt'; receipt=root/'receipt.json'
            with mock.patch.dict(sys.modules,{'datasets':fake}),mock.patch('huggingface_hub.hf_hub_download',return_value=str(shard)) as download,mock.patch.object(AutoTokenizer,'from_pretrained',return_value=Tokenizer()):
                result=prepare(source.parent,root/'model',output,receipt,root/'cache',manifest)
                self.assertEqual(result['source_mode'],'explicit_pinned_backup')
                self.assertIn('unverified',result['equivalence_to_original'])
                self.assertEqual(len(result['selections']),256)
                self.assertTrue(all(x.shape==(1,8192) for x in torch.load(output,weights_only=True)))
                self.assertEqual(prepare(source.parent,root/'model',output,receipt,root/'cache',manifest),result)
                self.assertEqual(download.call_count,1)
                output.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError,'provenance changed'):
                    prepare(source.parent,root/'model',output,receipt,root/'cache',manifest)
                shard.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError,'checksum differs'):
                    prepare(source.parent,root/'model',root/'new.pt',root/'new.json',root/'cache',manifest)


if __name__=='__main__': unittest.main()
