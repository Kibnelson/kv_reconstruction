"""Regressions for JSON string paths and the pinned fast-only checkpoint."""
import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
from kv_compare.profile_recovery import RECOVERY_CODE_V1,digest,identity,recover_partial


class PathAndResumeTests(unittest.TestCase):
    def test_runner_hashes_json_strings_and_path_objects_identically(self):
        from scripts.run_llama_profile_baselines import sha
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'payload'; p.write_bytes(b'fixture')
            self.assertEqual(sha(str(p)),sha(p))
            self.assertEqual(sha(str(p)),digest(p))

    def test_prior_recovery_migration_preserves_completed_native_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); lib=root/'libcodec.so'; lib.write_bytes(b'native fixture')
            recipe=dict(code=RECOVERY_CODE_V1,pins={'aqua':['repo','pin']},stages=['profile','exans','aqua'],aqua_calibration='backup')
            old=dict(identity=identity(recipe),recipe=recipe,status='partial',stages={
                'exans_build':dict(status='completed',artifacts={'libcodec.so':digest(lib)}),
                'aqua_train':dict(status='blocked')})
            (root/'results.json').write_text(json.dumps(old))
            new=dict(recipe,code={'current':'fixture'})
            migrated=recover_partial(old,new,root)
            self.assertEqual(migrated['stages']['exans_build'],old['stages']['exans_build'])
            self.assertIn('exans_build',migrated['resume_history'][-1]['preserved_completed_stages'])
            self.assertEqual(migrated['identity'],identity(new))
            old['stages']['aqua_train']['status']='completed'
            with self.assertRaisesRegex(ValueError,'completed baseline training'): recover_partial(old,new,root)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('triton'),'PyTorch/Triton required')
class ExansWrapperTests(unittest.TestCase):
    def test_full_sender_and_receiver_glue_with_string_paths_and_real_wire_framing(self):
        # Native math is substituted by a CPU raw-overflow transport fixture.
        # All production sender/receiver, hash, serialization and proof code runs.
        import torch
        from scripts import run_llama_profile_baselines as runner
        from kv_compare.causal_packet import word_sha
        from kv_compare.framed_baselines import frame
        from kv_compare.official_exans import Exans,RAW,RECORD,METHOD,STATE
        class Codec:
            def __init__(self,source,library,device,blocks):
                self.impl=Exans.__new__(Exans); self.impl.device=torch.device('cpu'); self.impl.blocks=blocks
            def encode(self,words):
                n=words.numel()*2; count=math.ceil(n/RAW)
                raw=torch.zeros(count*RAW,dtype=torch.uint8); raw[:n]=words.contiguous().view(torch.uint8).flatten()
                return frame(METHOD,words.shape,STATE,[('fit_indices',torch.empty(0,dtype=torch.int32)),
                    ('raw_indices',torch.arange(count,dtype=torch.int32)),('records',torch.empty(0,dtype=torch.uint8)),('raw',raw)],
                    dict(blocks=count,raw_block_bytes=RAW,record_bytes=RECORD,input_padding_bytes=count*RAW-n))
            def decode(self,packet): return self.impl.decode(packet)
            def accounting(self,packet): return Exans.accounting(packet)
            def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); suite=runner.Suite.__new__(runner.Suite)
            suite.out=root; suite.cache=root/'cache'; suite.device=torch.device('cpu'); suite.torch=torch
            suite.base={'protocol':dict(warmup=1,repeats=1)}
            words=torch.arange(48,dtype=torch.int16).reshape(2,3,8)
            path=root/'words.npy'; np.save(path,words.numpy().astype('<u2'))
            suite.inputs=[dict(document='test',path=str(path),file_sha256=digest(path),
                original_word_sha256=word_sha(words),values=words.numel(),tokens=3)]
            library=root/'libcodec.so'; library.write_bytes(b'fixture library bytes')
            (root/'exans_build.json').write_text(json.dumps(dict(library=str(library),library_sha256=digest(library))))
            def command(name,argv):
                self.assertEqual(name,'exans_receiver')
                runner.receive(Path(argv[-1]),torch,torch.device('cpu'))
            suite.command=command
            with mock.patch('kv_compare.official_exans.Exans',Codec),mock.patch.object(runner,'check_source'),mock.patch.object(torch.cuda,'empty_cache'):
                result=suite.exans()
            data=json.loads((root/'exans.json').read_text())
            proof=json.loads((root/'exans_receiver.json').read_text())
            self.assertFalse(data['gpu_measured']); self.assertFalse(proof['gpu_measured'])
            self.assertEqual(data['rows'][0]['status'],'verified')
            self.assertTrue(data['rows'][0]['original_exact'])
            self.assertEqual(len(proof['checks']),2)  # TEST and correctness-only overflow/tail fixture
            self.assertTrue(all((root/p).is_file() for p in result['artifacts']))


@unittest.skipUnless(os.environ.get('LLAMA_TOKENIZER_TEST_PATH') and importlib.util.find_spec('torch'),'pinned checkpoint tokenizer assets required')
class PinnedTokenizerTests(unittest.TestCase):
    def test_actual_checkpoint_fast_backend_matches_official_ppl_and_author_sampling(self):
        import torch
        from transformers import AutoTokenizer
        from scripts.prepare_aqua_calibration import tokenizer_plan,sample_records
        root=Path(os.environ['LLAMA_TOKENIZER_TEST_PATH'])
        plan=tokenizer_plan(root)
        self.assertTrue(plan['use_fast'])
        # This reproduces the pre-fix failure with the actual checkpoint files.
        with self.assertRaises((ImportError,NotImplementedError)):
            AutoTokenizer.from_pretrained(root,use_fast=False,local_files_only=True)
        prepared=AutoTokenizer.from_pretrained(root,use_fast=plan['use_fast'],local_files_only=True)
        official=AutoTokenizer.from_pretrained(root,padding_side='left',local_files_only=True)
        self.assertTrue(prepared.is_fast)
        text='def add(x, y): return x + y\n' * 16
        self.assertEqual(prepared(text)['input_ids'],official(text)['input_ids'])
        source=Path(os.environ['AQUA_TEST_SOURCE'])/'aquakv/datautils.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_red_pajama')
        data=[dict(text=text),dict(text='An example paragraph. '*100)]
        namespace=dict(load_dataset=lambda *a,**kw:data,random=random,trange=lambda n,**kw:range(n))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<author sampler>','exec'),namespace)
        random.seed(0); expected=namespace['get_red_pajama'](4,16,official)
        actual,metadata=sample_records(data,prepared,nsamples=4,seqlen=16,seed=0)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))
        self.assertEqual(len(metadata),4)


if __name__=='__main__': unittest.main()
