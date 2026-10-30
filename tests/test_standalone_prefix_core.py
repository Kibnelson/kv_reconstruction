"""Exact word arithmetic, lane-layout contracts and actual-CUDA prefix gates."""
import ast
import copy
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from kv_compare import standalone_prefix_core as core
from kv_compare import standalone_inverse_core as inverse
from kv_compare.team_onestep import gpu_coder as gc
from kv_compare.team_onestep import layerwise as lw
from tests.test_standalone_inverse_core import native_fixture
from tests.test_team_packet import fixed_state


def function(path,name):
    tree=ast.parse(Path(path).read_text())
    node=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name))
    node.decorator_list=[]
    if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
        node.body.pop(0)
    return node


class Rename(ast.NodeTransformer):
    def visit_Name(self,node):
        node.id={'_prefix_dec_value':'_dec_value_core','_take_prefix':'_take','_slots_warp':'_slots'}.get(node.id,node.id)
        return node
    def visit_Attribute(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='gc' and node.attr=='_take':
            return ast.copy_location(ast.Name(id='_take',ctx=ast.Load()),node)
        return node


def fake_module():
    alias='#linear1'
    identity='#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [], block = []}>'
    operands=', '.join(['tensor<32xi32, '+alias+'>']*5)
    site=('tt.elementwise_inline_asm "kv_prefix_warp32_v1" '
          '{packed_element = 1 : i32, pure = false} : '+operands+' -> '+operands)
    ttgir=(alias+' = '+identity+'\n'
           'module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, '
           'ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {\n'+site+'\n'+site+'\n}')
    return SimpleNamespace(hash='fixture',asm={'ttgir':ttgir,
        'ptx':'.target sm_86\n.reqntid 32\n'+('vote.ballot.sync.b32\n'*8)})


class ProofTests(unittest.TestCase):
    def test_exact_host_algebra_boundaries_and_independently_owned_receipt(self):
        receipt=core.prefix_proof()
        self.assertTrue(receipt['verified'])
        self.assertFalse(receipt['gpu_execution_verified'])
        self.assertEqual(receipt['host_halfwarp_masks_checked'],131072)
        self.assertEqual(receipt['additional_shared_device_bytes'],0)
        self.assertEqual((receipt['channels'],receipt['logical_groups'],receipt['warps']),(128,4,1))
        receipt['verified']=False
        self.assertTrue(core.prefix_proof()['verified'])
        words=core.prefix_cases(17)
        bases=np.full(len(words),(1<<31)-129,dtype=np.int32)
        actual,total=core._numpy_slots(words,bases)
        bits=((words[:,:,None]>>np.arange(32,dtype=np.uint32))&1).reshape(-1,128)
        self.assertTrue(np.array_equal(actual,bases[:,None]+np.cumsum(bits,axis=1,dtype=np.int32)-1))
        self.assertTrue(np.array_equal(total,bits.sum(axis=1)))
        self.assertEqual(int(actual[0,0]),int(bases[0])-1)
        self.assertEqual(int(actual[1,-1]),int(bases[1])+127)

    def test_only_prefix_helper_changes_in_take_word_and_layer_kernels(self):
        old=function(gc.__file__,'_take')
        new=function(core.__file__,'_take_prefix');new.name=old.name
        new=Rename().visit(new)
        self.assertEqual(ast.dump(old),ast.dump(new))
        old=function(inverse.__file__,'_dec_value_core')
        new=function(core.__file__,'_prefix_dec_value');new.name=old.name
        self.assertEqual(ast.dump(Rename().visit(old)),ast.dump(Rename().visit(new)))
        old=function(inverse.__file__,'_core_ldec_kernel')
        new=function(core.__file__,'_prefix_ldec_kernel');new.name=old.name
        removed=[n for n in new.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                 and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='static_assert']
        self.assertEqual(len(removed),2)
        new.body=[n for n in new.body if n not in removed]
        self.assertEqual(ast.dump(Rename().visit(old)),ast.dump(Rename().visit(new)))
        helper=function(core.__file__,'_slots_warp')
        asm=[n for n in ast.walk(helper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr=='inline_asm_elementwise']
        self.assertEqual(len(asm),1)
        kw={x.arg:ast.literal_eval(x.value) for x in asm[0].keywords if x.arg in ('is_pure','pack')}
        self.assertEqual(kw,dict(is_pure=False,pack=1))
        self.assertFalse(any(isinstance(n,ast.If) for n in ast.walk(helper)))

    def test_actual_compiled_layout_contract_fails_closed(self):
        receipt=core.validate_compiled_prefix(fake_module())
        self.assertTrue(receipt['verified'])
        self.assertEqual((receipt['prefix_call_sites'],receipt['ballots']),(2,8))
        mutations=[('ttgir','register = []','register = [[1]]'),
                   ('ttgir','lane = [[1], [2], [4], [8], [16]]','lane = [[2], [1], [4], [8], [16]]'),
                   ('ttgir','tensor<32xi32','tensor<64xi32'),
                   ('ttgir','pure = false','pure = true'),
                   ('ttgir','packed_element = 1','packed_element = 4'),
                   ('ttgir','"ttg.num-warps" = 1','"ttg.num-warps" = 4'),
                   ('ttgir','"ttg.num-ctas" = 1','"ttg.num-ctas" = 2'),
                   ('ttgir','"ttg.threads-per-warp" = 32','"ttg.threads-per-warp" = 64'),
                   ('ttgir','cuda:86','cuda:90'),('ptx','.target sm_86','.target sm_90'),
                   ('ptx','.reqntid 32','.reqntid 128'),('ptx','vote.ballot.sync.b32','vote.ballot.b32')]
        for field,find,replace in mutations:
            with self.subTest(field=field,find=find):
                module=fake_module();module.asm[field]=module.asm[field].replace(find,replace)
                with self.assertRaises(RuntimeError):core.validate_compiled_prefix(module)
        with self.assertRaises(RuntimeError):core.validate_compiled_prefix(fake_module(),expected_calls=1)
        with self.assertRaises(RuntimeError):core.runtime_prefix_proof('cpu')


def install(native,coarse,sparse):
    original=native._launch
    verified={}
    def launch(kernel,grid,*args,**kwargs):
        if kernel is lw._ldec_kernel:
            expanded=args[:20]+(coarse,sparse)+args[20:]
            options=dict(kwargs,PACKED=True)
            warps=options.pop('warps',1)
            if warps != 1:raise ValueError('prefix fixture requires one warp')
            module=core._prefix_ldec_kernel.warmup(*expanded,grid=grid,num_warps=warps,
                **native.opts,**options)
            if module.hash not in verified:
                verified[module.hash]=core.validate_compiled_prefix(module)
            return original(core._prefix_ldec_kernel,grid,*expanded,warps=warps,**options)
        return original(kernel,grid,*args,**kwargs)
    native._launch=launch
    return verified


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET')!='1','actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_masks_changing_iterations_exact_take_states_and_layout(self):
        self.assertIn('RTX A4500',torch.cuda.get_device_name(0))
        receipt=core.runtime_prefix_proof('cuda:0')
        self.assertTrue(receipt['verified'] and receipt['gpu_execution_verified'])
        self.assertEqual(receipt['changing_steps'],3)
        words=np.zeros((131072,4),dtype=np.uint32)
        words[:65536,0]=np.arange(65536,dtype=np.uint32)
        words[65536:,0]=np.arange(65536,dtype=np.uint32)<<16
        words[:,1]=~words[:,0];words[:,2]=0xffffffff;words[:,3]=words[:,0]^0x55555555
        exhaustive=core._run_prefix_probe('cuda:0',words,steps=1)
        self.assertEqual(exhaustive['prefix_calls'],131072)
        rng=np.random.default_rng(917)
        n=1024;shape=(n,128)
        frequency=rng.integers(1,65536,size=shape,dtype=np.uint32)
        frequency[:4]=np.asarray([1,2,3,65535],dtype=np.uint32)[:,None]
        cumulative=rng.integers(0,65536,size=shape,dtype=np.uint32)%(65537-frequency)
        delta=rng.integers(0,65536,size=shape,dtype=np.uint32)%frequency
        high=rng.integers(1,65536,size=shape,dtype=np.uint32)
        high[:,::2]=1
        x=(high<<16)|(cumulative+delta)
        active=rng.integers(0,2,size=shape,dtype=np.int32);active[:4]=1
        inputs=[torch.from_numpy(value.view(np.int32).copy()).to('cuda:0')
                for value in (x,cumulative,frequency,active)]
        buffer=(torch.arange(n*128,device='cuda:0',dtype=torch.int32)*997+65521).to(torch.int16)
        actual=torch.empty((n,129),device='cuda:0',dtype=torch.int32)
        reference=torch.empty_like(actual)
        module=core._probe_take_kernel.warmup(*inputs,buffer,actual,reference,grid=(n,),num_warps=1,
            num_stages=3,enable_fp_fusion=False)
        proof=core.validate_compiled_prefix(module,expected_calls=1)
        self.assertTrue(proof['verified'])
        core._probe_take_kernel[(n,)](*inputs,buffer,actual,reference,num_warps=1,
            num_stages=3,enable_fp_fusion=False)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual,reference))

    def test_actual_gpu_original_uniform_exact_packets_and_all_bf16_patterns(self):
        self.assertIn('RTX A4500',torch.cuda.get_device_name(0))
        core.runtime_prefix_proof('cuda:0')
        coarse,sparse,_=inverse.build_inverse_tables('cuda:0')
        for mode in ('uniform','exact'):
            native,words=native_fixture('cuda:0',mode)
            encoded,starts=native.encode(words)
            original,original_starts=encoded.clone(),starts.clone()
            expected=native.decode(encoded,starts,64)
            verified=install(native,coarse,sparse)
            first=native.decode(encoded,starts,64);later=native.decode(encoded,starts,64)
            torch.cuda.synchronize()
            self.assertTrue(verified)
            for value in (expected,first,later):self.assertTrue(torch.equal(value,words))
            self.assertNotEqual(first.data_ptr(),later.data_ptr())
            self.assertTrue(torch.equal(encoded,original))
            self.assertTrue(torch.equal(starts,original_starts))
        constants={name:value.to('cuda:0') for name,value in fixed_state(layers=1,tokens=256).items()}
        native=lw.LayerwiseOneStep(constants,128,128,tokens_per_program=32,chunks=2,mode='uniform',search=True)
        words=torch.arange(65536,device='cuda:0',dtype=torch.int32).to(torch.int16).reshape(1,256,256)
        encoded,starts=native.encode(words)
        original,original_starts=encoded.clone(),starts.clone()
        verified=install(native,coarse,sparse)
        result=native.decode(encoded,starts,256);torch.cuda.synchronize()
        self.assertTrue(verified)
        self.assertTrue(torch.equal(result,words))
        self.assertTrue(torch.equal(encoded,original))
        self.assertTrue(torch.equal(starts,original_starts))


if __name__=='__main__':unittest.main()
