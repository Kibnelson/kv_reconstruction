import importlib.util
import os
import unittest
import numpy as np
from qwen_kv_bench import pool_p1 as p, pool_attention as a
try:
    import torch
except ImportError:
    torch = None


def bits(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)


def sample():
    g = p.Layout(2,7,2,16,4); rng = np.random.default_rng(204)
    w = bits(rng.normal(size=g.shape)); tail = bits(rng.normal(size=(2,3,g.channels)))
    q = rng.normal(size=(2,8,16)).astype(np.float32)
    s = p.calibration(np.full((2,g.channels),127,np.uint8),np.tile(np.arange(1,9,dtype=np.uint8),(2,g.channels//8)))
    return g,w,tail,q,s


class ReferenceTests(unittest.TestCase):
    def test_zero_query_is_mean_and_gqa_maps_heads(self):
        g,w,t,q,s = sample(); q.fill(0)
        out = a.reference(q,w,g,t)
        values = (np.concatenate((w,t),1).astype(np.uint32)<<16).view(np.float32)
        expected = values[:,:,g.channels//2:].reshape(2,-1,2,16).mean(1,dtype=np.float64)
        np.testing.assert_allclose(out,np.repeat(expected,4,axis=1),rtol=0,atol=1e-15)

    def test_rejects_wrong_gqa_and_nonfinite_inputs(self):
        g,w,t,q,s = sample()
        with self.assertRaises(ValueError):a.reference(q[:,:3],w,g,t)
        w[0,0,0]=0x7fc0
        with self.assertRaises(ValueError):a.reference(q,w,g,t)


INTERPRETER=os.environ.get('TRITON_INTERPRET')=='1' and torch is not None and importlib.util.find_spec('triton') is not None
CUDA=torch is not None and torch.cuda.is_available() and importlib.util.find_spec('triton') is not None


@unittest.skipUnless(INTERPRETER,'CPU interpreter is explicit; not CUDA evidence')
class InterpreterTests(unittest.TestCase):
    def test_fused_loader_dense_and_raw_tail_match_independent_attention(self):
        from tests.test_pool_p1 import interpreted_encode
        from qwen_kv_bench.pool_attention_kernels import attention
        g,w,t,q,s=sample(); arena,desc,refs,widths,_=interpreted_encode(w,s,g,g.padded_raw_bytes)
        query=torch.from_numpy(q);words=torch.from_numpy(w);tail=torch.from_numpy(t)
        outputs=[]
        for packed in [False,True]:
            out=torch.empty_like(query);status=torch.empty(q.shape[:2],dtype=torch.int32)
            attention[(g.layers,q.shape[1])](query,words,tail,arena,arena.view(torch.uint16),arena.view(torch.int32),
                desc,refs,widths,out,status,PACKED=packed,CAPACITY=arena.numel(),T=g.tokens,C=g.channels,
                HEADS=g.heads,QHEADS=q.shape[1],D=g.head_dim,TILE=g.tile_tokens,PAGES=g.pages,
                N=g.sector_words,SECTORS=g.sectors,TAIL=t.shape[1],SCALE=g.head_dim**-.5)
            np.testing.assert_array_equal(status.numpy(),0);outputs.append(out.numpy().copy())
        np.testing.assert_array_equal(outputs[0],outputs[1])
        np.testing.assert_allclose(outputs[1],a.reference(q,w,g,t),rtol=2e-6,atol=2e-6)

    def test_corrupt_or_unpublished_pool_returns_error_without_dense_recovery(self):
        from tests.test_pool_p1 import interpreted_encode
        from qwen_kv_bench.pool_attention_kernels import attention
        g,w,t,q,s=sample();arena,desc,refs,widths,_=interpreted_encode(w,s,g,g.padded_raw_bytes)
        desc[0,2]=-1;query=torch.from_numpy(q);out=torch.empty_like(query);status=torch.empty(q.shape[:2],dtype=torch.int32)
        attention[(g.layers,q.shape[1])](query,torch.from_numpy(w),torch.from_numpy(t),arena,arena.view(torch.uint16),arena.view(torch.int32),
            desc,refs,widths,out,status,PACKED=True,CAPACITY=arena.numel(),T=g.tokens,C=g.channels,
            HEADS=g.heads,QHEADS=q.shape[1],D=g.head_dim,TILE=g.tile_tokens,PAGES=g.pages,
            N=g.sector_words,SECTORS=g.sectors,TAIL=t.shape[1],SCALE=g.head_dim**-.5)
        np.testing.assert_array_equal(status[0,:4].numpy(),1);np.testing.assert_array_equal(out[0,:4].numpy(),0)
        np.testing.assert_array_equal(status[1].numpy(),0)
        selected=torch.empty((1,q.shape[1],g.head_dim),dtype=torch.float32);selected_status=torch.empty((1,q.shape[1]),dtype=torch.int32)
        attention[(1,q.shape[1])](query[1:2],torch.from_numpy(w),torch.from_numpy(t),arena,arena.view(torch.uint16),arena.view(torch.int32),
            desc,refs,widths,selected,selected_status,PACKED=True,CAPACITY=arena.numel(),T=g.tokens,C=g.channels,
            HEADS=g.heads,QHEADS=q.shape[1],D=g.head_dim,TILE=g.tile_tokens,PAGES=g.pages,
            N=g.sector_words,SECTORS=g.sectors,TAIL=t.shape[1],SCALE=g.head_dim**-.5,LSTART=1)
        np.testing.assert_array_equal(selected_status.numpy(),0)
        np.testing.assert_allclose(selected.numpy(),a.reference(q[1:2],w,g,t,layer_start=1),rtol=2e-6,atol=2e-6)


@unittest.skipUnless(CUDA,'actual CUDA unavailable')
class CudaTests(unittest.TestCase):
    def test_actual_attention_uses_packed_prefix_and_raw_tail(self):
        g,w,t,q,s=sample();ds=p.to_cuda(s);ar=p.allocate_arena(g,ds,g.padded_raw_bytes)
        source=torch.from_numpy(w).cuda();p.encode_into(source,ds,ar)
        query=torch.from_numpy(q).cuda();tail=torch.from_numpy(t).cuda()
        dense,st=a.dense_attention(query,source,g,tail);packed,sp=a.packed_attention(query,ar,ds,tail)
        np.testing.assert_array_equal(st.cpu().numpy(),0);np.testing.assert_array_equal(sp.cpu().numpy(),0)
        np.testing.assert_allclose(dense.cpu().numpy(),packed.cpu().numpy(),rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(packed.cpu().numpy(),a.reference(q,w,g,t),rtol=3e-6,atol=3e-6)
        ar.descriptors[0,2]=-1
        layer,st=a.packed_attention(query[1:2],ar,ds,tail,layer_start=1)
        np.testing.assert_array_equal(st.cpu().numpy(),0)
        np.testing.assert_allclose(layer.cpu().numpy(),a.reference(q[1:2],w,g,t,layer_start=1),rtol=3e-6,atol=3e-6)


if __name__=='__main__':unittest.main()
