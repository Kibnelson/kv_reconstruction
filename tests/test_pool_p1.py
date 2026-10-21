"""Independent P1 byte tests, opt-in kernel interpreter, and real CUDA checks."""
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

from qwen_kv_bench import pool_p1 as p

try:
    import torch
except ImportError:
    torch=None


def sample():
    g=p.Layout(2,7,2,16,4);rng=np.random.default_rng(43)
    refs=rng.integers(20,230,(g.layers,g.channels),dtype=np.uint8)
    widths=np.tile(np.arange(1,9,dtype=np.uint8),(g.layers,g.channels//8))
    exps=np.clip(refs[:,None].astype(np.int32)+rng.integers(-2,3,g.shape),0,255)
    w=((exps<<7)|rng.integers(0,128,g.shape)|(rng.integers(0,2,g.shape)<<15)).astype(np.uint16)
    w[0,0,3]=0x7fff;w[1,6,40]=0xffff
    return g,w,p.calibration(refs,widths)


def interpreted_encode(words,state,g,capacity):
    """Test-only direct kernel calls. Public API never accepts a CPU device."""
    from qwen_kv_bench import pool_p1_kernels as k
    W=torch.from_numpy(words.copy());R=torch.from_numpy(state.references.copy());B=torch.from_numpy(state.widths.copy())
    desc=torch.empty((g.sectors,4),dtype=torch.int64);prefix=torch.empty(g.sectors,dtype=torch.int64)
    backing=torch.full((capacity+32,),0x5b,dtype=torch.uint8);arena=backing[:capacity]
    c=dict(T=g.tokens,C=g.channels,HEADS2=2*g.heads,D=g.head_dim,TILE=g.tile_tokens,PAGES=g.pages,N=g.sector_words)
    k.count_sectors[(g.sectors,)](W,R,B,desc,**c)
    torch.cumsum(desc[:,1],dim=0,out=prefix)
    k.place_offsets[((g.sectors+255)//256,)](desc,prefix,g.sectors,256)
    k.write_sectors[(g.sectors,)](W,R,B,desc,arena,arena.view(torch.uint16),arena.view(torch.int32),CAPACITY=capacity,**c)
    np.testing.assert_array_equal(backing[capacity:].numpy(),0x5b)
    return arena,desc,R,B,c


def interpreted_unpack(values,ids,g):
    from qwen_kv_bench import pool_p1_kernels as k
    a,desc,R,B,c=values;ids=torch.as_tensor(ids,dtype=torch.int64)
    out=torch.empty((len(ids),g.tile_tokens,g.head_dim),dtype=torch.uint16);status=torch.empty(len(ids),dtype=torch.int32)
    k.unpack_sectors[(len(ids),)](a,a.view(torch.uint16),a.view(torch.int32),desc,R,B,ids,out,status,CAPACITY=a.numel(),SECTORS=g.sectors,**c)
    return out.numpy(),status.numpy()


class ReferenceTests(unittest.TestCase):
    def test_mixed_widths_unaligned_fields_and_partial_tail(self):
        g,w,s=sample();a=p.reference_encode(w,s,g)
        ids=np.array([g.sectors-1,0,3,3,7],np.int64)
        np.testing.assert_array_equal(p.reference_unpack(a,s,ids),p.original_tiles(w,g,ids))
        self.assertTrue(np.all(a.descriptors[:,0]%16==0));self.assertTrue(np.all(a.descriptors[:,1]%16==0))
        self.assertGreater(np.count_nonzero(a.descriptors[:,2]==1),0)
        self.assertGreater(a.descriptors[:,3].sum(),0)

    def test_all_65536_patterns_through_packed_escape_path(self):
        w=np.empty(131072,np.uint16);w[::2]=127<<7;w[1::2]=np.arange(65536,dtype=np.uint16)
        g=p.Layout(1,512,8,16,4);w=w.reshape(g.shape)
        s=p.calibration(np.full((1,g.channels),127,np.uint8),np.ones((1,g.channels),np.uint8))
        a=p.reference_encode(w,s,g)
        self.assertTrue(np.all(a.descriptors[:,2]==1))
        ids=np.arange(g.sectors);np.testing.assert_array_equal(p.reference_unpack(a,s,ids),p.original_tiles(w,g,ids))

    def test_raw_fallback_preserves_every_pattern(self):
        g=p.Layout(1,256,8,16,4);w=np.arange(65536,dtype=np.uint16).reshape(g.shape)
        s=p.calibration(np.full((1,g.channels),127,np.uint8),np.full((1,g.channels),8,np.uint8))
        a=p.reference_encode(w,s,g);self.assertTrue(np.all(a.descriptors[:,2]==0))
        ids=np.arange(g.sectors);np.testing.assert_array_equal(p.reference_unpack(a,s,ids),p.original_tiles(w,g,ids))

    def test_capacity_exhaustion_is_unpublished(self):
        g,w,s=sample();a=p.reference_encode(w,s,g,capacity_bytes=16)
        self.assertTrue(np.all(a.descriptors[:,2]==-1))
        with self.assertRaisesRegex(ValueError,'unpublished'):p.reference_unpack(a,s,[0])
        np.testing.assert_array_equal(w,sample()[1])

    def test_bad_calibration_and_geometry(self):
        for value in [0,9,-1]:
            with self.assertRaises(ValueError):p.calibration([[127]],[[value]])
        with self.assertRaises(ValueError):p.calibration([[256]],[[2]])
        with self.assertRaises(ValueError):p.Layout(1,2,1,17)
        with self.assertRaises(ValueError):p.Layout(1,2,1,16,3)
        with self.assertRaises(ValueError):p.Layout(True,2,1,16)

    def test_wrong_state_and_corrupt_descriptors(self):
        g,w,s=sample();a=p.reference_encode(w,s,g)
        other=p.calibration(s.references,(s.widths%8)+1)
        with self.assertRaisesRegex(ValueError,'identity'):p.reference_unpack(a,other,[0])
        for field,value in [(0,-16),(1,17),(2,3),(3,999999)]:
            b=p.reference_encode(w,s,g);b.descriptors[0,field]=value
            with self.assertRaises(ValueError):p.reference_unpack(b,s,[0])
        for sid in [-1,g.sectors]:
            with self.assertRaises(ValueError):p.reference_unpack(a,s,[sid])

    def test_cuda_api_never_labels_cpu_as_gpu(self):
        _,_,s=sample()
        with self.assertRaises(RuntimeError):p.to_cuda(s,'cpu')
        if torch is None or not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):p.to_cuda(s,'cuda:0')


INTERPRETER=os.environ.get('TRITON_INTERPRET')=='1' and torch is not None and importlib.util.find_spec('triton') is not None


@unittest.skipUnless(INTERPRETER,'opt-in Triton CPU interpreter; not a CUDA result')
class KernelInterpreterTests(unittest.TestCase):
    def test_kernel_bytes_and_tiles_match_independent_reference(self):
        g,w,s=sample();a=p.reference_encode(w,s,g,capacity_bytes=g.padded_raw_bytes)
        values=interpreted_encode(w,s,g,g.padded_raw_bytes);arena,desc=values[:2]
        np.testing.assert_array_equal(desc.numpy(),a.descriptors)
        used=int(a.descriptors[:,1].sum());np.testing.assert_array_equal(arena.numpy()[:used],a.payload[:used])
        ids=[g.sectors-1,0,3,3,7];out,status=interpreted_unpack(values,ids,g)
        np.testing.assert_array_equal(status,0);np.testing.assert_array_equal(out,p.original_tiles(w,g,ids))

    def test_kernel_raw_all_patterns(self):
        g=p.Layout(1,256,8,16,4);w=np.arange(65536,dtype=np.uint16).reshape(g.shape)
        s=p.calibration(np.full((1,g.channels),127,np.uint8),np.full((1,g.channels),8,np.uint8))
        v=interpreted_encode(w,s,g,g.padded_raw_bytes);ids=np.arange(g.sectors)
        out,status=interpreted_unpack(v,ids,g);np.testing.assert_array_equal(status,0)
        np.testing.assert_array_equal(out,p.original_tiles(w,g,ids))

    def test_kernel_packed_all_patterns(self):
        g=p.Layout(1,512,8,16,4);w=np.empty(131072,np.uint16)
        w[::2]=127<<7;w[1::2]=np.arange(65536,dtype=np.uint16);w=w.reshape(g.shape)
        s=p.calibration(np.full((1,g.channels),127,np.uint8),np.ones((1,g.channels),np.uint8))
        v=interpreted_encode(w,s,g,g.padded_raw_bytes);ids=np.arange(g.sectors)
        self.assertTrue(np.all(v[1][:,2].numpy()==1))
        out,status=interpreted_unpack(v,ids,g);np.testing.assert_array_equal(status,0)
        np.testing.assert_array_equal(out,p.original_tiles(w,g,ids))

    def test_kernel_capacity_and_invalid_readers(self):
        g,w,s=sample();v=interpreted_encode(w,s,g,16)
        out,status=interpreted_unpack(v,[0,-1,g.sectors],g)
        self.assertTrue(np.all(status!=0));np.testing.assert_array_equal(out,0)
        for field,value in [(0,-16),(1,17),(2,3),(3,999999)]:
            v=interpreted_encode(w,s,g,g.padded_raw_bytes);v[1][0,field]=value
            out,status=interpreted_unpack(v,[0],g)
            self.assertNotEqual(status[0],0);np.testing.assert_array_equal(out,0)

    def test_qwen_head_dimension_and_large_scan(self):
        g=p.Layout(2,19,2,128,16);rng=np.random.default_rng(72)
        refs=np.full((g.layers,g.channels),127,np.uint8)
        widths=np.tile(np.arange(1,9,dtype=np.uint8),(g.layers,g.channels//8))
        exp=127+rng.integers(-2,3,g.shape)
        w=((exp<<7)|rng.integers(0,128,g.shape)|(rng.integers(0,2,g.shape)<<15)).astype(np.uint16)
        s=p.calibration(refs,widths);a=p.reference_encode(w,s,g,capacity_bytes=g.padded_raw_bytes)
        v=interpreted_encode(w,s,g,g.padded_raw_bytes)
        np.testing.assert_array_equal(v[1].numpy(),a.descriptors)
        n=int(a.descriptors[:,1].sum());np.testing.assert_array_equal(v[0][:n].numpy(),a.payload[:n])
        ids=np.arange(g.sectors)[::-1].copy();out,status=interpreted_unpack(v,ids,g)
        np.testing.assert_array_equal(status,0);np.testing.assert_array_equal(out,p.original_tiles(w,g,ids))

    def test_actual_frozen_page_kernel_byte_and_word_parity(self):
        root=Path(__file__).resolve().parents[1]/'benchmark_fixtures/team_transfer_smol_replay_v1'
        if not (root/'page_000_words.npy').is_file():self.skipTest('optional frozen fixture not present')
        w=np.load(root/'page_000_words.npy',allow_pickle=False);g=p.Layout(30,32,3,64)
        with np.load(root/'pool_state.npz',allow_pickle=False) as z:s=p.calibration(z['refs'],z['widths'])
        a=p.reference_encode(w,s,g,capacity_bytes=g.padded_raw_bytes);v=interpreted_encode(w,s,g,g.padded_raw_bytes)
        np.testing.assert_array_equal(v[1].numpy(),a.descriptors)
        n=int(a.descriptors[:,1].sum());np.testing.assert_array_equal(v[0][:n].numpy(),a.payload[:n])
        ids=np.arange(g.sectors);out,status=interpreted_unpack(v,ids,g)
        np.testing.assert_array_equal(status,0);np.testing.assert_array_equal(out,p.original_tiles(w,g,ids))


CUDA=bool(torch is not None and torch.cuda.is_available() and importlib.util.find_spec('triton') is not None)


@unittest.skipUnless(CUDA,'actual CUDA hardware unavailable; no CPU fallback')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_word_and_byte_parity(self):
        g,w,s=sample();oracle=p.reference_encode(w,s,g,capacity_bytes=g.padded_raw_bytes)
        device=p.to_cuda(s);arena=p.allocate_arena(g,device,g.padded_raw_bytes)
        p.encode_into(torch.from_numpy(w).cuda(),device,arena)
        np.testing.assert_array_equal(arena.descriptors.cpu().numpy(),oracle.descriptors)
        used=int(oracle.descriptors[:,1].sum())
        np.testing.assert_array_equal(arena.payload[:used].cpu().numpy(),oracle.payload[:used])
        ids=torch.tensor([g.sectors-1,0,3,3,7],device='cuda',dtype=torch.int64)
        out,status=p.unpack_tiles(arena,device,ids)
        np.testing.assert_array_equal(status.cpu().numpy(),0)
        np.testing.assert_array_equal(out.cpu().numpy(),p.original_tiles(w,g,ids.cpu().numpy()))

    def test_actual_gpu_capacity_is_unpublished(self):
        g,w,s=sample();device=p.to_cuda(s);a=p.allocate_arena(g,device,16)
        p.encode_into(torch.from_numpy(w).cuda(),device,a)
        out,status=p.unpack_tiles(a,device,torch.tensor([0],device='cuda'))
        self.assertNotEqual(status.item(),0);np.testing.assert_array_equal(out.cpu().numpy(),0)


if __name__=='__main__':unittest.main()
