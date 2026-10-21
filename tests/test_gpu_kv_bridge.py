import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from qwen_kv_bench import gpu_kv_bridge as b,pool_p1 as p,transfer_t1 as t1,pool_attention as attn
from tests.test_transfer_t1 import model_and_words
try:
    import torch
except ImportError:
    torch=None


class FilesystemTests(unittest.TestCase):
    def test_file_is_identical_and_existing_file_is_not_overwritten(self):
        data=b'indexed compressed bytes\x00\xff'*64
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'chunk.t1';out,stats=b.filesystem_roundtrip(data,path)
            self.assertEqual(out,data);self.assertTrue(stats['byte_identical']);self.assertEqual(stats['file_bytes'],len(data))
            with self.assertRaises(FileExistsError):b.filesystem_roundtrip(b'new data',path)
            self.assertEqual(path.read_bytes(),data)


INTERPRETER=os.environ.get('TRITON_INTERPRET')=='1' and torch is not None and importlib.util.find_spec('triton') is not None
CUDA=torch is not None and torch.cuda.is_available() and importlib.util.find_spec('triton') is not None


def exercise(case,device,interpreter=False):
    base,state,w,capture=model_and_words();e=t1.Engine(base,state,device,_interpreter=interpreter)
    g=p.Layout(2,1,1,8,4);s=p.calibration(np.full((2,16),127,np.uint8),np.full((2,16),3,np.uint8))
    ds=p.DeviceCalibration(torch.from_numpy(s.references.copy()).to(device),torch.from_numpy(s.widths.copy()).to(device),s.sha256)
    arena=p.allocate_arena(g,ds,g.padded_raw_bytes);p.encode_into(torch.from_numpy(w).to(device),ds,arena)
    packet=b.encode_pool(e,arena,ds,raw_fallback=False);wire=t1.export_packet(packet,{'model':'test'})
    del arena
    with tempfile.TemporaryDirectory() as tmp:
        external,stats=b.filesystem_roundtrip(wire,Path(tmp)/'chunk.t1')
    imported=t1.import_packet(external,e,{'model':'test'})
    installed,status=b.install_pool(e,imported,ds,g,g.padded_raw_bytes)
    np.testing.assert_array_equal(status.cpu().numpy(),0)
    ids=torch.arange(g.sectors,device=device,dtype=torch.int64);out,errors=p.unpack_tiles(installed,ds,ids)
    np.testing.assert_array_equal(errors.cpu().numpy(),0);np.testing.assert_array_equal(out.cpu().numpy(),p.original_tiles(w,g,np.arange(g.sectors)))
    q=np.arange(32,dtype=np.float32).reshape(2,2,8)/32
    output,error=attn.packed_attention(torch.from_numpy(q).to(device),installed,ds)
    np.testing.assert_array_equal(error.cpu().numpy(),0)
    np.testing.assert_allclose(output.cpu().numpy(),attn.reference(q,w,g),rtol=2e-6,atol=2e-6)
    imported.directory=imported.directory.clone();imported.directory[0,1]=3
    rejected,status=b.install_pool(e,imported,ds,g,g.padded_raw_bytes)
    case.assertTrue(bool(status.any().item()))
    np.testing.assert_array_equal(rejected.descriptors[:,2].cpu().numpy(),-1)


@unittest.skipUnless(INTERPRETER,'explicit CPU interpretation, not CUDA')
class InterpreterTests(unittest.TestCase):
    def test_pool_transfer_external_bytes_install_and_attention(self):
        # Only the CUDA-availability guard is replaced. The original P1, T1
        # and attention kernels all execute through Triton's CPU interpreter.
        with patch.object(p,'_cuda_torch',return_value=(torch,torch.device('cpu'))),patch.object(torch.cuda,'device',return_value=__import__('contextlib').nullcontext()):
            exercise(self,'cpu',True)


@unittest.skipUnless(CUDA,'actual CUDA unavailable')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_external_roundtrip_installs_attention_readable_pool(self):
        exercise(self,'cuda:0')


if __name__=='__main__':unittest.main()
