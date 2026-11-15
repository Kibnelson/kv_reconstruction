"""CPU failure injection for the runtime probe; no actual GPU validation here."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('peer_probe', ROOT/'scripts/pd_cudart_peer_probe.py')
peer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(peer)


class FakeRuntime:
    def __init__(self, failure=None):
        self.failure, self.buffers, self.calls = failure, {}, []
        self.next_pointer = 0

    def allocate(self, device, nbytes):
        self.next_pointer += 1
        self.buffers[self.next_pointer] = (device, np.zeros(nbytes, dtype=np.uint8))
        return self.next_pointer

    def free(self, device, pointer):
        assert self.buffers[pointer][0] == device
        del self.buffers[pointer]

    def upload(self, device, pointer, data):
        assert self.buffers[pointer][0] == device
        self.buffers[pointer][1][:] = data

    def download(self, device, pointer, nbytes):
        assert self.buffers[pointer][0] == device
        return self.buffers[pointer][1].copy()

    def local_copy(self, device, destination, source, nbytes):
        assert self.buffers[source][0] == self.buffers[destination][0] == device
        self.buffers[destination][1][:] = self.buffers[source][1]
        if self.failure == 'local':self.buffers[destination][1][-1] ^= 1

    def sync(self, device):
        self.calls.append(('sync',device))

    def peer_supported(self, source, destination):
        return self.failure != 'unavailable'

    def enable_peer(self, source, destination):
        self.calls.append(('enable',source,destination))
        return 'enabled'

    def peer_copy(self, source_device, source, destination_device, destination, nbytes):
        self.calls.append(('peer',source_device,destination_device))
        if self.failure == 'skip':return
        self.buffers[destination][1][:] = self.buffers[source][1]
        if self.failure == 'tail':self.buffers[destination][1][-1] ^= 1
        if self.failure == 'source':self.buffers[source][1][0] ^= 1


class PeerProbeTests(unittest.TestCase):
    def test_real_worker_without_visible_gpus_keeps_failure_json_without_torch_import(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'failure.json'
            child=subprocess.run([sys.executable,str(ROOT/'scripts/pd_cudart_peer_probe.py'),
                                  '--mib','1','--pairs','0-1','--output',str(output)],
                                 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_LAUNCH_BLOCKING='1'),
                                 text=True,capture_output=True,timeout=30)
            self.assertEqual(child.returncode,1,child.stdout+child.stderr)
            record=json.loads(output.read_text())
            self.assertFalse(record['torch_imported'])
            self.assertFalse(record['gpu_executed'])
            self.assertEqual(record['status'],'failed')
            self.assertIn('error',record)

    def test_complete_buffers_and_poison_are_checked(self):
        expected=np.arange((1<<20)+17,dtype=np.uint32).astype(np.uint8)
        actual=expected.copy()
        self.assertTrue(peer.comparison(expected,actual)['correct'])
        actual[-1]^=1
        stats=peer.comparison(expected,actual)
        self.assertEqual((stats['mismatched_bytes'],stats['first_mismatch']),(1,len(expected)-1))
        self.assertNotEqual(stats['expected_sha256'],stats['received_sha256'])
        stats=peer.comparison(expected,np.bitwise_not(expected))
        self.assertEqual(stats['untouched_poison_bytes'],len(expected))
        self.assertEqual(stats['mismatched_bytes'],len(expected))

    def test_runtime_copy_succeeds_only_with_both_controls_and_fresh_trials(self):
        runtime=FakeRuntime();record={}
        peer.probe(runtime,0,1,97,2,record)
        self.assertTrue(record['pairs'][0]['correct'])
        self.assertEqual(record['completed_reps'],2)
        self.assertEqual(len(record['host_controls']),4)
        self.assertEqual(len(record['peer_enable']),2)
        self.assertNotEqual(*[row['expected_sha256'] for row in record['trials']])
        self.assertTrue(all(row['source_unchanged'] for row in record['trials']))
        self.assertEqual(runtime.buffers,{})

    def test_bad_peer_source_and_local_copies_cannot_pass(self):
        for failure in ('tail','skip','source','local','unavailable'):
            with self.subTest(failure=failure):
                runtime=FakeRuntime(failure);record={}
                with self.assertRaises(RuntimeError):peer.probe(runtime,1,2,97,2,record)
                self.assertFalse(record['pairs'][0]['correct'])
                self.assertEqual(record['completed_reps'],0)
                self.assertEqual(runtime.buffers,{})
                if failure in ('local','unavailable'):
                    self.assertFalse(any(call[0]=='peer' for call in runtime.calls))
                if failure=='skip':self.assertEqual(record['trials'][0]['untouched_poison_bytes'],97)
                if failure=='source':self.assertFalse(record['trials'][0]['source_unchanged'])

    def test_peer_call_uses_explicit_device_ids_full_size_and_pointer_width(self):
        calls=[]
        class Function:
            def __init__(self,name):self.name=name
            def __call__(self,*args):
                calls.append((self.name,args))
                return b'fixture CUDA error' if self.name=='cudaGetErrorString' else 0
        class Library:
            def __getattr__(self,name):
                value=Function(name);setattr(self,name,value);return value
        library=Library()
        with patch.object(peer.ctypes,'CDLL',return_value=library):runtime=peer.Runtime('fixture')
        runtime.peer_copy(0,0x123456780,1,0x987654320,1<<30)
        self.assertEqual(calls,[('cudaSetDevice',(1,)),('cudaMemcpyPeer',(0x987654320,1,0x123456780,0,1<<30))])
        self.assertEqual(library.cudaMemcpyPeer.argtypes,
                         [ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t])
        library.cudaMemcpyPeer=lambda *args:999
        with self.assertRaisesRegex(RuntimeError,'cudaMemcpyPeer: CUDA 999'):
            runtime.peer_copy(0,1,1,2,97)

    def test_bundled_runtime_is_resolved_from_package_metadata_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            path=root/'libcudart.so.12';path.write_bytes(b'CPU fixture; never loaded')
            distribution=SimpleNamespace(files=[Path('libcudart.so.12'),Path('libcudart.so.13')],
                                         locate_file=lambda file:root/file)
            with patch.object(peer.importlib.metadata,'distribution',return_value=distribution):
                self.assertEqual(peer.runtime_library(),path)
            second=root/'libcudart.so.12.1';second.write_bytes(b'another fixture')
            distribution.files.append(Path(second.name))
            with patch.object(peer.importlib.metadata,'distribution',return_value=distribution),self.assertRaises(RuntimeError):
                peer.runtime_library()


if __name__ == '__main__':
    unittest.main()
