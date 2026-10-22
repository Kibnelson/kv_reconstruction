"""Build-tool provisioning checks; no downloads or GPU measurements."""
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

from scripts import setup_qwen_cuda_toolchain as setup
from qwen_kv_bench import quanto_unpack


class ToolchainTests(unittest.TestCase):
    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'bad.tar.xz'
            with tarfile.open(archive,'w:xz') as t:
                m=tarfile.TarInfo('cuda/../../escaped');m.size=3;t.addfile(m,io.BytesIO(b'bad'))
            with self.assertRaises(tarfile.FilterError):setup.extract_package(archive,root/'target')
            self.assertFalse((root/'escaped').exists())

    def test_compose_hardlinked_tools_library_alias_and_quoted_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'tools.tar.xz'
            with tarfile.open(archive,'w:xz') as t:
                entries=['bin/nvcc','bin/ptxas','nvvm/bin/cicc','include/cuda_runtime.h',
                    'lib/libcudadevrt.a','lib/libcudart_static.a','include/cub/cub.cuh']
                for name in entries:
                    m=tarfile.TarInfo('cuda/'+name);m.size=3;m.mode=0o644;t.addfile(m,io.BytesIO(b'abc'))
                m=tarfile.TarInfo('cuda/bin/nvlink');m.type=tarfile.LNKTYPE;m.linkname='cuda/bin/ptxas';t.addfile(m)
            component=setup.extract_package(archive,root/'components')
            self.assertTrue(os.access(component/'bin/nvlink',os.X_OK))
            toolkit=setup.compose_toolkit([component],root/"tools with 'quotes' $literal")
            self.assertTrue((toolkit/'lib64/libcudart_static.a').is_file())
            self.assertEqual((toolkit/'lib').resolve(),(toolkit/'lib64').resolve())
            env=root/'env.sh';setup.write_env(env,toolkit)
            r=subprocess.run(['bash','-c','source "$1"; printf "%s" "$CUDA_HOME"','bash',str(env)],
                capture_output=True,text=True,check=True)
            self.assertEqual(r.stdout,str(toolkit))

    def test_corrupt_cached_archive_is_replaced_and_download_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp);path=cache/'sample-linux-x86_64-1-archive.tar.xz';path.write_bytes(b'old')
            info=dict(version='1',bytes=4,sha256=hashlib.sha256(b'good').hexdigest())
            with mock.patch.object(setup.urllib.request,'urlopen',return_value=io.BytesIO(b'good')):
                self.assertEqual(setup.download_package('sample',info,cache).read_bytes(),b'good')
            path.unlink()
            with mock.patch.object(setup.urllib.request,'urlopen',return_value=io.BytesIO(b'evil')):
                with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):setup.download_package('sample',info,cache)
            self.assertFalse(path.exists())

    def test_existing_cuda12_toolkit_does_not_download_or_change_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);toolkit=root/'existing';env=root/'activate.sh'
            with mock.patch.object(setup,'existing_toolkit',return_value=(toolkit,'release 12.8')), \
                    mock.patch.object(setup,'download_package',side_effect=AssertionError('no download')):
                result=setup.prepare(root/'report',env)
            self.assertEqual(result['origin'],'existing CUDA 12.8 toolkit');self.assertEqual(result['packages'],{})
            self.assertTrue(env.is_file())

    def test_reject_unpinned_quanto_source(self):
        with self.assertRaisesRegex(ValueError,'differs from pinned'):quanto_unpack.patched_source(b'changed')

    @unittest.skipUnless(os.environ.get('QUANTO_UNPACK_VALIDATION_SOURCE'),'optional pinned native source absent')
    def test_quanto_gpu_kernel_bodies_preserved_while_host_stream_is_corrected(self):
        source=Path(os.environ['QUANTO_UNPACK_VALIDATION_SOURCE']).read_bytes()
        patched=quanto_unpack.patched_source(source)
        pattern=r'__global__ void unpack_\dbit_kernel\([^}]+\}'
        self.assertEqual(re.findall(pattern,source.decode()),re.findall(pattern,patched))
        self.assertEqual(len(re.findall(pattern,patched)),2)
        self.assertNotIn('<<<blocks, BLOCK_SIZE>>>',patched)
        self.assertEqual(patched.count('<<<blocks, BLOCK_SIZE, 0, stream>>>'),2)
        self.assertIn('cudaSetDevice(device)',patched)
