"""Capture import, accounting and strict GPU-reporting contracts."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import bench_pool_p1 as bench


def captures(root):
    root.mkdir();entries=[]
    for split in ['train','test']:
        path=root/(split+'.npz')
        words=np.full((2,16,8),127<<7,dtype=np.uint16)
        np.savez(path,K_bits=words,V_bits=words,Hk=np.array(1),hd=np.array(8))
        entries.append(dict(split=split,file=path.name,name=split,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (root/'capture_manifest.json').write_text(json.dumps(dict(format='block32_probability_captures_v1',entries=entries,
        model_identity=dict(model='synthetic runner contract fixture; not pretrained'))))


class RunnerTests(unittest.TestCase):
    def test_imported_captures_cpu_mode_and_complete_arena_accounting(self):
        with tempfile.TemporaryDirectory() as private:
            root=Path(private);data=root/'captures';captures(data);output=root/'report'
            with contextlib.redirect_stdout(io.StringIO()):
                code=bench.main(['--captures-from',str(data),'--cpu-oracle','--tile-tokens','4',
                    '--page-tokens','4','--page-starts=0,-1','--output',str(output)])
            self.assertEqual(code,0);r=json.loads((output/'results.json').read_text())
            self.assertFalse(r['coverage']['gpu_measured']);self.assertFalse(r['coverage']['gpu_T_to_P'])
            self.assertEqual(r['device']['selected'],'cpu_oracle');self.assertEqual(r['unique_model_words'],256)
            self.assertTrue(r['all_model_cases_have_exact_full_capacity_roundtrip'])
            for row in r['rows']:
                for a in row['arenas']:
                    self.assertIsNone(a['kv_to_p']);self.assertIsNone(a['full_sector_unpack'])
                    s=a['stats'];self.assertEqual(s['allocated_packet_bytes'],s['arena_capacity_bytes']+s['descriptor_bytes'])
                    self.assertEqual(s['allocated_packet_plus_calibration_bytes'],s['allocated_packet_bytes']+s['calibration_device_bytes'])

    def test_capture_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as private:
            data=Path(private)/'captures';captures(data)
            with (data/'test.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):bench.captured_cases(data,4,4,[0])

    def test_cuda_failure_report_cannot_turn_into_cpu_success(self):
        with tempfile.TemporaryDirectory() as private:
            output=Path(private)/'gpu'
            with patch('qwen_kv_bench.pool_p1._cuda_torch',side_effect=RuntimeError('requested CUDA unavailable')),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                code=bench.main(['--output',str(output)])
            self.assertEqual(code,1);r=json.loads((output/'results.json').read_text())
            self.assertEqual(r['status'],'failed');self.assertFalse(r['coverage']['gpu_measured'])
            self.assertEqual(r['rows'],[])


if __name__=='__main__':unittest.main()
