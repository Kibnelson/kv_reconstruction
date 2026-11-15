"""Software contracts for the copy probe; compiled CUDA still requires real GPUs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location('test_'+name,ROOT/'scripts'/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


class ProbeToolTests(unittest.TestCase):
    def test_serial_copy_uses_both_device_guards_and_completion(self):
        probe=module('pd_link_probe');events=[];active=[None]
        class Guard:
            def __init__(self,device):self.device=device
            def __enter__(self):self.old=active[0];active[0]=self.device
            def __exit__(self,*args):active[0]=self.old
        class Buffer:
            def __init__(self,device,data):self.device=device;self.data=bytearray(data)
            def copy_(self,other,non_blocking):
                events.append(('copy',other.device,self.device,active[0]))
                self.data[:]=other.data
        fake=SimpleNamespace(cuda=SimpleNamespace(device=Guard,synchronize=lambda device:events.append(('sync',device,active[0]))))
        src=Buffer('cuda:0',b'full payload');host=Buffer('cpu',b'0'*12);dst=Buffer('cuda:1',b'1'*12)
        with patch.object(probe,'torch',fake):probe.serial_host_copy(src,dst,host)
        self.assertEqual(events,[('copy','cuda:0','cpu','cuda:0'),('sync','cuda:0','cuda:0'),
            ('copy','cpu','cuda:1','cuda:1'),('sync','cuda:1','cuda:1')])
        self.assertEqual(dst.data,src.data)

    def test_tail_corruption_is_detected_beyond_old_sample(self):
        probe=module('pd_link_probe')
        src=torch.zeros((1<<20)+17,dtype=torch.uint8);dst=src.clone()
        self.assertTrue(probe.same_bytes(src,dst))
        dst[-1]=1
        self.assertFalse(probe.same_bytes(src,dst))

    def test_invalid_route_is_rejected_before_cuda(self):
        probe=module('pd_link_probe')
        with self.assertRaises(SystemExit) as ctx:probe.main(['--paths','unknown'])
        self.assertEqual(ctx.exception.code,2)

    def test_failed_bandwidth_child_returns_failure_and_keeps_report(self):
        specs=module('node_specs')
        with tempfile.TemporaryDirectory() as td,patch.object(specs,'system',return_value={'fixture':True}), \
                patch.object(specs,'summary',return_value='fixture report'), \
                patch.object(specs.subprocess,'run',return_value=SimpleNamespace(returncode=-6)):
            rc=specs.main(['--output',td,'--python','fixture-python'])
            report=json.loads((Path(td)/'specs.json').read_text())
            self.assertEqual(rc,1)
            self.assertIn('-6',report['bandwidth_error'])
            self.assertTrue((Path(td)/'summary.md').is_file())

    def test_inexact_pattern_returns_failure(self):
        specs=module('node_specs')
        with tempfile.TemporaryDirectory() as td:
            def child(*args,**kwargs):
                (Path(td)/'bandwidth.json').write_text(json.dumps({'patterns':[{'correct':False}]}))
                return SimpleNamespace(returncode=0)
            with patch.object(specs,'system',return_value={'fixture':True}),patch.object(specs,'summary',return_value='fixture'),patch.object(specs.subprocess,'run',side_effect=child):
                self.assertEqual(specs.main(['--output',td,'--python','fixture-python']),1)

    def test_healthy_patterns_pass_and_diagnostic_timings_do_not(self):
        specs=module('node_specs')
        for diagnostic,expected in [(False,0),(True,1)]:
            with self.subTest(diagnostic=diagnostic),tempfile.TemporaryDirectory() as td:
                def child(*args,**kwargs):
                    (Path(td)/'bandwidth.json').write_text(json.dumps({'patterns':[{'correct':True}],
                        'diagnostic_only':diagnostic}))
                    return SimpleNamespace(returncode=0)
                with patch.object(specs,'system',return_value={'fixture':True}),patch.object(specs,'summary',return_value='fixture'),patch.object(specs.subprocess,'run',side_effect=child):
                    self.assertEqual(specs.main(['--output',td,'--python','fixture-python']),expected)


if __name__=='__main__':unittest.main()
