"""CPU orchestration contracts; fixtures are not actual GPU validation."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pd_gpu_diagnostics', ROOT / 'scripts/pd_gpu_diagnostics.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def record(pair=(0, 1), route='host', mib=64):
    return {'gpu_executed':True, 'diagnostic_only':True, 'gpus':['fixture'] * 4,
            'torch':'2.11.0+cu128', 'cuda_runtime':'12.8', 'paths':[route], 'mib':mib,
            'host_legs':[{'gpu':i} for i in range(4)],
            'pairs':[{'src':pair[0], 'dst':pair[1], 'correct':True, route:{'correct':True}}]}


class GPUDiagnosticTests(unittest.TestCase):
    def test_pairs_are_allocated_distinct_and_directed(self):
        self.assertEqual(diag.parse_pairs('0-1,1-0'), [(0, 1), (1, 0)])
        for bad in ('', '0-0', '4-1', '0-1,0-1', '-1-2', '0-1;false'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                diag.parse_pairs(bad)

    def test_zero_exit_needs_complete_exact_gpu_proof(self):
        diag.validate_case(record(), (0, 1), 'host', 64)
        changes = [{'gpu_executed':False}, {'diagnostic_only':False}, {'gpus':['fixture']},
                   {'torch':'2.13.0+cu130'}, {'cuda_runtime':'13.0'}, {'paths':['direct']},
                   {'mib':896}, {'host_legs':[]}, {'pairs':[]},
                   {'pairs':[{'src':0, 'dst':1, 'correct':True, 'host':{'correct':False}}]}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                diag.validate_case(dict(record(), **change), (0, 1), 'host', 64)

    def test_children_use_fresh_process_groups_and_keep_failure_logs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            command = [sys.executable, '-c', 'import os; print(os.getpid(),os.getpgrp(),flush=True); raise SystemExit(7)']
            ids = []
            for i in range(2):
                log = root / f'{i}.log'
                code, timeout = diag.run_child(command, log, dict(os.environ), 10)
                self.assertEqual((code, timeout), (7, False))
                pid, group = map(int, log.read_text().split())
                self.assertEqual(pid, group)
                ids.append(pid)
            self.assertNotEqual(*ids)

    def test_child_timeout_returns_and_keeps_trace(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / 'timeout.log'
            command = [sys.executable, '-c', 'import time; print("fixture waiting",flush=True); time.sleep(30)']
            code, timeout = diag.run_child(command, log, dict(os.environ), .1)
            self.assertTrue(timeout)
            self.assertNotEqual(code, 0)
            self.assertIn('fixture waiting', log.read_text())

    def test_diagnostic_report_retains_failures_and_never_validates_performance(self):
        for bad_route in (None, 'direct'):
            with self.subTest(bad_route=bad_route), tempfile.TemporaryDirectory() as td:
                output = Path(td)
                calls = []
                def child(command, logfile, env, timeout):
                    route = command[command.index('--paths') + 1]
                    pair = tuple(map(int, command[command.index('--pairs') + 1].split('-')))
                    calls.append((route, env['CUDA_LAUNCH_BLOCKING']))
                    logfile.write_text('CPU fixture only\n')
                    if route == bad_route:
                        return -6, False
                    Path(command[-1]).write_text(json.dumps(record(pair, route)))
                    return 0, False
                with patch.dict(os.environ, {'SLURM_JOB_ID':'cpu-fixture'}), patch.object(diag, 'run_child', child):
                    rc = diag.main(['--pin', 'a'*40, '--output', str(output), '--pairs', '0-1'])
                result = json.loads((output / 'diagnostics.json').read_text())
                self.assertEqual(rc, 0 if bad_route is None else 1)
                self.assertEqual(calls, [('host', '1'), ('host_pipe', '1'), ('direct', '1')])
                self.assertFalse(result['performance_validated'])
                self.assertFalse(result['serving_validated'])
                self.assertFalse(result['automatic_merge'])
                self.assertEqual(result['all_selected_copies_passed'], bad_route is None)
                self.assertTrue((output / 'summary.md').is_file())
                self.assertTrue((output / 'share.zip').is_file())

    def test_launcher_is_syntactically_valid_and_requests_only_diagnostic_resources(self):
        path = ROOT / 'scripts/run_t_pd_diagnostics.sh'
        subprocess.run(['bash', '-n', str(path)], check=True)
        text = path.read_text()
        self.assertIn('--gpus=4 -c 4 --mem=16G -t 00:45:00', text)
        self.assertIn('/tmp/$USER/kvpd-t-matched-$USER', text)
        self.assertIn('git -C "$T_PD_DIAG_REPO" archive "$T_PD_DIAG_PIN"', text)
        self.assertIn("'torch==2.11.0+cu128'", text)
        self.assertNotIn('pd_setup.sh', text)
        self.assertNotIn('rm -rf', text)

    def test_login_launcher_archives_its_pin_and_reuses_only_own_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo, node, binaries = root / 'repo', root / 'node', root / 'bin'
            (repo / 'scripts').mkdir(parents=True)
            binaries.mkdir()
            for name in ('run_t_pd_diagnostics.sh', 'pd_gpu_diagnostics.py', 'pd_env.sh'):
                shutil.copyfile(ROOT / 'scripts' / name, repo / 'scripts' / name)
            probe = '''import argparse,json
p=argparse.ArgumentParser()
for k in ('mib','pairs','paths','reps','output'):p.add_argument('--'+k)
a=p.parse_args();i,j=map(int,a.pairs.split('-'))
d={'gpu_executed':True,'diagnostic_only':True,'gpus':['CPU fixture']*4,
'torch':'2.11.0+cu128','cuda_runtime':'12.8','paths':[a.paths],'mib':int(a.mib),
'host_legs':[{'gpu':i} for i in range(4)],'pairs':[{'src':i,'dst':j,'correct':True,a.paths:{'correct':True}}]}
open(a.output,'w').write(json.dumps(d))
print('CPU fixture only; no GPU execution')
'''
            (repo / 'scripts/pd_link_probe.py').write_text(probe)
            def git(*args):
                return subprocess.run(['git', '-C', str(repo), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            branch = 'experiment/t-pd-matched-campaign-v1'
            git('init', '-q', '-b', branch)
            git('config', 'user.name', 'fixture')
            git('config', 'user.email', 'fixture@example.invalid')
            git('add', '.')
            git('commit', '-qm', 'CPU launcher fixture')
            origin = root / 'origin.git'
            subprocess.run(['git', 'clone', '-q', '--bare', str(repo), str(origin)], check=True)
            git('remote', 'add', 'origin', str(origin))
            uv = node / 'kvpd-cache/bin/uv'
            uv.parent.mkdir(parents=True)
            uv.write_text('#!' + sys.executable + '''
import os,sys
from pathlib import Path
if sys.argv[1]=='venv':
 d=Path(sys.argv[-1])/'bin';d.mkdir(parents=True);os.symlink(sys.executable,d/'python')
elif 'freeze' in sys.argv:print('CPU fixture packages')
''')
            uv.chmod(0o755)
            (binaries / 'srun').write_text('#!/bin/bash\nexport SLURM_JOB_ID=cpu-fixture\nexec "${@: -2:1}" "${@: -1}"\n')
            (binaries / 'nvidia-smi').write_text('#!/bin/sh\necho CPU-fixture-no-GPU\n')
            for file in binaries.iterdir():
                file.chmod(0o755)
            env = {k:v for k,v in os.environ.items() if not k.startswith(('SLURM_', 'T_PD_', 'KVPD_'))}
            env.update(PATH=str(binaries) + ':' + os.environ['PATH'], T_PD_ROOT=str(node),
                       T_PD_NODE=socket.gethostname().split('.')[0], T_PD_STATE_HOME=str(root / 'state'),
                       T_PD_DIAG_PAIRS='0-1', KVPD_ROOT='/tmp/do-not-use-other-project')
            result = subprocess.run(['bash', str(repo / 'scripts/run_t_pd_diagnostics.sh')],
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report_path = next((root / 'state/diagnostics').glob('*/diagnostics.json'))
            report = json.loads(report_path.read_text())
            self.assertEqual(report['pin'], git('rev-parse', 'HEAD'))
            self.assertEqual(len(report['cases']), 3)
            self.assertFalse(report['serving_validated'])
            self.assertTrue(next((node / 'kvpd-diagnostics').glob('*/code/scripts/pd_link_probe.py')).is_file())
            self.assertEqual(git('status', '--porcelain'), '')


if __name__ == '__main__':
    unittest.main()
