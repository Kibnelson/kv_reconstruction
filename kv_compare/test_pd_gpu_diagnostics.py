"""CPU orchestration contracts; fixtures are not actual GPU validation."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
import zipfile
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

    def test_login_launcher_bootstraps_fresh_node_and_preserves_failure_evidence(self):
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
            # No node root or package cache exists. Only the account's uv exists.
            uv = root / 'home/.local/bin/uv'
            uv.parent.mkdir(parents=True)
            uv_content = '#!' + sys.executable + '''
import os,sys
from pathlib import Path
if sys.argv[1]=='--version':print('uv CPU-fixture')
elif sys.argv[1]=='venv':
 if os.environ.get('FIXTURE_FAIL_STAGE')=='create_environment':raise SystemExit(9)
 d=Path(sys.argv[-1])/'bin';d.mkdir(parents=True);os.symlink(sys.executable,d/'python')
elif 'install' in sys.argv and os.environ.get('FIXTURE_FAIL_STAGE')=='install_packages':raise SystemExit(9)
elif 'freeze' in sys.argv:print('CPU fixture packages')
'''
            uv.write_text(uv_content)
            uv.chmod(0o755)
            (binaries / 'srun').write_text('''#!/bin/bash
if [ "${FIXTURE_REMOVE_UV_SOURCE:-0}" = 1 ]; then rm -f -- "$T_PD_DIAG_UV_SOURCE"; fi
if [ "${FIXTURE_MUTATE_UV_SNAPSHOT:-0}" = 1 ]; then printf '\\n# changed\\n' >> "$T_PD_DIAG_UV_SNAPSHOT"; fi
export SLURM_JOB_ID=cpu-fixture
exec "${@: -2:1}" "${@: -1}"
''')
            (binaries / 'nvidia-smi').write_text('#!/bin/sh\necho CPU-fixture-no-GPU\n')
            for file in binaries.iterdir():
                file.chmod(0o755)
            env = {k:v for k,v in os.environ.items() if not k.startswith(('SLURM_', 'T_PD_', 'KVPD_'))}
            env.update(PATH=str(binaries) + ':/usr/bin:/bin', HOME=str(root/'home'),
                       T_PD_UV=str(uv), T_PD_ROOT=str(node),
                       T_PD_NODE=socket.gethostname().split('.')[0], T_PD_STATE_HOME=str(root / 'state'),
                       T_PD_DIAG_PAIRS='0-1', KVPD_ROOT='/tmp/do-not-use-other-project')
            self.assertFalse(node.exists())
            result = subprocess.run(['bash', str(repo / 'scripts/run_t_pd_diagnostics.sh')],
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report_path = next((root / 'state/diagnostics').glob('*/diagnostics.json'))
            report = json.loads(report_path.read_text())
            self.assertEqual(report['pin'], git('rev-parse', 'HEAD'))
            self.assertEqual(len(report['cases']), 3)
            self.assertFalse(report['serving_validated'])
            self.assertTrue(next((node / 'kvpd-diagnostics').glob('*/code/scripts/pd_link_probe.py')).is_file())
            self.assertIn('DIAG_JOB=cpu-fixture', result.stdout)
            self.assertEqual((report_path.parent/'node_status.txt').read_text().strip(), 'stage=complete exit=0 job=cpu-fixture')
            runtime_uv = next((node/'kvpd-diagnostics').glob('*/bin/uv'))
            self.assertEqual(runtime_uv.read_bytes(), uv.read_bytes())
            self.assertFalse((node/'kvpd-cache/bin/uv').exists())
            self.assertTrue((node/'kvpd-cache/uv').is_dir())
            self.assertTrue((node/'kvpd-cache/python').is_dir())
            digest = hashlib.sha256(uv.read_bytes()).hexdigest()
            self.assertIn(f'sha256={digest}', (report_path.parent/'uv_provenance.txt').read_text())
            self.assertEqual((report_path.parent/'uv_version.txt').read_text(), 'uv CPU-fixture\n')
            self.assertFalse(list((root/'state/diagnostics').glob('uv-diagnostic.*')))
            with zipfile.ZipFile(report_path.parent/'share.zip') as bundle:
                self.assertIn('uv_provenance.txt', bundle.namelist())
                self.assertFalse(any('uv-diagnostic.' in name or name.endswith('/bin/uv') for name in bundle.namelist()))
            # PATH takes precedence over the home fallback when no override is set.
            fallback_env = {k:v for k,v in env.items() if k != 'T_PD_UV'}
            for use_path in (True, False):
                with self.subTest(use_path=use_path):
                    if use_path:
                        shutil.copyfile(uv, binaries/'uv')
                        (binaries/'uv').chmod(0o755)
                    else:
                        (binaries/'uv').unlink()
                    selected = binaries/'uv' if use_path else uv
                    launched = subprocess.run(['bash', str(repo/'scripts/run_t_pd_diagnostics.sh')],
                                              env=fallback_env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(launched.returncode, 0, launched.stdout+launched.stderr)
                    self.assertIn(f'DIAG_UV_SOURCE={selected}', launched.stdout)
            # The allocation uses the snapshot even if a symlink target vanishes.
            link = root/'uv-link'
            link.symlink_to(uv)
            launched = subprocess.run(['bash', str(repo/'scripts/run_t_pd_diagnostics.sh')],
                                      env=dict(env, T_PD_UV=str(link), FIXTURE_REMOVE_UV_SOURCE='1'),
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(launched.returncode, 0, launched.stdout+launched.stderr)
            self.assertFalse(uv.exists())
            uv.write_text(uv_content)
            uv.chmod(0o755)
            failed = subprocess.run(['bash', str(repo/'scripts/run_t_pd_diagnostics.sh')],
                                    env=dict(env, FIXTURE_MUTATE_UV_SNAPSHOT='1'),
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(failed.returncode, 1, failed.stdout+failed.stderr)
            self.assertIn('DIAG_NODE_STOP stage=prepare_uv', failed.stdout+failed.stderr)
            self.assertIn('does not match the login snapshot SHA-256', failed.stdout+failed.stderr)
            self.assertNotIn('DIAG_BEGIN pair=', failed.stdout)
            # Quiet installer exits used to produce only srun's generic exit.
            for stage in ('create_environment', 'install_packages'):
                with self.subTest(stage=stage):
                    failed = subprocess.run(['bash', str(repo / 'scripts/run_t_pd_diagnostics.sh')],
                                            env=dict(env, FIXTURE_FAIL_STAGE=stage), capture_output=True,
                                            text=True, timeout=30)
                    self.assertEqual(failed.returncode, 9, failed.stdout+failed.stderr)
                    self.assertIn(f'DIAG_NODE_STOP stage={stage}', failed.stdout+failed.stderr)
                    self.assertIn('DIAG_JOB=cpu-fixture', failed.stdout)
                    self.assertNotIn('DIAG_BEGIN pair=', failed.stdout)
                    output = Path(next(line.split('=', 1)[1] for line in failed.stdout.splitlines() if line.startswith('OUTPUT=')))
                    self.assertEqual((output/'node_status.txt').read_text().strip(), f'stage={stage} exit=9 job=cpu-fixture')
            uv.chmod(0o644)
            failed = subprocess.run(['bash', str(repo / 'scripts/run_t_pd_diagnostics.sh')],
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(failed.returncode, 1, failed.stdout+failed.stderr)
            self.assertIn('DIAG_LOGIN_STOP stage=uv_preflight', failed.stdout+failed.stderr)
            self.assertIn(f'uv source is not a readable executable: {uv}', failed.stdout+failed.stderr)
            self.assertNotIn('DIAG_JOB=', failed.stdout)
            self.assertNotIn('DIAG_BEGIN pair=', failed.stdout)
            uv.chmod(0o755)
            failed = subprocess.run(['bash', str(repo/'scripts/run_t_pd_diagnostics.sh')],
                                    env=dict(fallback_env, HOME=str(root/'missing-home')),
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(failed.returncode, 1, failed.stdout+failed.stderr)
            self.assertIn('install uv in your own account or set T_PD_UV', failed.stdout+failed.stderr)
            self.assertNotIn('DIAG_JOB=', failed.stdout)
            node.chmod(0o777)
            failed = subprocess.run(['bash', str(repo / 'scripts/run_t_pd_diagnostics.sh')],
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(failed.returncode, 2, failed.stdout+failed.stderr)
            self.assertIn('DIAG_NODE_STOP stage=private_root', failed.stdout+failed.stderr)
            self.assertIn('mode=777', failed.stdout+failed.stderr)
            node.chmod(0o700)
            self.assertFalse(list((root/'state/diagnostics').glob('uv-diagnostic.*')))
            self.assertEqual(git('status', '--porcelain'), '')


if __name__ == '__main__':
    unittest.main()
