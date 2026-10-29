from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_exans_official_preflight as run


class PreflightTests(unittest.TestCase):
    def test_new_output_must_remain_inside_repository(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.assertEqual(run.output_path(root / 'runs/new', root), root / 'runs/new')
            with self.assertRaises(ValueError): run.output_path(root, root)
            with self.assertRaises(ValueError): run.output_path(root.parent / 'outside', root)
            (root / 'existing').mkdir()
            with self.assertRaises(ValueError): run.output_path(root / 'existing', root)
            (root / 'alias').symlink_to(root.parent)
            with self.assertRaises(ValueError): run.output_path(root / 'alias/outside', root)

    def test_runtime_links_accept_composed_toolkit_symlinks(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); build = root / 'build'; home = root / 'toolkit'; sdk = root / 'sdk'
            component = root / 'components/cudart/lib'
            for p in (build, home / 'lib64', sdk / 'lib64', component): p.mkdir(parents=True)
            (build / 'libopenlake_expans.so').write_bytes(b'built library')
            (component / 'libcudart.so.12').write_bytes(b'cached cuda runtime')
            (home / 'lib64/libcudart.so.12').symlink_to(component / 'libcudart.so.12')
            (sdk / 'lib64/libnvcomp.so.5').write_bytes(b'pinned SDK')
            text = '\n'.join([
                f'libopenlake_expans.so => {build}/libopenlake_expans.so (0x1)',
                f'libcudart.so.12 => {home}/lib64/libcudart.so.12 (0x2)',
                f'libnvcomp.so.5 => {sdk}/lib64/libnvcomp.so.5 (0x3)'])
            actual = run.validate_runtime_links(text, build, home, sdk)
            self.assertEqual(Path(actual['libcudart.so.12']), component / 'libcudart.so.12')
            # ldd may print the final resolved path instead of the toolkit alias.
            resolved = text.replace(str(home / 'lib64/libcudart.so.12'), str(component / 'libcudart.so.12'))
            self.assertEqual(run.validate_runtime_links(resolved, build, home, sdk), actual)
            wrong = root / 'other.so'; wrong.write_bytes(b'unrelated build')
            changed = text.replace(str(build / 'libopenlake_expans.so'), str(wrong))
            with self.assertRaises(ValueError): run.validate_runtime_links(changed, build, home, sdk)

    def test_ldd_missing_dependencies_are_rejected_even_without_nonzero_exit(self):
        with self.assertRaises(ValueError):
            run.validate_runtime_links('libcudart.so.12 => not found', Path('/b'), Path('/h'), Path('/s'))
        with self.assertRaises(ValueError):
            run.validate_runtime_links('linux-vdso.so.1 (0x1)', Path('/b'), Path('/h'), Path('/s'))

    def test_runtime_environment_binds_build_and_does_not_mutate_parent(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); build = root / 'build'; home = root / 'toolkit'; sdk = root / 'sdk'
            for p in (build, home / 'lib64', sdk / 'lib64'): p.mkdir(parents=True)
            original = dict(PATH='/usr/bin', LD_LIBRARY_PATH='/unrelated', TRITON_INTERPRET='1',
                PYTHONPATH='/other/python', LD_PRELOAD='/other/preload.so')
            env = run.runtime_environment(original, home, sdk, build)
            self.assertEqual(original['TRITON_INTERPRET'], '1')
            for key in ('TRITON_INTERPRET', 'PYTHONPATH', 'LD_PRELOAD'): self.assertNotIn(key, env)
            self.assertEqual(env['CUDA_HOME'], str(home))
            self.assertEqual(env['CUDACXX'], str(home / 'bin/nvcc'))
            self.assertEqual(env['LD_LIBRARY_PATH'].split(':')[0], str(build))
            self.assertNotIn('/unrelated', env['LD_LIBRARY_PATH'])

    def test_source_pin_and_modifications_stop_before_build(self):
        with patch.object(run, 'git', return_value='wrong pin'):
            with self.assertRaises(ValueError): run.source_receipt(Path('/source'))
        def dirty(source, *args):
            return run.PIN if args[0] == 'rev-parse' else ' M cuda/source.cu'
        with patch.object(run, 'git', side_effect=dirty):
            with self.assertRaises(ValueError): run.source_receipt(Path('/source'))

    def test_summary_never_promotes_correctness_to_performance(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            run.summary(dict(status='completed', gpu_correctness_test_passed=True), out)
            text = (out / 'summary.md').read_text()
            self.assertIn('GPU performance measured: **False**', text)
            self.assertIn('exited zero', text)
            self.assertNotIn('GB/s', text)
            run.summary(dict(status='failed', gpu_correctness_test_passed=False, error='missing toolkit'), out)
            text = (out / 'summary.md').read_text()
            self.assertIn('missing toolkit', text)
            self.assertNotIn('round-tripped bit-exactly', text)


if __name__ == '__main__':
    unittest.main()
