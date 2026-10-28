import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_nuc_baseline_readiness as inventory


class MemoryTests(unittest.TestCase):
    def test_host_memory_units_and_missing_fields(self):
        actual = inventory.parse_meminfo('MemTotal: 1000 kB\nMemAvailable: 200 kB\nSwapTotal: 30 kB\n')
        self.assertEqual(actual, dict(MemTotal=1024000, MemAvailable=204800, SwapTotal=30720, SwapFree=None))
        with self.assertRaises(ValueError): inventory.parse_meminfo('MemTotal: -2 kB')
        with self.assertRaises(ValueError): inventory.parse_meminfo('MemTotal: 2 MB')

    def test_linux_limit_sentinels_are_not_zero(self):
        for raw in (None, 'max', '-1', str(9223372036854771712)):
            self.assertIsNone(inventory.parse_limit(raw))
        self.assertEqual(inventory.parse_limit('0'), 0)
        self.assertEqual(inventory.parse_limit('123'), 123)
        with self.assertRaises(ValueError): inventory.parse_limit('-4')

    def test_cgroup_paths_and_v1_v2_mounts(self):
        self.assertEqual(inventory.cgroup_directory('/sys/fs/cgroup', '/', '/slice/job'), '/sys/fs/cgroup/slice/job')
        self.assertEqual(inventory.cgroup_directory('/sys/fs/cgroup', '/docker/x', '/'), '/sys/fs/cgroup')
        with self.assertRaises(ValueError): inventory.cgroup_directory('/sys/fs/cgroup', '/', '/../escape')
        v2 = inventory.cgroup_locations('0::/slice/job\n', '1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n')
        self.assertEqual(v2[0]['version'], 2)
        v1 = inventory.cgroup_locations('5:memory:/job\n', '1 0 0:1 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n')
        self.assertEqual(v1[0]['version'], 1)

    def test_ancestor_cgroup_limit_is_charged(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); proc = root / 'proc'; (proc / 'self').mkdir(parents=True)
            (proc / 'self/cgroup').write_text('0::/job')
            (proc / 'self/mountinfo').write_text('1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw')
            for directory, maximum, current in [('sys/fs/cgroup', '1000', '750'), ('sys/fs/cgroup/job', 'max', '100')]:
                p = root / directory; p.mkdir(parents=True, exist_ok=True)
                (p / 'memory.max').write_text(maximum); (p / 'memory.current').write_text(current)
            result = inventory.cgroup_inventory(proc, root)
            self.assertEqual(result['effective_limit_bytes'], 1000)
            self.assertEqual(result['effective_available_bytes'], 250)


class InventoryTests(unittest.TestCase):
    def test_missing_probe_is_reported(self):
        with patch.object(inventory.subprocess, 'run', side_effect=FileNotFoundError('not installed')):
            probe = inventory.command(['missing'])
        self.assertEqual(probe['status'], 'unavailable'); self.assertIsNone(probe['returncode'])

    def test_snapshot_index_cannot_escape_and_no_large_weight_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); (root / 'config.json').write_text('{}')
            (root / 'part.safetensors').write_bytes(b'weights')
            (root / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'x':'part.safetensors'})))
            result = inventory.snapshot_inventory(root)
            self.assertTrue(result['cached_files_present']); self.assertFalse(result['weight_contents_verified'])
            (root / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'x':'../escape'})))
            self.assertFalse(inventory.snapshot_inventory(root)['cached_files_present'])

    def test_output_requires_new_contained_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.assertEqual(inventory.new_output(root, root / 'runs/new'), root / 'runs/new')
            with self.assertRaises(ValueError): inventory.new_output(root, root)
            (root / 'existing').mkdir()
            with self.assertRaises(FileExistsError): inventory.new_output(root, root / 'existing')
            with self.assertRaises(ValueError): inventory.new_output(root, root.parent / 'outside')
            (root / 'alias').symlink_to(root.parent)
            with self.assertRaises(ValueError): inventory.new_output(root, root / 'alias/outside')

    def test_readiness_is_not_benchmark_or_memory_fit_guarantee(self):
        env = dict(metadata={'versions':{name:'installed' for name in inventory.ENV_PACKAGES}})
        args = dict(memory=dict(MemTotal=64 << 30, MemAvailable=40 << 30),
                    cgroups=dict(effective_limit_bytes=32 << 30, effective_available_bytes=30 << 30),
                    limits={'RLIMIT_AS':{'soft_bytes':None}}, gpu=dict(visible=True),
                    authors={name:dict(pinned_tracked_clean=True) for name in inventory.PINS},
                    envs={name:env for name in inventory.PINS}, model=dict(cached_files_present=True),
                    exans=dict(library_verified=True))
        r = inventory.readiness(**args)
        self.assertEqual(r['splitzip']['status'], 'eligible_for_verified_preflight')
        self.assertEqual(r['aqua']['status'], 'blocked')
        self.assertFalse(r['splitzip']['performance_measured']); self.assertFalse(r['automatic_aqua_retry'])
        self.assertEqual(r['aqua_activation_lower_bound_bytes'], 24 << 30)
        self.assertIn('does not guarantee', r['aqua_memory_scope'])
        args['cgroups']['effective_limit_bytes'] = 8 << 30
        self.assertTrue(any('below the 24 GiB' in s for s in inventory.readiness(**args)['aqua']['blockers']))
        args['model']['cached_files_present'] = False
        self.assertEqual(inventory.readiness(**args)['exans']['status'], 'eligible_for_verified_preflight')
        self.assertEqual(inventory.readiness(**args)['splitzip']['status'], 'blocked')


if __name__ == '__main__':
    unittest.main()
