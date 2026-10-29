import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_splitzip_official_native as run


def native_result(snapshot):
    n = 268435456
    return dict(device='cuda:0', numel=n, chunk_size=1024, raw_bytes=n * 2,
        warmup=10, iters=50, repeats=5, n_escapes=0, compressed_bytes=n * 3 // 2,
        calibration=dict(calibration_source='model_forward', model=str(snapshot),
            dataset='wikitext', dataset_config='wikitext-2-raw-v1', split='train',
            max_prompts=4, max_values=4194304),
        benchmark_source=dict(source='model_forward', model=str(snapshot),
            dataset='wikitext', dataset_config='wikitext-2-raw-v1', split='test', max_prompts=4),
        encode_seconds=dict(mean=0.01), decode_seconds=dict(mean=0.02),
        encode_gbs=dict(mean=53.7), decode_gbs=dict(mean=26.8))


class NativeProtocolTests(unittest.TestCase):
    def test_valid_native_bytes_and_seven_byte_escapes(self):
        r = native_result('/snapshot')
        self.assertEqual(run.native_result_validation(r, '/snapshot')['native_bpv'], 12)
        r['n_escapes'] = 17
        r['compressed_bytes'] += 7 * 17
        self.assertTrue(run.native_result_validation(r, '/snapshot')['original_word_exact'])
        r['compressed_bytes'] -= 4 * 17  # Paper shorthand cannot replace actual released storage.
        with self.assertRaises(ValueError):
            run.native_result_validation(r, '/snapshot')

    def test_rejects_protocol_changes_and_invalid_timings(self):
        original = native_result('/snapshot')
        variants = [dict(warmup=3), dict(numel=123), dict(chunk_size=512), dict(n_escapes=-1)]
        for fields in variants:
            r = dict(original, **fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                run.native_result_validation(r, '/snapshot')
        for path, value in [('split', 'test'), ('max_values', 1000), ('model', '/other')]:
            r = copy.deepcopy(original); r['calibration'][path] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                run.native_result_validation(r, '/snapshot')
        r = copy.deepcopy(original); r['benchmark_source']['source'] = 'synthetic'
        with self.assertRaises(ValueError): run.native_result_validation(r, '/snapshot')
        for value in (float('nan'), float('inf'), 0, -1):
            r = copy.deepcopy(original); r['decode_seconds']['mean'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                run.native_result_validation(r, '/snapshot')

    def test_failed_validation_still_has_readable_failure_summary(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            run.summary(dict(status='failed', gpu_measured=False, error='bad native JSON',
                native={'malformed': True}), out)
            text = (out / 'summary.md').read_text()
            self.assertIn('failed', text)
            self.assertNotIn('| True |', text)

    def test_output_is_new_and_contained(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.assertEqual(run.output_path(root / 'runs/new', root), root / 'runs/new')
            with self.assertRaises(ValueError): run.output_path(root, root)
            with self.assertRaises(ValueError): run.output_path(root.parent / 'outside', root)
            (root / 'old').mkdir()
            with self.assertRaises(ValueError): run.output_path(root / 'old', root)
            (root / 'link').symlink_to(root.parent)
            with self.assertRaises(ValueError): run.output_path(root / 'link/outside', root)

    def test_author_source_tampering_and_shadow_modules_are_rejected(self):
        def git_result(source, *args):
            return run.PIN if args[0] == 'rev-parse' else ''
        with patch.object(run, 'git', side_effect=git_result), patch.object(run, 'sha', return_value='wrong'):
            with self.assertRaises(ValueError): run.source_validation(Path('/author'))
        def shadow(source, *args):
            if args[0] == 'rev-parse': return run.PIN
            if args[0] == 'ls-files': return 'torch.py'
            return ''
        with patch.object(run, 'git', side_effect=shadow), patch.object(run, 'sha', side_effect=lambda p: run.AUTHOR_HASHES[p.name]):
            with self.assertRaises(ValueError): run.source_validation(Path('/author'))

    def test_full_model_hash_detects_corrupt_cached_weight(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); snapshot = root / 'snapshots' / run.REVISION
            snapshot.mkdir(parents=True); blobs = root / 'blobs'; blobs.mkdir()
            original = b'original weight words'
            blob = blobs / hashlib.sha256(original).hexdigest(); blob.write_bytes(original)
            (snapshot / 'part.safetensors').symlink_to(blob)
            data = {'config.json':'{}', 'tokenizer.json':'{}', 'tokenizer_config.json':'{}',
                'model.safetensors.index.json':json.dumps({'weight_map':{'x':'part.safetensors'}})}
            for name, value in data.items(): (snapshot / name).write_text(value)
            entry = dict(weights=[dict(name='part.safetensors', bytes=len(original))],
                files={name:dict(sha256=run.sha(snapshot / name)) for name in data})
            receipt = run.model_receipt(snapshot, entry)
            self.assertTrue(receipt['files']['part.safetensors']['hf_blob_hash_verified'])
            blob.write_bytes(b'X' * len(original))
            with self.assertRaises(ValueError): run.model_receipt(snapshot, entry)


if __name__ == '__main__':
    unittest.main()
