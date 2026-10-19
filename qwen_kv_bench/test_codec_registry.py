"""Synthetic adapter tests; no checkpoint download, GPU, or old fitted state."""
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from qwen_kv_bench import codec_registry as registry
from kv_research import learned_codec
from kv_coset_fields import learned_lower


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(701)
        training = {'layer_00_K': rng.integers(14900, 16300, (40, 64), dtype=np.uint16)}
        models = {f'learned{size}': learned_codec.train_model(training,
            {'n_centroids': size, 'iterations': 1, 'max_samples_per_key': 320}) for size in (16, 64, 256)}
        book = np.tile(np.linspace(14500, 16500, 64, dtype=np.uint16)[:, None], (1, 8))
        models['fixed64'] = learned_codec.LearnedModel({'layer_00_K': book}, {'n_centroids': 64}, {})
        for conditional in (False, True):
            name = 'lower16_conditional' if conditional else 'lower16'
            models[name] = learned_lower.train_model(training,
                {'n_centroids': 16, 'conditional': conditional, 'iterations': 1, 'max_samples_per_key': 128})
        for model in models.values():
            model.config.update(calibration_namespace='qwen_kv_bench', source_model_identity='synthetic-target')
        cls.artifacts = SimpleNamespace(models=models, paths={},
            metadata={'model_identity': 'synthetic-target', 'calibration_namespace': 'qwen_kv_bench'})

    def test_presets_and_requirements(self):
        self.assertEqual(len(registry.candidate_specs('smoke')), 5)
        main = registry.required_artifacts('main')
        self.assertEqual(main['magnitude_sizes'], [64])
        self.assertEqual(main['fixed_sizes'], [64])
        all_options = registry.required_artifacts('all')
        self.assertEqual(all_options['magnitude_sizes'], [16, 64, 256])
        self.assertEqual(all_options['conditional_lower_sizes'], [16])
        self.assertGreater(len(registry.candidate_specs('all', True)), len(registry.candidate_specs('all')))

    def test_all_concrete_adapters_and_parameter_grid(self):
        rng = np.random.default_rng(43)
        finite = rng.integers(15000, 16100, (13, 16), dtype=np.uint16)
        finite |= rng.integers(0, 2, finite.shape, dtype=np.uint16) << 15
        edge = np.array([[0, 0x8000, 1, 0x7f80, 0xffff, 0xff80, 0x7fc0, 0x7f7f, 17]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as folder:
            codecs = registry.build_registry('all', self.artifacts, folder, grid=True,
                expected_model_identity='synthetic-target')
            for name, adapter in codecs.items():
                for words in (finite, edge):
                    encoded = adapter.encode(words, 0, 'K')
                    np.testing.assert_array_equal(adapter.decode(encoded, 0, 'K'), words, err_msg=name)
                    self.assertEqual(sum(encoded.stats['component_bytes'].values()), len(encoded.data), name)
                    self.assertEqual(encoded.stats['registry_mechanism_counts']['encoded_records'], 1)
                self.assertEqual(adapter.traits['execution'], 'cpu_reference')
                self.assertFalse(adapter.traits['gpu_native'])
                if adapter.parent is not None:
                    inner = adapter.parent.encode(finite, 0, 'K')
                    wrapped = adapter.encode_from_parent(inner)
                    np.testing.assert_array_equal(adapter.decode(wrapped, 0, 'K'), finite)
                    self.assertTrue(wrapped.stats['wrapper_inner_byte_identity'])
                    self.assertEqual(adapter.state_bytes, adapter.parent.state_bytes)
                if adapter.traits['family'] == 'binary':
                    self.assertEqual(adapter.state_bytes,
                        len(learned_codec.model_bytes(self.artifacts.models['learned64'])) + adapter.state_info['transform_bytes'])

    def test_target_state_is_required_and_identity_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                registry.build_registry('smoke', None, folder)
            with self.assertRaises(ValueError):
                registry.build_registry('smoke', self.artifacts, folder, expected_model_identity='different-target')
            codecs = registry.build_registry(workspace=folder, names=['raw', 'exponent_huffman'])
            self.assertEqual(set(codecs), {'raw', 'exponent_huffman'})

    def test_real_calibration_artifacts_interoperate(self):
        from qwen_kv_bench.calibration import train_calibration
        rng = np.random.default_rng(219)
        training = [{f'layer_00_{kind}': rng.integers(14900, 16300, (16, 64), dtype=np.uint16)
            for kind in ('K', 'V')} for _ in range(3)]
        identity = {'model_id': 'synthetic-target', 'revision': 'unit'}
        with tempfile.TemporaryDirectory() as folder:
            artifacts = train_calibration(training, folder + '/calibration',
                registry.required_artifacts('all'), max_samples_per_key=320,
                iterations=1, model_identity=identity)
            codecs = registry.build_registry('all', artifacts, folder, expected_model_identity=identity)
            for name in ('learned256', 'fixed64', 'lower16_conditional_forced', 'hybrid_tile', 'binary_dense_d64_bf16'):
                words = training[0]['layer_00_K'][:3]
                encoded = codecs[name].encode(words, 0, 'K')
                np.testing.assert_array_equal(codecs[name].decode(encoded, 0, 'K'), words)


if __name__ == '__main__':
    unittest.main()
