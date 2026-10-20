"""CPU tensor validation plus explicitly conditional CUDA predictor parity."""
import copy
import unittest

import numpy as np

from qwen_kv_bench import transfer_reference as reference
from qwen_kv_bench.test_transfer_reference import capture
from qwen_kv_bench.transfer_gpu_predictor import TorchTransferPredictor

try:
    import torch
except ImportError:
    torch = None


def numpy_means(model, source, start=0, metadata=None):
    layers, tokens, width = source.shape
    positions = np.arange(start, start + tokens)
    previous = np.empty((tokens, 0), dtype=np.int64)
    output = np.empty(source.shape, dtype=np.float64)
    for layer in range(layers):
        fixed = reference._all_fixed(source[layer], positions, model)
        meta = metadata[layer].astype(np.int64) * 512 if model.metadata_rank else np.empty((tokens, 0), dtype=np.int64)
        features = np.concatenate((previous, meta, np.zeros((tokens, width), dtype=np.int64)), axis=1)
        offset = previous.shape[1] + model.metadata_rank
        for index, block in enumerate(model.predictors[layer]):
            ch0 = index * model.block_size
            output[layer, :, ch0:ch0 + model.block_size] = reference._predict(features, block, ch0, positions, model)
            features[:, offset + ch0:offset + ch0 + model.block_size] = fixed[:, ch0:ch0 + model.block_size]
        previous = features[:, offset:]
    return output


@unittest.skipUnless(torch is not None, "PyTorch is optional")
class TransferGpuPredictorTests(unittest.TestCase):
    def setUp(self):
        self.train = [capture(1), capture(2)]
        self.scale = capture(3)
        self.source = reference.words(capture(4, 7))

    def test_all_predictor_modes_cpu_tensor_parity(self):
        for mode in ("mean", "previous", "aqua", "block"):
            with self.subTest(mode=mode):
                model = reference.fit(self.train, self.scale, mode=mode, block_size=4, max_positions=128)
                replay = TorchTransferPredictor.from_reference(model, input_chunk=3, token_chunk=2)
                means = replay.prediction_sweep(torch.from_numpy(self.source), start=17)
                np.testing.assert_array_equal(means.numpy(), numpy_means(model, self.source, 17))
                scale = replay.inverse_scales(1, 0, 7).numpy()
                np.testing.assert_array_equal(scale, np.broadcast_to(model.predictors[1][0]["inv_s"], scale.shape))

    def test_paid_metadata_and_int32_storage_match(self):
        rng = np.random.default_rng(5)
        train_meta = [rng.integers(-127, 128, size=(3, 64, 2), dtype=np.int8) for _ in self.train]
        scale_meta = rng.integers(-127, 128, size=(3, 64, 2), dtype=np.int8)
        metadata = rng.integers(-127, 128, size=(3, 7, 2), dtype=np.int8)
        model = reference.fit(self.train, self.scale, block_size=4, metadata_train=train_meta, metadata_scale=scale_meta)
        original = TorchTransferPredictor.from_reference(model)
        compact = TorchTransferPredictor.from_reference(model, coefficient_dtype="int32", input_chunk=5, token_chunk=3)
        means = compact.prediction_sweep(torch.from_numpy(self.source), metadata=torch.from_numpy(metadata))
        np.testing.assert_array_equal(means.numpy(), numpy_means(model, self.source, metadata=metadata))
        self.assertEqual(compact.storage_statistics()["coefficient_tensor_bytes"] * 2,
                         original.storage_statistics()["coefficient_tensor_bytes"])

    def test_fixed_word_and_rope_boundary_parity(self):
        model = reference.fit(self.train, self.scale, block_size=4)
        replay = TorchTransferPredictor.from_reference(model)
        # Signed zeros, signed subnormals, and values around unit and fixed-point
        # half-even boundaries. Values remain within the reference's domain.
        source = np.array([[0, 0x8000, 1, 0x8001, 0x3f80, 0xbf80, 0x3f81, 0x3f7f,
                            0x3700, 0x3780, 0x37c0, 0xb700, 0xb780, 0xb7c0, 0x4000, 0xc000]], np.uint16)
        actual = replay.canonical_features(torch.from_numpy(source), replay.positions(13, 1))
        expected = reference._all_fixed(source, np.array([13]), model)
        np.testing.assert_array_equal(actual.numpy(), expected)
        source[0, 0] = 0x7f80
        with self.assertRaisesRegex(ValueError, "finite fixed-point"):
            replay.fixed_words(torch.from_numpy(source))

    def test_overflow_and_storage_cast_guards(self):
        model = reference.fit(self.train, self.scale, block_size=4)
        broken = copy.deepcopy(model)
        broken.predictors[0][1]["A"][0, 0] = 1 << 32
        with self.assertRaisesRegex(ValueError, "fit INT32"):
            TorchTransferPredictor.from_reference(broken, coefficient_dtype="int32")
        broken.predictors[0][1]["A"][0, 0] = 1 << 50
        replay = TorchTransferPredictor.from_reference(broken)
        features = torch.ones((1, 16), dtype=torch.int64) * (1 << 20)
        with self.assertRaisesRegex(ValueError, "accumulation would overflow"):
            replay.predict_block(features, 0, 1, replay.positions(0, 1))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is unavailable; CPU validation is not GPU evidence")
    def test_cuda_predictor_parity(self):
        model = reference.fit(self.train, self.scale, block_size=4, max_positions=128)
        replay = TorchTransferPredictor.from_reference(model, device="cuda", input_chunk=3, token_chunk=2)
        means = replay.prediction_sweep(torch.from_numpy(self.source).to("cuda"), start=17)
        np.testing.assert_array_equal(means.cpu().numpy(), numpy_means(model, self.source, 17))


if __name__ == "__main__":
    unittest.main()
