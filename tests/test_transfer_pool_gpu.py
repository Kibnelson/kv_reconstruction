"""CPU tensor correctness/parity tests, plus a real CUDA check when available."""
from pathlib import Path
import unittest

import numpy as np
import torch

from qwen_kv_bench import transfer_pool_gpu as pool
from scripts import bench_team_pool_reference as reference


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/team_transfer_original/combined.py"


class TensorPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_original_parity(self, words, refs, widths, device="cpu", tile_tokens=16):
        state = pool.make_state(refs, widths, device=device)
        source_words = torch.from_numpy(np.ascontiguousarray(words)).to(device)
        packet = pool.encode_pool(source_words, state, tile_tokens=tile_tokens, tiles_per_batch=1)
        decoded = pool.decode_pool(packet, state, tiles_per_batch=2)
        self.assertEqual(decoded.device, source_words.device)
        self.assertTrue(torch.equal(decoded, source_words))
        original = reference.load_reference(SOURCE, [], words.shape[2], block_tokens=tile_tokens)
        records = []
        for layer in range(words.shape[0]):
            for start in range(0, words.shape[1], tile_tokens):
                records.append(original["pack_block"](words[layer, start:start + tile_tokens], refs[layer], widths[layer]))
        expected = np.concatenate(records)
        np.testing.assert_array_equal(packet.payload.cpu().numpy(), expected)
        np.testing.assert_array_equal(packet.offsets.cpu().numpy(), np.r_[0, np.cumsum([len(record) for record in records])])
        stats = pool.packet_stats(packet, state, include_offsets=True)
        self.assertEqual(stats["payload_bytes"], len(expected))
        self.assertEqual(stats["physical_state_tensor_bytes"], 2 * refs.size)
        self.assertEqual(stats["block_index_bytes"], 8 * (len(records) + 1))
        self.assertEqual(stats["physical_packet_tensor_bytes"], len(expected) + stats["block_index_bytes"])
        self.assertEqual(stats["escape_header_bytes"], 2 * len(records))
        return packet

    def test_all_bf16_patterns_and_original_wire_parity(self):
        words = np.arange(65536, dtype=np.uint16).reshape(2, 32, 1024)
        refs = np.full((2, 1024), 127, np.int64)
        widths = np.tile(np.arange(1, 9), (2, 128))
        packet = self.assert_original_parity(words, refs, widths)
        self.assertGreater(sum(packet.escape_counts.tolist()), 0)

    def test_zero_escapes_and_dense_padding(self):
        # One-token records exercise bit padding beyond original 16-token use.
        words = np.full((1, 3, 3), np.uint16(127 << 7))
        refs = np.full((1, 3), 127, np.int64)
        widths = np.array([[1, 2, 3]])
        packet = self.assert_original_parity(words, refs, widths, tile_tokens=1)
        self.assertEqual(sum(packet.escape_counts.tolist()), 0)

    def test_refuses_wrong_geometry_or_device(self):
        state = pool.make_state([[127] * 16], [[1] * 16])
        with self.assertRaisesRegex(ValueError, "uint16"):
            pool.encode_pool(torch.zeros((1, 16, 16), dtype=torch.float32), state)
        with self.assertRaisesRegex(ValueError, "multiple"):
            pool.encode_pool(torch.zeros((1, 17, 16), dtype=torch.uint16), state)
        with self.assertRaisesRegex(ValueError, "widths"):
            pool.make_state([[127]], [[9]])

    def test_corrupt_escape_header_is_rejected(self):
        state = pool.make_state([[127] * 16], [[1] * 16])
        words = torch.full((1, 16, 16), 127 << 7, dtype=torch.uint16)
        packet = pool.encode_pool(words, state)
        payload = packet.payload.clone()
        payload[0] = 1
        damaged = pool.PoolPacket(payload, packet.offsets, packet.shape, packet.tile_tokens)
        with self.assertRaisesRegex(ValueError, "record length"):
            pool.decode_pool(damaged, state)

    def test_packet_copy_carries_directory_and_payload(self):
        state = pool.make_state([[127] * 16], [[1] * 16])
        words = torch.full((1, 16, 16), 127 << 7, dtype=torch.uint16)
        packet = pool.encode_pool(words, state)
        moved = pool.packet_to_device(packet, "cpu")
        self.assertEqual(set(moved.tensor_fields), {"payload", "offsets"})
        self.assertTrue(torch.equal(moved.payload, packet.payload))
        self.assertTrue(torch.equal(moved.offsets, packet.offsets))
        self.assertTrue(torch.equal(pool.decode_pool(moved, state), words))

    def test_malformed_layout_rejected_before_payload_gathers(self):
        state = pool.make_state([[127] * 16], [[1] * 16])
        words = torch.full((1, 16, 16), 127 << 7, dtype=torch.uint16)
        packet = pool.encode_pool(words, state)
        short = pool.PoolPacket(torch.zeros(1, dtype=torch.uint8), torch.tensor([0, 1]), packet.shape, 16)
        with self.assertRaisesRegex(ValueError, "minimum"):
            pool.decode_pool(short, state)
        negative = pool.PoolPacket(packet.payload, torch.tensor([-1, packet.payload.numel()]), packet.shape, 16)
        with self.assertRaisesRegex(ValueError, "endpoints"):
            pool.decode_pool(negative, state)
        for shape, tile_tokens in (((1, 0, 16), 16), ((1, 16, 16), 0), ((1, 16, 16), -1)):
            malformed = pool.PoolPacket(packet.payload, packet.offsets, shape, tile_tokens)
            with self.assertRaises(ValueError):
                pool.decode_pool(malformed, state)
        stale_state = pool.PoolState(state.references, state.widths, (17,))
        with self.assertRaisesRegex(ValueError, "stale"):
            pool.decode_pool(packet, stale_state)

    @unittest.skipUnless((ROOT / "runs/transfer_smoke_cpu/captures/train_train_railway_512.npz").exists(),
                         "fresh real-model integration captures are not present")
    def test_real_capture_pages_all_layers_match_original_bytes(self):
        capture_dir = ROOT / "runs/transfer_smoke_cpu/captures"
        fit = [reference.load_capture(capture_dir / (name + ".npz")) for name in reference.TRAIN_NAMES]
        layers, _, channels = fit[0]["K_bits"].shape
        original = reference.load_reference(SOURCE, fit, 2 * channels)
        fitted = [original["fit_pool"](layer) for layer in range(layers)]
        refs = np.stack([item[0] for item in fitted])
        widths = np.stack([item[1] for item in fitted])
        measured = 0
        word_count = 0
        for name in reference.TEST_NAMES:
            capture = reference.load_capture(capture_dir / (name + ".npz"))
            for start in (0, 480):
                words = np.stack([original["words"](capture, layer)[start:start + 32] for layer in range(layers)])
                packet = self.assert_original_parity(words, refs, widths)
                measured += packet.payload_nbytes
                word_count += words.size
        self.assertEqual(word_count, 1_474_560)
        self.assertEqual(measured, 2_114_562)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable; CPU tests are not GPU measurements")
    def test_actual_cuda_roundtrip_matches_original_bytes(self):
        rng = np.random.default_rng(119)
        words = rng.integers(0, 65536, (2, 32, 32), dtype=np.uint16)
        refs = np.full((2, 32), 127, np.int64)
        widths = np.tile(np.arange(1, 9), (2, 4))
        packet = self.assert_original_parity(words, refs, widths, device="cuda")
        self.assertEqual(packet.payload.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
