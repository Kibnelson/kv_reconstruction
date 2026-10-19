"""Study gates and independently restored frozen control packets."""
import copy
import unittest

import numpy as np

from qwen_kv_bench import neural_study as study


class NeuralStudyTests(unittest.TestCase):
    def test_continuation_gate_requires_all_finite_bitwise_evidence(self):
        good = dict(all_logits_bitwise_equal=True, all_tokens_equal=True,
                    all_appended_KV_bitwise_equal=True, per_step=[dict(logits_finite=True)])
        self.assertTrue(study._generation_passed(good))
        for key in ("all_logits_bitwise_equal", "all_tokens_equal", "all_appended_KV_bitwise_equal"):
            bad = copy.deepcopy(good)
            bad[key] = False
            self.assertFalse(study._generation_passed(bad))
            del bad[key]
            self.assertFalse(study._generation_passed(bad))
        for step in ({}, {"logits_finite": False}):
            self.assertFalse(study._generation_passed(dict(good, per_step=[step])))
        self.assertFalse(study._generation_passed({}))
        self.assertFalse(study._generation_passed(dict(good, per_step=[])))

    def test_frozen_control_real_container_and_train_only(self):
        # Include signed zero, subnormal, NaN payloads and infinities: anchor
        # coding is bit exact even though neural conditioning needs finite KV.
        words = np.array([[0, 0x8000], [1, 0x7fc1], [0x7f80, 0xff80]], np.uint16)
        captures = [[(words, words[:, ::-1].copy()), (words ^ np.uint16(0x123), words.copy())]]
        state = study.fit_anchor(captures, [dict(split="train", name="fixture")], "test", "cpu")
        result = study._baseline(captures[0], state, [0, 2], 2, 2)
        self.assertTrue(result["container_roundtrip_verified"])
        self.assertEqual(result["raw_sample_bytes"], 2 * 2 * words.nbytes)
        self.assertEqual(result["records"], 4)
        self.assertEqual(result["packet_bytes"], result["inner_packet_bytes"] + result["indexed_envelope_bytes"])
        with self.assertRaises(ValueError):
            study.fit_anchor(captures, [dict(split="test")], "test", "cpu")


if __name__ == "__main__":
    unittest.main()
