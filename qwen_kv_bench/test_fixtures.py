"""Meaningful chat-boundary and exact-budget tests, without a model download."""

import copy
import unittest

from qwen_kv_bench.fixtures import DEV, SPLITS, TEST, TRAIN, build_input_ids, build_messages


class FakeTokenizer:
    """Character tokenizer with realistic atomic chat markers and offsets."""
    markers = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")

    def __init__(self):
        self.template_calls = []

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        ids, offsets = [], []
        i = 0
        while i < len(text):
            marker = next((m for m in self.markers if text.startswith(m, i)), None)
            end = i + len(marker) if marker else i + 1
            ids.append(100000 + self.markers.index(marker) if marker else ord(text[i]))
            offsets.append((i, end))
            i = end
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids):
        return "".join(self.markers[x - 100000] if x >= 100000 else chr(x) for x in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking=None):
        self.template_calls.append((tokenize, add_generation_prompt, enable_thinking))
        text = "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if enable_thinking is False:
                text += "<think>\n\n</think>\n\n"
        return self(text)["input_ids"] if tokenize else text


class OldTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)


class BoundaryMergeTokenizer(FakeTokenizer):
    """A token may cross the user-prefix/body or body/suffix boundary."""
    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        original = super().__call__(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
        ids, offsets = [], []
        i = 0
        while i < len(original["input_ids"]):
            if i + 2 < len(original["input_ids"]):
                start = original["offset_mapping"][i][0]
                end = original["offset_mapping"][i + 2][1]
                if text[start:end] in ("\n\nA", "Z\n\n"):
                    ids.append(200000 + (0 if text[start:end] == "\n\nA" else 1))
                    offsets.append((start, end))
                    i += 3
                    continue
            ids.append(original["input_ids"][i])
            offsets.append(original["offset_mapping"][i])
            i += 1
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids):
        return "".join(("\n\nA", "Z\n\n")[x - 200000] if x >= 200000 else super(BoundaryMergeTokenizer, self).decode([x]) for x in ids)


class FixtureTests(unittest.TestCase):
    def test_exact_lengths_and_all_fixed_content(self):
        tokenizer = FakeTokenizer()
        before = copy.deepcopy(SPLITS)
        for fixtures in SPLITS.values():
            for fixture in fixtures:
                for length in (512, 2048):
                    ids = build_input_ids(tokenizer, fixture, length)
                    self.assertEqual(len(ids), length)
                    text = tokenizer.decode(ids)
                    self.assertIn(fixture["prefix"] + "\n\n", text)
                    self.assertIn("\n\n" + fixture["suffix"], text)
                    if fixture.get("needle"):
                        self.assertIn(fixture["needle"], text)
                    self.assertTrue(text.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
                    self.assertEqual(text.count("<|im_start|>"), 3)
        self.assertEqual(SPLITS, before)
        self.assertTrue(all(thinking is False and generation is True for _, generation, thinking in tokenizer.template_calls))

    def test_deterministic_disjoint_splits(self):
        names = [f["name"] for fixtures in SPLITS.values() for f in fixtures]
        bodies = [f["body"] for fixtures in SPLITS.values() for f in fixtures]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(bodies), len(set(bodies)))
        self.assertEqual([len(TRAIN), len(DEV), len(TEST)], [3, 1, 3])
        self.assertEqual({f["kind"] for f in TRAIN}, {"text", "code", "structured"})
        tokenizer = FakeTokenizer()
        for split, fixtures in SPLITS.items():
            for fixture in fixtures:
                self.assertEqual(fixture["split"], split)
                self.assertEqual(build_input_ids(tokenizer, fixture, 1024), build_input_ids(tokenizer, fixture, 1024))

    def test_too_small_and_invalid_budget(self):
        for length in (1, 10, 0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                build_input_ids(FakeTokenizer(), TEST[2], length)
        with self.assertRaises(ValueError):
            build_input_ids(FakeTokenizer(), {"prefix": "p", "body": "", "suffix": "s"}, 512)

    def test_old_template_signature_fallback(self):
        tokenizer = OldTokenizer()
        ids = build_input_ids(tokenizer, TEST[0], 1024)
        self.assertEqual(len(ids), 1024)
        self.assertTrue(tokenizer.decode(ids).endswith("<|im_end|>\n<|im_start|>assistant\n"))

    def test_boundary_crossing_tokens_are_preserved(self):
        tokenizer = BoundaryMergeTokenizer()
        fixture = {"prefix": "PREFIX", "body": "ABCDEFGHIJKLMNZ", "suffix": "SUFFIX", "needle": "NEEDLE"}
        ids = build_input_ids(tokenizer, fixture, 256)
        text = tokenizer.decode(ids)
        self.assertEqual(len(ids), 256)
        self.assertIn("PREFIX\n\nA", text)
        self.assertIn("Z\n\nNEEDLE\n\nSUFFIX", text)
        self.assertTrue(text.endswith("<think>\n\n</think>\n\n"))

    def test_template_offset_mismatch_is_rejected(self):
        class Mismatch(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                output = super().apply_chat_template(messages, **kwargs)
                return output + [999] if kwargs["tokenize"] else output
        with self.assertRaisesRegex(ValueError, "differ"):
            build_input_ids(Mismatch(), TEST[0], 1024)

    def test_missing_offsets_and_non_typeerror_not_hidden(self):
        class MissingOffsets(FakeTokenizer):
            def __call__(self, text, **kwargs):
                return super().__call__(text, add_special_tokens=False)
        with self.assertRaisesRegex(ValueError, "offset"):
            build_input_ids(MissingOffsets(), TEST[0], 1024)

        class Broken(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                raise RuntimeError("broken template")
        with self.assertRaisesRegex(RuntimeError, "broken template"):
            build_input_ids(Broken(), TEST[0], 1024)

    def test_build_messages_default_body_and_needle(self):
        messages = build_messages(TEST[2])
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertNotIn("<cycle>", messages[-1]["content"])
        self.assertIn(TEST[2]["needle"], messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
