"""Small, original fixtures for a pretrained-model KV codec smoke test.

These are hand-written synthetic coverage cases, not a representative corpus or
a retrieval/quality benchmark. Calibration and held-out cases have disjoint
topics, names, and answer strings. Longer cases repeat and number fixture text;
different lengths of one fixture are correlated, not independent samples.

Public interface:
    CALIBRATION_PROMPTS / HELDOUT_PROMPTS: lists of prompt dictionaries.
    make_needle_prompt(fraction=0.5, variant=0): a structured lookup fixture.
    build_input_ids(tokenizer, prompt, target_tokens): exactly sized token IDs.

Use add_special_tokens=False consistently. No chat template is assumed: the
chosen SmolLM2-135M checkpoint is a base causal language model. The runner should
record actual token IDs, checkpoint revision, dtype, and attention backend.
"""

from __future__ import annotations

from typing import Any


CALIBRATION_PROMPTS = [
    {
        "name": "calibration_wetland_notes",
        "kind": "text",
        "split": "calibration",
        "prefix": "Field notebook: a wetland survey.\n",
        "body": (
            "Survey <cycle>: After the rain, a narrow stream crossed the meadow "
            "and entered a shallow pool. Reeds slowed the water, while fine soil "
            "settled between their roots. The observers marked the bank with a "
            "wooden stake and measured the water depth before sunset. A second "
            "team checked the gauge the next morning and compared its notes. "
            "They distinguished a change in water level from a change in the "
            "position of the measuring stake.\n"
        ),
        "suffix": "\nThe survey report explained that",
    },
    {
        "name": "calibration_python_queue",
        "kind": "code",
        "split": "calibration",
        "prefix": "# Python queue utility and review notes\n",
        "body": (
            "def drain_batch_<cycle>(queue, limit):\n"
            "    result = []\n"
            "    while queue and len(result) < limit:\n"
            "        item = queue.pop(0)\n"
            "        if item is not None:\n"
            "            result.append(item)\n"
            "    return result\n\n"
            "# Preserve FIFO order and skip empty items.\n"
            "# A production queue should use a deque for efficient removal.\n"
        ),
        "suffix": "\n# A useful boundary test for this function is",
    },
    {
        "name": "calibration_catalog_lookup",
        "kind": "retrieval",
        "split": "calibration",
        "prefix": "Catalog excerpts. Each line describes a different item.\n",
        "body": (
            "Item elm-<cycle>: shelf C, color brown, inspected on Tuesday.\n"
            "Item birch-<cycle>: shelf D, color white, inspected on Thursday.\n"
            "Item spruce-<cycle>: shelf E, color green, inspected on Monday.\n"
        ),
        "needle": "Item cedar-619: shelf J, color copper, inspected on Friday.\n",
        "needle_fraction": 0.5,
        "suffix": "\nQuestion: Which shelf holds item cedar-619?\nAnswer:",
        "target_answer": "J",
    },
]


HELDOUT_PROMPTS = [
    {
        "name": "heldout_observatory_story",
        "kind": "text",
        "split": "heldout",
        "prefix": "An account of a night at a small observatory.\n",
        "body": (
            "Entry <cycle>: Mira arrived while the sky was still pale. She "
            "opened the dome, checked the telescope balance, and placed a red "
            "lamp beside the notebook. Clouds hid the western horizon, but the "
            "eastern stars remained clear. Her colleague adjusted the focus "
            "while she recorded the time of each exposure. They kept the "
            "unprocessed images as well as the corrected copies so that a "
            "later reader could examine the original observations.\n"
        ),
        "suffix": "\nBefore closing the dome, Mira decided to",
    },
    {
        "name": "heldout_python_interval_merge",
        "kind": "code",
        "split": "heldout",
        "prefix": "# Review a function that merges overlapping intervals.\n",
        "body": (
            "def merge_intervals_<cycle>(intervals):\n"
            "    merged = []\n"
            "    for start, end in sorted(intervals):\n"
            "        if merged and start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n\n"
            "# Inputs may be empty, nested, adjacent, or disjoint.\n"
            "# Input order should not change the sorted union.\n"
        ),
        "suffix": "\n# Explain what happens when two intervals share an endpoint:\n#",
    },
    {
        "name": "heldout_delivery_lookup",
        "kind": "retrieval",
        "split": "heldout",
        "prefix": "Dispatch records. Use the exact record ID to answer.\n",
        "body": (
            "Delivery river-<cycle>: recipient Anika; locker 12; status ready.\n"
            "Delivery hill-<cycle>: recipient Joel; locker 34; status pending.\n"
            "Delivery lake-<cycle>: recipient Priya; locker 57; status ready.\n"
        ),
        "needle": "Delivery summit-842: recipient Dalia; locker 68; status held.\n",
        "needle_fraction": 0.5,
        "suffix": "\nQuestion: What is the locker number for summit-842?\nAnswer:",
        "target_answer": "68",
    },
]


for _fixture in CALIBRATION_PROMPTS + HELDOUT_PROMPTS:
    # Compatibility field for runners that size plain text themselves. To keep
    # a needle and the final question at every length, use build_input_ids.
    _fixture["text"] = (
        _fixture["prefix"]
        + _fixture["body"].replace("<cycle>", "0")
        + _fixture.get("needle", "")
        + _fixture["suffix"]
    )


def make_needle_prompt(fraction: float = 0.5, variant: int = 0) -> dict[str, Any]:
    """Return an original key/value lookup case with controlled placement.

    ``fraction`` controls the insertion position in the available filler, not
    an exact percentage of the whole token sequence. This fixture tests codec
    round trips on structured contexts. A 135M base model need not solve the
    lookup; generation equality and task accuracy must be reported separately.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if variant < 0:
        raise ValueError("variant must be nonnegative")
    record_id = f"vault-z{903 + variant}"
    answer = f"marigold-{731 + 13 * variant}"
    prompt: dict[str, Any] = {
        "name": f"heldout_needle_{variant}_position_{fraction:.2f}",
        "kind": "needle",
        "split": "heldout",
        "prefix": "Archive entries. Each record has one access phrase.\n",
        "body": (
            "Record north-<cycle>: phrase amber-harbor; owner team A.\n"
            "Record south-<cycle>: phrase silver-meadow; owner team B.\n"
            "Record east-<cycle>: phrase violet-bridge; owner team C.\n"
            "Record west-<cycle>: phrase indigo-forest; owner team D.\n"
        ),
        "needle": f"Record {record_id}: phrase {answer}; owner team Z.\n",
        "needle_fraction": fraction,
        "suffix": (
            f"\nQuestion: What is the access phrase for {record_id}?\nAnswer:"
        ),
        "target_answer": answer,
        "record_id": record_id,
    }
    prompt["text"] = (
        prompt["prefix"]
        + prompt["body"].replace("<cycle>", "0")
        + prompt["needle"]
        + prompt["suffix"]
    )
    return prompt


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def build_input_ids(
    tokenizer: Any, prompt: dict[str, Any], target_tokens: int
) -> list[int]:
    """Build exactly ``target_tokens`` IDs, preserving fixed text and needle.

    Text segments are tokenized independently and concatenated. This guarantees
    the exact length and insertion point; it is a synthetic token stream rather
    than tokenization of a single reconstructed text string. Numbered filler is
    truncated as needed, so it may end inside a source line or tokenized word.
    The complete encoded needle and final question remain present.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    prefix = _encode(tokenizer, prompt.get("prefix", ""))
    suffix = _encode(tokenizer, prompt.get("suffix", ""))
    needle = _encode(tokenizer, prompt.get("needle", ""))
    filler_length = target_tokens - len(prefix) - len(suffix) - len(needle)
    if filler_length < 0:
        raise ValueError(
            f"{prompt['name']}: target length {target_tokens} is smaller than "
            f"the fixed fields ({len(prefix) + len(suffix) + len(needle)} tokens)"
        )

    filler: list[int] = []
    cycle = 0
    body_template = prompt.get("body", "")
    while len(filler) < filler_length:
        part = _encode(tokenizer, body_template.replace("<cycle>", str(cycle)))
        if not part:
            raise ValueError("body must encode to at least one token")
        filler.extend(part)
        cycle += 1
    filler = filler[:filler_length]

    if needle:
        fraction = float(prompt.get("needle_fraction", 0.5))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("needle_fraction must be in [0, 1]")
        cut = int(round(fraction * len(filler)))
        result = prefix + filler[:cut] + needle + filler[cut:] + suffix
    else:
        result = prefix + filler + suffix

    if len(result) != target_tokens:
        raise AssertionError("internal error: prompt has wrong token length")
    return result


def fixture_manifest() -> dict[str, Any]:
    """Human-readable provenance and protocol notes for the result JSON."""
    return {
        "provenance": "Original hand-written synthetic smoke fixtures",
        "benchmark": False,
        "calibration_names": [p["name"] for p in CALIBRATION_PROMPTS],
        "heldout_names": [p["name"] for p in HELDOUT_PROMPTS],
        "sizing": "Independent segment tokenization; numbered/repeated filler",
        "caveat": (
            "Coverage of prose, Python, and structured retrieval is not evidence "
            "of representative long-context or task-quality performance. "
            "Different lengths/placements are correlated synthetic samples."
        ),
    }
