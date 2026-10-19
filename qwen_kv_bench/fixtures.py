"""Original synthetic fixtures for the Qwen KV representation experiment.

These fixtures provide coverage, not a representative quality benchmark. Splits
have different topics, entities and answer strings. Different lengths of one
fixture remain correlated. Train on TRAIN only; DEV is for model selection and
TEST is for the final frozen comparison.

Exact sizing operates on the real chat-template token stream. Only tokens
entirely inside the repetitive user-body character span may be removed; the
user prefix, optional needle, question and assistant generation suffix survive.
This requires a fast tokenizer with character offset mappings, as provided by
the Qwen tokenizer. No decoding/re-encoding occurs after token removal.
"""

from __future__ import annotations

import math
from typing import Any


SYSTEM = "You are a helpful assistant. Answer the user's question clearly."

TRAIN = (
    {
        "name": "train_ferry_maintenance",
        "kind": "text",
        "prefix": "Read the maintenance diary of a small coastal ferry.",
        "body": (
            "Inspection <cycle>: Leena checked the stern gate before the first "
            "crossing. The crew cleaned salt from the hinges, measured the "
            "hydraulic pressure and recorded the seal condition. A spare hose "
            "remained in the workshop. The supervisor compared the readings "
            "with yesterday's log before signing the departure sheet.\n"
        ),
        "suffix": "Explain why the crew compares measurements across days.",
    },
    {
        "name": "train_python_bounded_ring",
        "kind": "code",
        "prefix": "Review this bounded event-buffer implementation and its notes.",
        "body": (
            "def keep_recent_<cycle>(events, capacity):\n"
            "    if capacity < 0:\n"
            "        raise ValueError('negative capacity')\n"
            "    buffer = []\n"
            "    for event in events:\n"
            "        buffer.append(event)\n"
            "        if len(buffer) > capacity:\n"
            "            buffer.pop(0)\n"
            "    return buffer\n"
            "# Keep arrival order among the retained events.\n\n"
        ),
        "suffix": "Explain the result when capacity is zero, then name one useful test.",
    },
    {
        "name": "train_seed_vault_records",
        "kind": "structured",
        "prefix": "Read the seed-vault inventory. Match accession IDs exactly.",
        "body": (
            "Accession barley-<cycle>: cabinet R; viability 92; tray 14.\n"
            "Accession oat-<cycle>: cabinet S; viability 88; tray 26.\n"
            "Accession rye-<cycle>: cabinet T; viability 95; tray 39.\n"
        ),
        "needle": "Accession millet-581: cabinet W; viability 97; tray 63.",
        "suffix": "Which tray contains accession millet-581? Give only the tray number.",
        "target_answer": "63",
    },
)

DEV = (
    {
        "name": "dev_orchard_sensor_log",
        "kind": "structured",
        "prefix": "Read the orchard sensor observations and the final calibration note.",
        "body": (
            "Reading <cycle>: probe west reports soil moisture 31 percent; "
            "probe east reports 28 percent. The shaded reference reports "
            "temperature 17 degrees. A technician records battery voltage "
            "before cleaning the protective cover. Rainfall is recorded "
            "separately from the irrigation timer.\n"
        ),
        "needle": "Calibration note: reference probe hazel-427 has offset minus 3.",
        "suffix": "What offset is recorded for reference probe hazel-427?",
        "target_answer": "minus 3",
    },
)

TEST = (
    {
        "name": "test_pottery_kiln_diary",
        "kind": "text",
        "prefix": "Read a pottery studio's kiln-firing diary.",
        "body": (
            "Firing <cycle>: Tamsin arranged glazed bowls on separate shelves "
            "and left space around the thermocouple. She raised the temperature "
            "gradually, checked the witness cones and noted the vent position. "
            "After cooling, each shelf was photographed before the pieces were "
            "removed. Small glaze samples stayed beside their matching bowls "
            "so that later firings could be compared fairly.\n"
        ),
        "suffix": "Why are glaze samples kept with the bowls after firing?",
    },
    {
        "name": "test_python_dependency_walk",
        "kind": "code",
        "prefix": "Review a dependency traversal with explicit cycle detection.",
        "body": (
            "def visit_<cycle>(node, graph, active, complete):\n"
            "    if node in active:\n"
            "        raise ValueError('dependency cycle')\n"
            "    if node in complete:\n"
            "        return\n"
            "    active.add(node)\n"
            "    for parent in graph.get(node, []):\n"
            "        visit_<cycle>(parent, graph, active, complete)\n"
            "    active.remove(node)\n"
            "    complete.add(node)\n\n"
        ),
        "suffix": "Explain why both the active and complete sets are necessary.",
    },
    {
        "name": "test_museum_loan_register",
        "kind": "structured",
        "prefix": "Read the museum loan register. Use the exact object identifier.",
        "body": (
            "Object mosaic-<cycle>: lender Bellmere; case H; return month March.\n"
            "Object compass-<cycle>: lender Norwick; case L; return month June.\n"
            "Object tapestry-<cycle>: lender Esthaven; case P; return month October.\n"
        ),
        "needle": "Object astrolabe-936: lender Quillport; case Y; return month February.",
        "suffix": "Which lender owns object astrolabe-936? Give only the lender name.",
        "target_answer": "Quillport",
    },
)

SPLITS = {"train": TRAIN, "dev": DEV, "test": TEST}
for _split, _fixtures in SPLITS.items():
    for _fixture in _fixtures:
        _fixture["split"] = _split


def _user_content(fixture: dict[str, Any], body: str) -> tuple[str, int, int]:
    prefix = str(fixture["prefix"]) + "\n\n"
    tail = "\n\n"
    if fixture.get("needle"):
        tail += str(fixture["needle"]) + "\n\n"
    tail += str(fixture["suffix"])
    return prefix + body + tail, len(prefix), len(prefix) + len(body)


def build_messages(fixture: dict[str, Any], body: str | None = None) -> list[dict[str, str]]:
    """Build stable Qwen instruct messages; use a supplied body when sizing."""
    if body is None:
        body = str(fixture["body"]).replace("<cycle>", "0")
    content, _, _ = _user_content(fixture, body)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def _template(tokenizer: Any, messages: list[dict[str, str]], *, tokenize: bool) -> Any:
    # Older supported tokenizers may reject enable_thinking. Do not hide any
    # other exception or silently change generation-prompt behavior.
    kwargs = {"tokenize": tokenize, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("tokenizer output must be a list, tuple, or array")
    return list(value)


def _ids(value: Any) -> list[int]:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    values = _list(value)
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("expected one tokenized prompt")
        values = _list(values[0])
    return [int(x) for x in values]


def _encoded_body_span(tokenizer: Any, fixture: dict[str, Any], body: str) -> tuple[list[int], int, int]:
    messages = build_messages(fixture, body)
    canonical_ids = _ids(_template(tokenizer, messages, tokenize=True))
    rendered = _template(tokenizer, messages, tokenize=False)
    if not isinstance(rendered, str):
        raise TypeError("chat template with tokenize=False must return text")
    content, body_start, body_end = _user_content(fixture, body)
    if rendered.count(content) != 1:
        raise ValueError("chat template must contain the complete user content exactly once")
    user_start = rendered.index(content)
    body_start += user_start
    body_end += user_start
    try:
        mapped = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, NotImplementedError) as exc:
        raise ValueError("exact body sizing requires a fast tokenizer with offset mappings") from exc
    if _ids(mapped) != canonical_ids:
        raise ValueError("chat-template IDs differ from rendered-text IDs; refusing unsafe truncation")
    if "offset_mapping" not in mapped:
        raise ValueError("tokenizer did not return offset mappings")
    offsets = _list(mapped["offset_mapping"])
    if len(offsets) != len(canonical_ids):
        raise ValueError("offset count differs from token count")
    interior = [i for i, pair in enumerate(offsets)
                if len(pair) == 2 and body_start <= int(pair[0]) < int(pair[1]) <= body_end]
    if not interior:
        raise ValueError("fixture body contains no independently trimmable tokens")
    first, end = interior[0], interior[-1] + 1
    if interior != list(range(first, end)):
        raise ValueError("body offsets are not contiguous; refusing unsafe truncation")
    return canonical_ids, first, end


def build_input_ids(tokenizer: Any, fixture: dict[str, Any], target_tokens: int) -> list[int]:
    """Return exactly target_tokens IDs while preserving all fixed chat content.

    Numbered body repeats supply enough input. A contiguous interior token span
    is shortened, retaining the front portion of the body. Boundary tokens that
    overlap user prefix or suffix are never removed. All optional needle text
    is outside the removable body span. Very small budgets raise ValueError.
    """
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens <= 0:
        raise ValueError("target_tokens must be a positive integer")
    if not str(fixture.get("body", "")):
        raise ValueError("fixture body must be nonempty")
    repeats = 1
    for _ in range(32):
        body = "".join(str(fixture["body"]).replace("<cycle>", str(i)) for i in range(repeats))
        ids, first, end = _encoded_body_span(tokenizer, fixture, body)
        fixed = len(ids) - (end - first)
        if target_tokens < fixed:
            raise ValueError(f"target_tokens={target_tokens} cannot preserve {fixed} fixed template/user tokens")
        if len(ids) >= target_tokens:
            keep = target_tokens - fixed
            result = ids[:first + keep] + ids[end:]
            assert len(result) == target_tokens
            return result
        missing = target_tokens - len(ids)
        repeats += max(1, math.ceil(missing / max(1, end - first)) * repeats)
    raise ValueError("could not build a sufficiently long body")
