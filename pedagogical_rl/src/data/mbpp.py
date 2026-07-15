"""
MBPP data layer.

Loads the MBPP ("Mostly Basic Python Problems") dataset and normalizes every
record into a single immutable `Problem` shape, so the rest of the pipeline never
has to know which MBPP variant it came from.

Two variants exist with DIFFERENT field names -- this module hides that:
    full ("mbpp")            : text,   code, test_list, test_setup_code (str)
    sanitized (config=...)   : prompt, code, test_list, test_imports    (list)

The `Problem.tests` feed straight into rewards.executor.run_tests(...).
Requires `pip install datasets`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    """One normalized coding task."""

    task_id: int
    prompt: str                 # natural-language description shown to the tutor/student
    reference_solution: str     # gold code (for reference/analysis, NOT given to models)
    tests: tuple[str, ...]      # assert statements -> executor.run_tests
    setup_code: str = ""        # imports/fixtures run before the candidate

    # tests is a *tuple*, not a list: a frozen dataclass must stay hashable, and a
    # list is mutable/unhashable. Tuple keeps the whole Problem immutable + hashable.


# MBPP's conventional task_id splits (the dataset ships as one table; you slice it).
# Reference: the original MBPP paper / HF card.
_SPLIT_RANGES = {
    "prompt": range(1, 11),         # 1-10   : few-shot exemplars
    "test": range(11, 511),         # 11-510 : evaluation
    "validation": range(511, 601),  # 511-600
    "train": range(601, 975),       # 601-974: training
}


def _normalize(record: dict) -> Problem:
    """Map a raw MBPP record (either variant) into a Problem."""
    # Description: full uses `text`, sanitized uses `prompt`.
    prompt = record.get("text") or record.get("prompt") or ""

    # Setup: full has `test_setup_code` (a string); sanitized has `test_imports`
    # (a list of import lines). Collapse either into one setup string.
    if record.get("test_setup_code"):
        setup_code = record["test_setup_code"]
    else:
        setup_code = "\n".join(record.get("test_imports", []))

    return Problem(
        task_id=int(record["task_id"]),
        prompt=prompt.strip(),
        reference_solution=record["code"],
        tests=tuple(record["test_list"]),
        setup_code=setup_code,
    )


def load_mbpp(split: str = "test", *, sanitized: bool = True) -> list[Problem]:
    """Load and normalize MBPP.

    Args:
        split: one of "train", "test", "validation", "prompt".
        sanitized: use the 427-problem human-verified subset (recommended for
            reliable rewards). Set False for the full ~974-problem set.

    Returns:
        A list of Problem objects for the requested split, sorted by task_id.

    Raises:
        ImportError: if `datasets` is not installed.
        ValueError: if `split` is not a known MBPP split name.
    """
    if split not in _SPLIT_RANGES:
        raise ValueError(f"unknown split {split!r}; expected one of {list(_SPLIT_RANGES)}")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("MBPP loading needs HuggingFace datasets: pip install datasets") from e

    # Both variants live under the "google-research-datasets/mbpp" repo; the
    # sanitized subset is a named config. We load every split of the table and
    # slice by task_id so split semantics are identical across variants.
    config = "sanitized" if sanitized else "full"
    raw = load_dataset("google-research-datasets/mbpp", config, split="train+test+validation+prompt")

    wanted = _SPLIT_RANGES[split]
    problems = [_normalize(r) for r in raw if int(r["task_id"]) in wanted]
    problems.sort(key=lambda p: p.task_id)
    return problems
