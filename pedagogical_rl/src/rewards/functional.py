"""
Functional-correctness reward.

Turns a student model's *raw text output* into a correctness score by:
  1. extracting the Python code from the model's chatter (```python ... ``` fences),
  2. running it against the Problem's tests via the sandboxed executor.

This is the "did the student actually solve it" signal. Pedagogical shaping
(withholding, Socratic style, etc.) lives separately in shaping.py.
"""

from __future__ import annotations

import re

from ..data.mbpp import Problem
from .executor import ExecutionResult, run_tests

# Matches a fenced code block, with or without a language tag:
#   ```python\n<code>\n```   or   ```\n<code>\n```
# DOTALL so <code> can span newlines; non-greedy so we don't swallow past a fence.
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code(raw_output: str) -> str:
    """Pull the submitted Python out of a model's raw response.

    Models wrap code in markdown fences and surround it with prose. We take the
    LAST fenced block, since a model that reasons then answers puts its final
    solution last. If there are no fences, we assume the whole output is code.
    """
    blocks = _FENCE_RE.findall(raw_output)
    if blocks:
        return blocks[-1].strip()
    return raw_output.strip()


def functional_reward(
    problem: Problem,
    raw_output: str,
    *,
    timeout_s: float = 10.0,
    memory_mb: int = 512,
) -> ExecutionResult:
    """Score a student's raw solution text against a Problem's tests.

    Args:
        problem: the task, carrying `tests` and `setup_code`.
        raw_output: the student model's unedited response (code + chatter).
        timeout_s / memory_mb: forwarded to the sandboxed executor.

    Returns:
        ExecutionResult -- use `.reward` for the scalar, or the flags/error for
        shaping and logging. Never raises.
    """
    code = extract_code(raw_output)
    return run_tests(
        code,
        list(problem.tests),
        setup_code=problem.setup_code,
        timeout_s=timeout_s,
        memory_mb=memory_mb,
    )
