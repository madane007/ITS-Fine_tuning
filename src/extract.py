# src/extract.py
# Parses the PostgreSQL SQL dump and extracts strategy1 training examples.

import json
import re
import random
from collections import defaultdict
from prompts import (
    build_personality_negative_prompt,
    build_positive_prompt,
    make_example,
)

SEED = 42
random.seed(SEED)

S1_COLS = {
    "id": 0, "task_id": 1, "personality": 2, "problem_text": 3,
    "test_cases": 4, "conversation": 5, "execution_result": 6,
    "solved": 7, "tests_passed": 8, "total_tests": 9, "turns": 10,
    "timestamp": 11, "date_folder": 12, "quality_bucket": 13,
    "model_used": 14, "created_at": 15, "discarded": 16,
}
S1_EXPECTED_COLS = 17


def safe_json(raw: str):
    """Parse JSON safely, return None on failure."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def looks_like_python(text: str) -> bool:
    """Heuristic: does this string contain Python code?
    Needs at least 2 structural signals to avoid false positives
    on plain English tutor messages.
    """
    if not text or len(text.strip()) < 20:
        return False
    signals = [
        r"\bdef\s+\w+\s*\(",
        r"\breturn\b",
        r"\bfor\s+\w+\s+in\b",
        r"\bif\s+.+:",
        r"\bwhile\s+.+:",
        r"^\s{2,}",
    ]
    matches = sum(1 for s in signals if re.search(s, text, re.MULTILINE))
    return matches >= 2


def extract_code(content: str) -> str | None:
    """Extract Python code from a tutor turn.
    Handles: pure code, ```python ... ``` blocks, or def blocks in mixed text.
    """
    content = content.replace("\\r\\n", "\n").replace("\\n", "\n").strip()

    if looks_like_python(content):
        return content

    code_block = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
    if code_block:
        c = code_block.group(1).strip()
        if looks_like_python(c):
            return c

    def_match = re.search(r"(def\s+\w+\s*\(.*?)(?:\n\n|\Z)", content, re.DOTALL)
    if def_match:
        c = def_match.group(1).strip()
        if looks_like_python(c):
            return c

    return None


def get_tutor_code_turns(conversation: list) -> list[str]:
    """Return all tutor code outputs in order, deduped (consecutive identical removed)."""
    codes = []
    for msg in conversation:
        if msg.get("role") != "tutor":
            continue
        code = extract_code(msg.get("content", ""))
        if code and (not codes or code != codes[-1]):
            codes.append(code)
    return codes


def extract_strategy1(sql_path: str) -> list[dict]:
    """Parse SQL dump and return strategy1 training examples."""

    print(f"\n🔍 Parsing SQL dump: {sql_path}")
    with open(sql_path, encoding="utf-8", errors="replace") as f:
        sql_lines = f.readlines()

    # Find the strategy1 COPY block
    s1_start = None
    for i, line in enumerate(sql_lines):
        if line.startswith("COPY public.strategy1_conversations "):
            s1_start = i + 1
            break
    if s1_start is None:
        raise ValueError("Could not find strategy1_conversations COPY block")

    s1_rows = []
    for line in sql_lines[s1_start:]:
        stripped = line.strip()
        if stripped.startswith("COPY ") or stripped.startswith("--"):
            break
        if not stripped or stripped == r"\.":
            continue
        if stripped.count("\t") == S1_EXPECTED_COLS - 1:
            s1_rows.append(stripped)

    print(f"   Raw strategy1 rows: {len(s1_rows)}")

    # Build training examples
    examples = []
    stats = defaultdict(int)
    per_personality = defaultdict(lambda: {"neg": 0, "pos": 0})

    for row in s1_rows:
        cols = row.split("\t")
        if len(cols) != S1_EXPECTED_COLS:
            stats["wrong_col_count"] += 1
            continue

        problem     = cols[S1_COLS["problem_text"]]
        personality = cols[S1_COLS["personality"]]
        solved      = cols[S1_COLS["solved"]] == "t"
        discarded   = cols[S1_COLS["discarded"]] == "t"

        if discarded:
            stats["discarded"] += 1
            continue
        if not problem or not personality:
            stats["missing_problem_or_personality"] += 1
            continue

        conversation = safe_json(cols[S1_COLS["conversation"]])
        if not conversation:
            stats["bad_json"] += 1
            continue

        tutor_codes = get_tutor_code_turns(conversation)
        if not tutor_codes:
            stats["no_code_turns"] += 1
            continue

        if len(tutor_codes[0].strip()) < 30:
            stats["first_attempt_too_short"] += 1
            continue

        first_attempt = tutor_codes[0]
        final_code    = tutor_codes[-1]
        one_shot      = (first_attempt == final_code)

        if one_shot and solved:
            pos_prompt = build_positive_prompt(problem)
            examples.append(make_example(pos_prompt, first_attempt))
            per_personality[personality]["pos"] += 1
            stats["positive_added_from_one_shot"] += 1
        else:
            neg_prompt = build_personality_negative_prompt(problem, personality)
            examples.append(make_example(neg_prompt, first_attempt))
            per_personality[personality]["neg"] += 1
            stats["negative_added"] += 1

            if solved and final_code != first_attempt:
                pos_prompt = build_positive_prompt(problem)
                examples.append(make_example(pos_prompt, final_code))
                per_personality[personality]["pos"] += 1
                stats["positive_added_from_fix"] += 1

    print(f"\n   Strategy1 examples extracted: {len(examples)}")
    print(f"\n   Per personality breakdown:")
    print(f"   {'PERSONALITY':25s} {'NEG':>6s} {'POS':>6s}")
    for p, counts in per_personality.items():
        print(f"   {p:25s} {counts['neg']:>6d} {counts['pos']:>6d}")
    print(f"\n   Skipped rows:")
    for reason, count in stats.items():
        if reason in ("negative_added", "positive_added"):
            continue
        print(f"     {reason:40s} {count}")

    return examples