"""
GPU smoke test for classroom.py: run ONE small batched rollout end-to-end.

Uses the 3B model for tutor=student=judge (quality irrelevant -- we're only
proving the full loop runs and produces rewards). Small G/K/turns for speed.

    conda activate grpo
    cd ~/pedagogical_trainer
    PYTHONPATH=. python scripts/smoke_classroom.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mbpp import load_mbpp
from src.env.classroom import Classroom
from src.models.engines import load_engine

MODEL = "unsloth/Llama-3.2-3B-Instruct"

print("loading one 3B engine, shared across all three roles...")
eng = load_engine(MODEL, max_new_tokens=200)

# Same engine object for tutor/student/judge -- fine for a plumbing test.
room = Classroom(
    tutor=eng, student=eng, judge=eng,
    group_size=2,      # 2 conversations for this one problem (tiny GRPO group)
    k_solutions=2,     # 2 final solutions each
    judge_samples=1,   # 1 sample per rubric (2 rubrics -> 2 verdicts)
    max_turns=2,       # short dialogue for speed
)

problems = load_mbpp("test")[:1]   # a single problem
print(f"rolling out {room.group_size} conversations for task {problems[0].task_id}...")

grouped = room.rollout_batch(problems)

print("\n=== results ===")
for pi, group in enumerate(grouped):
    print(f"problem {group[0].problem.task_id}: group of {len(group)}")
    for gi, r in enumerate(group):
        b = r.reward
        print(f"  conv {gi}: total={b.total:.3f}  r_sol={b.r_sol:.2f}  r_ped={b.r_ped:.0f}  "
              f"judges={b.n_judges_accept}/{b.n_judges_total}  turns={r.conversation.turn_count}")

print("\nOK: classroom.rollout_batch runs end-to-end and produces rewards")
