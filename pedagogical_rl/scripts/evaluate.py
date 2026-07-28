"""
Evaluate the BASE tutor and trained checkpoints on held-out MBPP -- vLLM backend.

vLLM handles the big generation batches with paged attention (no OOM) and is much
faster than transformers. The tutor uses vLLM's NATIVE LoRA: one base model, swap
the adapter per variant with set_adapter() -- no reloading, no merging, no
hot-swap bug.

All three models share ONE GPU via gpu_memory_utilization splits (must sum < ~0.95).
On a single 80GB A100 the defaults below fit. If vLLM complains about memory, lower
the *_MEM fractions.

    delta_solve   = post-dialog solve rate - pre-dialog (untutored)  -> did it teach?
    pedagogy_pass = fraction of dialogues all judges accepted        -> did it not leak?

    PYTHONPATH=. python scripts/evaluate.py
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.mbpp import load_mbpp
from src.env.classroom import Classroom
from src.models.engines import load_vllm

# ============================ CONFIG ============================
TUTOR_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
STUDENT_ID = "unsloth/Llama-3.2-3B-Instruct"
JUDGE_ID = "Qwen/Qwen2.5-7B-Instruct"

CHECKPOINTS = {
    "step30": "./checkpoints/step30",
    "step60": "./checkpoints/step60",
    "step90": "./checkpoints/step90",
}

# 3 vLLM models on ONE GPU -> split its memory (fractions of total, sum < ~0.95).
STUDENT_MEM = 0.20     # 3B  -> ~16GB on an 80GB card
JUDGE_MEM = 0.35       # 7B  -> ~28GB
TUTOR_MEM = 0.35       # 7B  -> ~28GB
# If you have >1 GPU, you can instead give each 0.9 and place them via
# CUDA_VISIBLE_DEVICES in separate processes -- but one 80GB GPU fits all three.

N_PROBLEMS = 25
GROUP_SIZE = 2
K_SOLUTIONS = 8
MAX_TURNS = 4
JUDGE_SAMPLES = 2
MAX_NEW_TOKENS = 256
LAMBDA_PED = 0.75
SEED = 0
OUT_JSON = "./eval_results.json"
# ===============================================================


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(room, problems, label):
    # Re-seed so student/judge sampling is comparable across variants -> differences
    # are attributable to the TUTOR. (vLLM uses its own seeding per request; this
    # keeps the Python-side problem shuffling identical.)
    random.seed(SEED)
    torch.manual_seed(SEED)

    groups = room.rollout_batch(problems)
    bd = [r.reward for g in groups for r in g]
    m = {
        "variant": label,
        "solve_pre": _mean([b.pre_solve_rate for b in bd]),
        "solve_post": _mean([b.r_sol for b in bd]),
        "delta_solve": _mean([b.delta_solve_rate for b in bd]),
        "pedagogy_pass": _mean([1.0 if b.pedagogy_passed else 0.0 for b in bd]),
        "mean_reward": _mean([b.total for b in bd]),
    }
    print(f"  [{label:8s}] solve {m['solve_pre']:.3f}->{m['solve_post']:.3f} "
          f"(delta {m['delta_solve']:+.3f}) | ped_pass {m['pedagogy_pass']:.3f} "
          f"| reward {m['mean_reward']:+.3f}")
    return m


def main():
    problems = load_mbpp("test")[:N_PROBLEMS]
    print(f"evaluating on {len(problems)} held-out problems, {GROUP_SIZE} dialogues each\n")

    print("loading student + judge + tutor on vLLM (shared GPU)...")
    student = load_vllm(STUDENT_ID, gpu_memory_utilization=STUDENT_MEM, max_new_tokens=MAX_NEW_TOKENS)
    judge = load_vllm(JUDGE_ID, gpu_memory_utilization=JUDGE_MEM, max_new_tokens=MAX_NEW_TOKENS)
    tutor = load_vllm(TUTOR_ID, gpu_memory_utilization=TUTOR_MEM, max_new_tokens=MAX_NEW_TOKENS,
                      enable_lora=True, max_lora_rank=16)

    room = Classroom(
        tutor, student, judge,
        group_size=GROUP_SIZE, k_solutions=K_SOLUTIONS,
        judge_samples=JUDGE_SAMPLES, max_turns=MAX_TURNS, lambda_ped=LAMBDA_PED,
    )

    variants = [("base", None)]
    variants += [(n, p) for n, p in CHECKPOINTS.items() if os.path.exists(p)]

    results = []
    print("\n== evaluating variants (swapping the tutor's LoRA) ==")
    for name, path in variants:
        print(f"-- variant: {name}")
        tutor.set_adapter(path)          # None = base; else apply that adapter
        results.append(evaluate(room, problems, name))

    # ---- comparison table ----
    print("\n" + "=" * 74)
    print(f"{'variant':10s} {'solve_pre':>9} {'solve_post':>10} {'delta':>8} {'ped_pass':>9} {'reward':>8}")
    print("-" * 74)
    for m in results:
        print(f"{m['variant']:10s} {m['solve_pre']:>9.3f} {m['solve_post']:>10.3f} "
              f"{m['delta_solve']:>+8.3f} {m['pedagogy_pass']:>9.3f} {m['mean_reward']:>+8.3f}")
    print("=" * 74)

    best = max(results, key=lambda m: m["mean_reward"])
    print(f"best mean_reward: {best['variant']} ({best['mean_reward']:+.3f})")
    if best["variant"] == "base":
        print("-> No checkpoint beat base. Training didn't help (yet).")
    else:
        print(f"-> {best['variant']} is the best tutor. Use that adapter.")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
