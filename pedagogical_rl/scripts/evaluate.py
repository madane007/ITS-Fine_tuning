"""
Evaluate the BASE tutor and one-or-more trained checkpoints on held-out MBPP.

Runs the same held-out problems through the tutoring pipeline for each variant
(base, then each checkpoint) and prints a comparison table of the paper's headline
metrics, so you can see WHERE training peaked -- RL often improves then degrades,
so the last checkpoint isn't necessarily the best.

    delta_solve   = post-dialog solve rate - pre-dialog (untutored)  -> did it teach?
    pedagogy_pass = fraction of dialogues all judges accepted        -> did it not leak?

Memory-efficient: loads the tutor once, then hot-swaps LoRA adapters. Any
checkpoint whose path doesn't exist is skipped, so you can point it at whatever
checkpoints you actually have.

    PYTHONPATH=. python scripts/evaluate.py
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel

from src.data.mbpp import load_mbpp
from src.env.classroom import Classroom
from src.models.engines import load_engine

# ============================ CONFIG ============================
TUTOR_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
STUDENT_ID = "unsloth/Llama-3.2-3B-Instruct"
JUDGE_ID = "Qwen/Qwen2.5-7B-Instruct"

# label -> adapter path. Edit paths to where YOUR checkpoints are; missing ones
# are skipped automatically (so it's fine to list all even if you only have some).
CHECKPOINTS = {
    "step30": "./checkpoints/step30",
    "step60": "./checkpoints/step60",
    "step90": "./checkpoints/step90",
}

# All on GPU 0 (a single-GPU qsub job). If you requested multiple GPUs, spread
# them out (e.g. 0, 1, 1) to relieve memory pressure.
TUTOR_GPU, STUDENT_GPU, JUDGE_GPU = 0, 0, 0

N_PROBLEMS = 25        # held-out problems
GROUP_SIZE = 2         # dialogues per problem (averaged)
K_SOLUTIONS = 8        # student solutions per dialogue -> solve rate
MAX_TURNS = 4
JUDGE_SAMPLES = 2
MAX_NEW_TOKENS = 256
LAMBDA_PED = 0.75
SEED = 0
OUT_JSON = "./eval_results.json"
# ===============================================================


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(tutor, student, judge, problems, label):
    # Re-seed before each variant so student/judge sampling is identical across
    # variants -> differences are attributable to the TUTOR, not luck.
    random.seed(SEED)
    torch.manual_seed(SEED)

    room = Classroom(
        tutor, student, judge,
        group_size=GROUP_SIZE, k_solutions=K_SOLUTIONS,
        judge_samples=JUDGE_SAMPLES, max_turns=MAX_TURNS, lambda_ped=LAMBDA_PED,
    )
    with torch.no_grad():
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


def load_tutor(adapter_path):
    """Fresh base tutor, with the adapter MERGED IN (baked into the weights) so
    there's zero ambiguity about which adapter is active. Reloading per variant is
    slower but foolproof -- hot-swapping adapters on one model silently failed."""
    tutor = load_engine(TUTOR_ID, max_new_tokens=MAX_NEW_TOKENS, device_map={"": TUTOR_GPU})
    if adapter_path:
        tutor.model = PeftModel.from_pretrained(tutor.model, adapter_path)
        tutor.model = tutor.model.merge_and_unload()   # bake adapter into base weights
    tutor.model.eval()
    return tutor


def main():
    import gc

    problems = load_mbpp("test")[:N_PROBLEMS]
    print(f"evaluating on {len(problems)} held-out problems, {GROUP_SIZE} dialogues each\n")

    print("loading student + judge (frozen)...")
    student = load_engine(STUDENT_ID, max_new_tokens=MAX_NEW_TOKENS, device_map={"": STUDENT_GPU})
    judge = load_engine(JUDGE_ID, max_new_tokens=MAX_NEW_TOKENS, device_map={"": JUDGE_GPU})

    # base first, then every checkpoint whose path exists
    variants = [("base", None)]
    variants += [(n, p) for n, p in CHECKPOINTS.items() if os.path.exists(p)]

    results = []
    print("\n== evaluating variants (reloading tutor each time) ==")
    for name, path in variants:
        print(f"-- loading tutor: {name}")
        tutor = load_tutor(path)
        results.append(evaluate(tutor, student, judge, problems, name))
        del tutor                      # free the 15GB tutor before the next variant
        gc.collect()
        torch.cuda.empty_cache()

    # ---- comparison table ----
    print("\n" + "=" * 74)
    print(f"{'variant':10s} {'solve_pre':>9} {'solve_post':>10} {'delta':>8} {'ped_pass':>9} {'reward':>8}")
    print("-" * 74)
    for m in results:
        print(f"{m['variant']:10s} {m['solve_pre']:>9.3f} {m['solve_post']:>10.3f} "
              f"{m['delta_solve']:>+8.3f} {m['pedagogy_pass']:>9.3f} {m['mean_reward']:>+8.3f}")
    print("=" * 74)

    best_delta = max(results, key=lambda m: m["delta_solve"])
    best_ped = max(results, key=lambda m: m["pedagogy_pass"])
    best_rew = max(results, key=lambda m: m["mean_reward"])
    print(f"best delta_solve   : {best_delta['variant']} ({best_delta['delta_solve']:+.3f})")
    print(f"best pedagogy_pass : {best_ped['variant']} ({best_ped['pedagogy_pass']:.3f})")
    print(f"best mean_reward   : {best_rew['variant']} ({best_rew['mean_reward']:+.3f})")
    if best_rew["variant"] == "base":
        print("\n-> No checkpoint beat base on reward. Training didn't help (yet).")
    else:
        print(f"\n-> {best_rew['variant']} is the best tutor. Use that adapter.")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
