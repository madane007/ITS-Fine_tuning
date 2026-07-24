"""
Evaluate a trained tutor adapter against the BASE tutor.

Runs the same held-out MBPP problems through the full tutoring pipeline twice --
once with the untrained base tutor, once with the LoRA adapter applied -- and
reports the paper's two headline metrics:

    delta solve rate  = post-dialog solve rate - pre-dialog (untutored) solve rate
                        -> did tutoring actually help the student?
    pedagogy pass     = fraction of dialogues where all judges accepted
                        -> (1 - this) is roughly the leak/unhelpful rate

A trained tutor should have HIGHER delta_solve and HIGHER pedagogy_pass than base.

Uses the TEST split (training used "train"), so this is held-out.
The tutor is loaded once and the adapter applied afterwards, so peak memory is
one tutor, not two.

    PYTHONPATH=. python scripts/evaluate.py
"""

import json
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

ADAPTER_PATH = "./step90"       # <-- point this at the downloaded adapter folder

TUTOR_GPU, STUDENT_GPU, JUDGE_GPU = 0, 1, 2   # all 0 if single big GPU

N_PROBLEMS = 20        # held-out problems to evaluate on
GROUP_SIZE = 2         # dialogues per problem (averaging, not GRPO groups here)
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
    """Run all problems through the pipeline with this tutor; aggregate metrics."""
    # Same seed before each variant so the student/judge sampling is comparable.
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
    turns = [r.conversation.turn_count for g in groups for r in g]
    m = {
        "variant": label,
        "n_dialogues": len(bd),
        "solve_pre": _mean([b.pre_solve_rate for b in bd]),
        "solve_post": _mean([b.r_sol for b in bd]),
        "delta_solve": _mean([b.delta_solve_rate for b in bd]),
        "pedagogy_pass": _mean([1.0 if b.pedagogy_passed else 0.0 for b in bd]),
        "mean_reward": _mean([b.total for b in bd]),
        "mean_turns": _mean(turns),
    }
    print(
        f"[{label:8s}] solve {m['solve_pre']:.3f}->{m['solve_post']:.3f} "
        f"(delta {m['delta_solve']:+.3f}) | ped_pass {m['pedagogy_pass']:.3f} "
        f"| reward {m['mean_reward']:+.3f} | turns {m['mean_turns']:.1f}"
    )
    return m


def main():
    problems = load_mbpp("test")[:N_PROBLEMS]     # held out from training
    print(f"evaluating on {len(problems)} held-out problems, "
          f"{GROUP_SIZE} dialogues each\n")

    print("loading student + judge (frozen)...")
    student = load_engine(STUDENT_ID, max_new_tokens=MAX_NEW_TOKENS,
                          device_map={"": STUDENT_GPU})
    judge = load_engine(JUDGE_ID, max_new_tokens=MAX_NEW_TOKENS,
                        device_map={"": JUDGE_GPU})

    print("loading BASE tutor...")
    tutor = load_engine(TUTOR_ID, max_new_tokens=MAX_NEW_TOKENS,
                        device_map={"": TUTOR_GPU})
    base = evaluate(tutor, student, judge, problems, "base")

    print(f"\napplying adapter from {ADAPTER_PATH} ...")
    tutor.model = PeftModel.from_pretrained(tutor.model, ADAPTER_PATH)
    tutor.model.eval()
    trained = evaluate(tutor, student, judge, problems, "trained")

    # ---- verdict ----
    d_delta = trained["delta_solve"] - base["delta_solve"]
    d_ped = trained["pedagogy_pass"] - base["pedagogy_pass"]
    print("\n" + "=" * 62)
    print(f"delta_solve   base {base['delta_solve']:+.3f} -> trained "
          f"{trained['delta_solve']:+.3f}   (change {d_delta:+.3f})")
    print(f"pedagogy_pass base {base['pedagogy_pass']:.3f} -> trained "
          f"{trained['pedagogy_pass']:.3f}   (change {d_ped:+.3f})")
    print("=" * 62)
    if d_delta > 0 and d_ped >= 0:
        print("RESULT: trained tutor teaches better AND stayed pedagogical.")
    elif d_delta > 0:
        print("RESULT: better teaching, but pedagogy dropped -- check for leaking.")
    elif d_ped > 0:
        print("RESULT: more pedagogical, but not better at raising solve rate.")
    else:
        print("RESULT: no improvement over base on this sample.")

    with open(OUT_JSON, "w") as f:
        json.dump({"base": base, "trained": trained}, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
