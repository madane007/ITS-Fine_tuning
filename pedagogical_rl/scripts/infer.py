"""
Interactive inference: base model + (optional) LoRA adapter from a GRPO checkpoint.

Loads the tutor via the same TransformersEngine used in training/rollouts, then
wraps it with a PeftModel if --adapter points at a `checkpoints/stepN` dir saved
by src/train/grpo.py (tutor.model.save_pretrained). Drives an interactive
tutor<->you dialogue over one MBPP problem, using the exact TEACHER_SYSTEM
prompt/state machine from src/env/conversation.py so what you see matches what
the tutor saw during training.

Examples:
    # trained adapter against a real MBPP problem
    PYTHONPATH=. python scripts/infer.py --adapter checkpoints/step10 --task_id 5

    # base model, no adapter, for A/B comparison
    PYTHONPATH=. python scripts/infer.py --task_id 5 --no_adapter

    # skip MBPP entirely with a custom system prompt, single-shot (no REPL)
    PYTHONPATH=. python scripts/infer.py --adapter checkpoints/step10 \\
        --system "You are a helpful assistant." --message "hi" --once
"""

from __future__ import annotations

import argparse

from src.data.mbpp import Problem, load_mbpp
from src.env.conversation import END_TOKEN, TEACHER_SYSTEM
from src.models.engines import load_engine

# Matches TUTOR_ID in src/train/grpo.py -- the model the shipped adapters were
# trained against. Override with --model_id if you trained a different base.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"


def find_problem(task_id: int) -> Problem:
    """MBPP splits partition task_id ranges; search all of them for the id."""
    for split in ("train", "test", "validation", "prompt"):
        for p in load_mbpp(split):
            if p.task_id == task_id:
                return p
    raise ValueError(f"no MBPP problem with task_id={task_id}")


def build_system_prompt(args: argparse.Namespace) -> str:
    if args.system is not None:
        return args.system
    problem = find_problem(args.task_id)
    print(f"[problem {problem.task_id}] {problem.prompt}\n")
    return TEACHER_SYSTEM.format(
        prompt=problem.prompt,
        tests="\n".join(problem.tests),
        solution=problem.reference_solution,
        end_token=END_TOKEN,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="base model to load")
    ap.add_argument("--adapter", default=None, help="path to a saved LoRA adapter dir (e.g. checkpoints/step10)")
    ap.add_argument("--no_adapter", action="store_true", help="run the bare base model, ignoring --adapter")
    ap.add_argument("--device", default="auto", help='device_map, e.g. "auto", "cuda:0", "cpu"')

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--task_id", type=int, help="MBPP task_id to tutor on")
    src.add_argument("--system", help="custom system prompt, bypassing MBPP entirely")

    ap.add_argument("--message", default=None, help="first student message; if omitted, the tutor opens")
    ap.add_argument("--once", action="store_true", help="generate a single turn and exit (no REPL)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--greedy", action="store_true", help="deterministic decoding instead of sampling")
    args = ap.parse_args()

    print(f"loading base model: {args.model_id}")
    engine = load_engine(args.model_id, max_new_tokens=args.max_new_tokens, device_map=args.device)

    if args.adapter and not args.no_adapter:
        from peft import PeftModel

        print(f"loading LoRA adapter: {args.adapter}")
        engine.model = PeftModel.from_pretrained(engine.model, args.adapter)
        engine.model.eval()

    system_prompt = build_system_prompt(args)
    chat = [{"role": "system", "content": system_prompt}]
    if args.message:
        chat.append({"role": "user", "content": args.message})

    gen_kwargs = dict(temperature=args.temperature, greedy=args.greedy)

    def tutor_turn() -> str:
        reply = engine.one(chat, **gen_kwargs)
        chat.append({"role": "assistant", "content": reply})
        print(f"\ntutor> {reply}\n")
        return reply

    reply = tutor_turn()
    if args.once:
        return

    while END_TOKEN not in reply:
        try:
            student_msg = input("student> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if student_msg in ("/quit", "/exit"):
            break
        chat.append({"role": "user", "content": student_msg})
        reply = tutor_turn()


if __name__ == "__main__":
    main()
