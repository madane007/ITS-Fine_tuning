"""
GRPO training loop for the tutor -- the final piece.

Each step:
  1. sample a batch of problems
  2. classroom.rollout_batch -> G conversations per problem, each with a reward
  3. per group: advantage_i = (reward_i - mean) / (std + eps)      [GRPO baseline]
  4. for each conversation, recompute the log-prob of the TUTOR's generated tokens
     (with gradients) and form the policy-gradient loss  -advantage * logprob
  5. backprop -> update the tutor's LoRA adapter

This is a minimal, readable GRPO: REINFORCE with a group-relative baseline, LoRA on
the tutor, student+judge frozen. No PPO clipping / KL term yet -- add later if the
policy drifts. Rollout generation runs under no_grad (in engines); only the
log-prob recomputation here carries gradients.

Run on obelix (grpo env). Defaults use the 3B for ALL roles so you can smoke-test
the loop cheaply; switch to the paper models in CONFIG once it runs.

    PYTHONPATH=. python src/train/grpo.py
"""

from __future__ import annotations

import json
import os
import random

import torch
from peft import LoraConfig, get_peft_model

from src.data.mbpp import load_mbpp, Problem
from src.env.classroom import Classroom
from src.env.conversation import Role, TEACHER_SYSTEM, END_TOKEN
from src.models.engines import load_engine

# ============================ CONFIG ============================
# Real models: tutor is code-specialized Qwen, student is the weaker 3B (headroom),
# judge is a 7B (lighter than the paper's 14B for a first run -- scale up later).
TUTOR_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
STUDENT_ID = "unsloth/Llama-3.2-3B-Instruct"
JUDGE_ID = "Qwen/Qwen2.5-7B-Instruct"

# GPU placement (4x V100-32GB): pin each model to its OWN card so three
# device_map="auto" loads don't all pile onto GPU 0. Each ~7B fp16 fits in 32GB.
TUTOR_GPU = 0     # + LoRA + optimizer + training activations
STUDENT_GPU = 0
JUDGE_GPU = 0     # GPU 3 spare (headroom / bump a model here if one OOMs)

PROBLEMS_PER_STEP = 2      # problems sampled per training step
GROUP_SIZE = 4             # G: conversations per problem (need >1 for advantages)
K_SOLUTIONS = 4            # K: student solutions for the solve rate (paper uses 8; lower = less memory)
MAX_TURNS = 4
JUDGE_SAMPLES = 2
MAX_NEW_TOKENS = 256

LR = 1e-5
NUM_STEPS = 10
SAVE_EVERY = 10
CKPT_DIR = os.environ.get("CKPT_DIR", "./checkpoints")
METRICS_LOG = os.path.join(CKPT_DIR, "metrics.jsonl")   # one JSON row per step, for plotting
LAMBDA_PED = 0.75
ADV_EPS = 1e-6
SEED = 0
# ===============================================================


def build_teacher_prompt(problem: Problem, history: list[dict]) -> list[dict]:
    """Reconstruct the exact chat the tutor SAW before a given turn -- mirrors
    Conversation.render_for(TEACHER) but for a message prefix, so we can score the
    log-prob of each tutor turn under the context it was generated in."""
    system = TEACHER_SYSTEM.format(
        prompt=problem.prompt,
        tests="\n".join(problem.tests),
        solution=problem.reference_solution,
        end_token=END_TOKEN,
    )
    chat = [{"role": "system", "content": system}]
    for m in history:
        speaker = "assistant" if m["role"] == Role.TEACHER else "user"
        chat.append({"role": speaker, "content": m["content"]})
    return chat


def tutor_turn_samples(conversation) -> list[tuple[list[dict], str]]:
    """(prompt_chat, completion_text) for every tutor turn in a conversation."""
    samples, history = [], []
    for m in conversation.messages:
        if m["role"] == Role.TEACHER:
            samples.append((build_teacher_prompt(conversation.problem, history), m["content"]))
        history.append(m)
    return samples


def completion_logprob(model, tokenizer, chat: list[dict], completion: str):
    """Sum of log-probs of `completion` tokens given `chat`, WITH gradients.
    Returns (summed_logprob, n_tokens)."""
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt_text + completion, return_tensors="pt").input_ids.to(model.device)
    n_prompt = prompt_ids.shape[1]

    logits = model(full_ids).logits[:, :-1, :]      # position t predicts token t+1
    targets = full_ids[:, 1:]
    # Fused cross-entropy = per-token NLL without materializing a full [L, vocab]
    # softmax tensor (big for Qwen's ~152k vocab). log-prob = -NLL.
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    )                                                # [L-1]
    comp_nll = nll[n_prompt - 1:]                    # only the completion's tokens
    return -comp_nll.sum(), comp_nll.numel()


def group_advantages(rewards: list[float]) -> list[float]:
    """GRPO baseline: standardize rewards within a group."""
    t = torch.tensor(rewards, dtype=torch.float32)
    adv = (t - t.mean()) / (t.std() + ADV_EPS)
    return adv.tolist()


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)   # holds checkpoints + metrics.jsonl

    # --- models: tutor is trainable (LoRA); student + judge frozen. Each pinned
    #     to its own GPU via device_map={"": gpu_index}. ---
    print(f"loading tutor (LoRA) on GPU {TUTOR_GPU}:", TUTOR_ID)
    tutor = load_engine(TUTOR_ID, max_new_tokens=MAX_NEW_TOKENS, device_map={"": TUTOR_GPU})
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    tutor.model = get_peft_model(tutor.model, lora)
    tutor.model.train()
    # Gradient checkpointing: recompute layer activations during backward instead of
    # storing them all -> big VRAM saving, lets the 7B tutor train on a 32GB V100.
    # (~30% slower.) enable_input_require_grads is needed for it to work with LoRA.
    tutor.model.gradient_checkpointing_enable()
    tutor.model.enable_input_require_grads()
    tutor.model.print_trainable_parameters()

    print(f"loading student on GPU {STUDENT_GPU}:", STUDENT_ID)
    student = load_engine(STUDENT_ID, max_new_tokens=MAX_NEW_TOKENS, device_map={"": STUDENT_GPU})
    print(f"loading judge on GPU {JUDGE_GPU}:", JUDGE_ID)
    judge = load_engine(JUDGE_ID, max_new_tokens=MAX_NEW_TOKENS, device_map={"": JUDGE_GPU})

    room = Classroom(
        tutor, student, judge,
        group_size=GROUP_SIZE, k_solutions=K_SOLUTIONS,
        judge_samples=JUDGE_SAMPLES, max_turns=MAX_TURNS, lambda_ped=LAMBDA_PED,
    )

    optimizer = torch.optim.AdamW(
        [p for p in tutor.model.parameters() if p.requires_grad], lr=LR
    )

    problems = load_mbpp("train")          # TODO: filter to the learnable-zone set
    print(f"training on {len(problems)} problems\n")

    for step in range(1, NUM_STEPS + 1):
        batch = random.sample(problems, PROBLEMS_PER_STEP)

        # 1-2. rollout (generation under no_grad). Turn the KV cache ON for fast
        #      generation, then OFF for the checkpointed backward pass (they're
        #      mutually exclusive).
        tutor.model.config.use_cache = True
        with torch.no_grad():
            groups = room.rollout_batch(batch)
        torch.cuda.empty_cache()   # free rollout activations before the loss pass
        tutor.model.config.use_cache = False

        # 3. advantages + 4-5. policy-gradient update. Backward PER TUTOR TURN so each
        #    turn's graph is freed immediately -> peak memory = one turn, not a whole
        #    conversation (this is what fixes the OOM on the 7B tutor).
        optimizer.zero_grad()
        flat = [(r, a) for g in groups
                for r, a in zip(g, group_advantages([x.reward.total for x in g]))]
        n_used, step_loss = 0, 0.0
        for result, adv in flat:
            if adv == 0.0:
                continue                    # no signal in this group
            samples = tutor_turn_samples(result.conversation)
            if not samples:
                continue
            for chat, completion in samples:
                lp, ntok = completion_logprob(tutor.model, tutor.tokenizer, chat, completion)
                if ntok == 0:
                    continue
                # mean log-prob for this turn, scaled so the whole step averages cleanly
                loss = -adv * (lp / ntok) / (len(flat) * len(samples))
                loss.backward()             # per-turn -> frees the graph now
                step_loss += loss.item()
            n_used += 1

        if n_used > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in tutor.model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

        # ---- metrics: aggregate every conversation this step ----
        bd = [r.reward for g in groups for r in g]
        turns = [r.conversation.turn_count for g in groups for r in g]
        m = {
            "step": step,
            "mean_reward": _mean([b.total for b in bd]),
            "reward_std": _std([b.total for b in bd]),        # >0 means GRPO has signal
            "solve_pre": _mean([b.pre_solve_rate for b in bd]),   # untutored baseline
            "solve_post": _mean([b.r_sol for b in bd]),           # after tutoring (r_sol)
            "delta_solve": _mean([b.delta_solve_rate for b in bd]),  # the headline metric
            "pedagogy_pass": _mean([1.0 if b.pedagogy_passed else 0.0 for b in bd]),
            "mean_turns": _mean(turns),
            "convs_updated": n_used,
            "loss": step_loss,
        }
        print(
            f"step {step:4d} | reward {m['mean_reward']:+.3f} +/-{m['reward_std']:.2f} "
            f"| solve {m['solve_pre']:.2f}->{m['solve_post']:.2f} (d{m['delta_solve']:+.2f}) "
            f"| ped_pass {m['pedagogy_pass']:.2f} | turns {m['mean_turns']:.1f} "
            f"| upd {n_used} | loss {step_loss:+.3f}"
        )
        with open(METRICS_LOG, "a") as f:
            f.write(json.dumps(m) + "\n")

        if step % SAVE_EVERY == 0:
            path = os.path.join(CKPT_DIR, f"step{step}")
            tutor.model.save_pretrained(path)
            print(f"  saved LoRA adapter -> {path}")

    print("done.")


if __name__ == "__main__":
    main()
