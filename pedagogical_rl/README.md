# ITS — Pedagogical RL for Code Tutoring

Training an LLM **tutor** to guide a **student** model through Python problems
*without giving away the answer*, using GRPO. A from-scratch adaptation of
[Dinucu-Jianu et al., EMNLP 2025 — "From Problem-Solving to Teaching Problem-Solving"](https://arxiv.org/abs/2505.15607)
([code](https://github.com/eth-lre/PedagogicalRL)) to the **MBPP** coding dataset.

The tutor is rewarded when the student *solves the problem after tutoring* **and**
the tutoring stayed pedagogical (didn't leak the solution, stayed helpful).

## Reward

```
r = r_sol + (r_ped - 1) * λ
```

- **`r_sol`** — post-dialog solve rate: sample K student solutions, run them against
  MBPP's tests in a sandbox, take the fraction that pass. (Ground-truth reward — no
  learned reward model needed, unlike the original math setup.)
- **`r_ped`** — product of binary judge verdicts (Answer-Leakage + Helpfulness); ALL
  must accept or it's 0.
- **`λ`** — pedagogy weight (default 0.75).

Optional shaping terms (from the reference implementation): early-`<end>` bonus,
`<think>`-usage bonus, over-length penalty.

## Architecture

```
src/
├── data/mbpp.py        Load + normalize MBPP -> Problem objects
├── rewards/
│   ├── executor.py     Sandboxed subprocess that runs candidate code vs tests
│   ├── functional.py   Model text -> extract code -> executor -> r_sol
│   └── shaping.py      Compose the full reward (paper formula + shaping)
├── env/
│   ├── conversation.py Dialogue state machine (one tutor<->student dialogue)
│   └── classroom.py    Batched rollouts: G conversations per problem
├── models/engines.py   Swappable LM inference wrapper (transformers now, vLLM later)
└── train/grpo.py       GRPO loop: group-relative advantages + LoRA on the tutor
```

**Models (real run):** tutor `Qwen2.5-Coder-7B-Instruct` (LoRA-trained), student
`Llama-3.2-3B-Instruct` (frozen), judge `Qwen2.5-7B-Instruct` (frozen). Each pinned
to its own GPU.

## Setup

```bash
conda create -n grpo python=3.11 -y && conda activate grpo
pip install -r requirements.txt
# On CUDA 11.8 hardware (e.g. V100), install the cu118 torch build instead:
# pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

## Run

The reward/data/env logic is CPU-testable; the rollouts + training need GPUs.

```bash
# unit-check the sandbox oracle (CPU, no GPU)
python -c "from src.rewards.executor import run_tests; print(run_tests('def add(a,b):return a+b', ['assert add(2,3)==5']))"

# GPU smoke test: one small model plays all roles, one batched rollout
PYTHONPATH=. python scripts/smoke_classroom.py

# real training run (edit CONFIG at the top of src/train/grpo.py first)
PYTHONPATH=. python src/train/grpo.py
```

Training logs rich per-step metrics to stdout and to `CKPT_DIR/metrics.jsonl`
(reward, pre/post solve rate, Δ solve rate, pedagogy-pass rate, turns, loss), and
saves the LoRA adapter to `CKPT_DIR/stepN/` every `SAVE_EVERY` steps.

Key `CONFIG` knobs in `src/train/grpo.py`: model IDs + per-GPU placement,
`GROUP_SIZE` (G), `K_SOLUTIONS` (K), `MAX_TURNS`, `LAMBDA_PED`, `NUM_STEPS`,
`CKPT_DIR`. Lower `GROUP_SIZE`/`MAX_TURNS`/`K_SOLUTIONS` if you hit OOM.

## Notes for reproduction

- **CUDA 11.8 / V100:** use fp16 (no bf16 on Volta), `attn_implementation="sdpa"`
  (no FlashAttention-2), and cu118 torch. Do **not** set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (breaks on old drivers).
- **Fitting the 7B tutor on a 32 GB card:** gradient checkpointing + per-turn
  backward are enabled in `grpo.py` for this reason.
- **Filter the dataset:** MBPP is easy for capable students; filter to the
  "learnable zone" (problems the student doesn't already solve) so rewards have
  variance. `scripts/colab_baseline.ipynb` measures per-problem solve rate.
- **Status:** full pipeline built and verified end-to-end on real models; tuning
  the learning dynamics (Δ solve rate, leak rate) is ongoing. vLLM inference and
  a PPO clip / KL term are planned optimizations.

## License / attribution

Method and reward design follow arXiv:2505.15607 (eth-lre/PedagogicalRL). MBPP is
Apache-2.0 (Google Research).
