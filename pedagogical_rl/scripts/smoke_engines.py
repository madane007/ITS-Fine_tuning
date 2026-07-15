"""
GPU smoke test: prove engines.py works and connects to conversation.py + mbpp.py.

Run on obelix in the `grpo` env (loads the small 3B model, ~6GB fp16):
    conda activate grpo
    cd ~/pedagogical_trainer
    python scripts/smoke_engines.py

Expects a GPU. Downloads the 3B model on first run (point HF_HOME at scratch if
your home disk is tight).
"""

import sys
from pathlib import Path

# Make `src` importable when run as `python scripts/smoke_engines.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mbpp import load_mbpp
from src.env.conversation import Conversation, Role
from src.models.engines import load_engine

MODEL = "unsloth/Llama-3.2-3B-Instruct"  # small, ungated; stands in for the student

print("loading engine:", MODEL)
eng = load_engine(MODEL, max_new_tokens=200)
print("loaded on:", eng.model.device)

# 1. Bare generation works?
chat = [{"role": "user", "content": "Write a Python function add(a,b) returning a+b. "
                                    "Only the function in a ```python block."}]
print("\n--- one() greedy ---")
print(eng.one(chat, greedy=True))

# 2. K-sampling works (this is how we get K student solutions for r_sol)?
print("\n--- k_samples(k=2) ---")
for i, s in enumerate(eng.k_samples(chat, k=2)):
    print(f"[sample {i}] {s[:80]!r}")

# 3. Integration: generate a real student turn from a Conversation's student view.
p = load_mbpp("test")[0]
c = Conversation(p)
c.record_pre_dialog([])                       # open the dialogue
c.add_teacher_message("What built-in could remove duplicate characters?")
student_view = c.render_for(Role.STUDENT)     # conversation builds what the model sees
reply = eng.one(student_view)                 # engine runs the model on it
print("\n--- student turn on MBPP task", p.task_id, "---")
print(reply[:300])

print("\nOK: engines.py generates and connects to conversation.py + mbpp.py")
