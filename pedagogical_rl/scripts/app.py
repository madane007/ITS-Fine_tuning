"""
Streamlit checkpoint tester: pick a base model + (optional) LoRA checkpoint +
an MBPP task (or custom system prompt), then chat with the tutor live.

This is the GUI counterpart to `infer.py` -- same engine, same TEACHER_SYSTEM
state machine, but as a long-running server so you can flip between
checkpoints/tasks without re-invoking a CLI each time.

Run (from the `pedagogical_rl/` directory):
    PYTHONPATH=. streamlit run scripts/app.py

To keep it running in the background (e.g. on a remote GPU box):
    PYTHONPATH=. nohup streamlit run scripts/app.py --server.port 8501 \\
        --server.address 0.0.0.0 > streamlit.log 2>&1 &
Then open http://<host>:8501 in a browser (use an SSH tunnel if the box isn't
publicly reachable: `ssh -L 8501:localhost:8501 <host>`).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from infer import CODE_BLOCK_INSTRUCTION, DEFAULT_MODEL_ID, find_problem, has_code_block
from src.env.conversation import END_TOKEN, TEACHER_SYSTEM
from src.models.engines import load_engine

DEFAULT_CKPT_DIR = REPO_ROOT / "checkpoints"
KNOWN_MODELS = [
    DEFAULT_MODEL_ID,          # tutor, matches TUTOR_ID in src/train/grpo.py
    "unsloth/Llama-3.2-3B-Instruct",  # student, matches STUDENT_ID
]

st.set_page_config(page_title="Tutor Checkpoint Tester", layout="wide")


def discover_checkpoints(root: Path) -> list[Path]:
    """Any dir under `root` containing adapter_config.json, newest/highest step first."""
    if not root.exists():
        return []
    hits = [p.parent for p in root.rglob("adapter_config.json")]

    def sort_key(p: Path):
        digits = "".join(c for c in p.name if c.isdigit())
        return (0, -int(digits)) if digits else (1, p.name)

    return sorted(hits, key=sort_key)


def build_system_prompt(task_mode: str, task_id: int | None, custom_prompt: str, require_code_block: bool) -> str:
    if task_mode == "MBPP task_id":
        problem = find_problem(task_id)
        system = TEACHER_SYSTEM.format(
            prompt=problem.prompt,
            tests="\n".join(problem.tests),
            solution=problem.reference_solution,
            end_token=END_TOKEN,
        )
        st.session_state.loaded_problem = problem
    else:
        system = custom_prompt
        st.session_state.loaded_problem = None
    if require_code_block:
        system = f"{system}\n\n{CODE_BLOCK_INSTRUCTION}"
    return system


@st.cache_resource(show_spinner=False)
def _load_base_engine(model_id: str, device: str, max_new_tokens: int):
    return load_engine(model_id, max_new_tokens=max_new_tokens, device_map=device)


def load_model(model_id: str, adapter_path: str | None, device: str, max_new_tokens: int):
    engine = _load_base_engine(model_id, device, max_new_tokens)

    # Undo any adapter from a previous selection so we start from a clean base
    # model every time (PeftModel.unload() strips the LoRA layers back off).
    from peft import PeftModel

    if isinstance(engine.model, PeftModel):
        engine.model = engine.model.unload()

    if adapter_path:
        engine.model = PeftModel.from_pretrained(engine.model, adapter_path)
        engine.model.eval()

    return engine


def tutor_turn(chat: list[dict], engine, gen_kwargs: dict, require_code_block: bool, max_retries: int) -> tuple[str, bool]:
    reply = engine.one(chat, **gen_kwargs)
    warn = False
    if require_code_block:
        attempts = 0
        while not has_code_block(reply) and attempts < max_retries:
            attempts += 1
            nudge = chat + [{"role": "user", "content": CODE_BLOCK_INSTRUCTION}]
            reply = engine.one(nudge, **gen_kwargs)
        warn = not has_code_block(reply)
    chat.append({"role": "assistant", "content": reply})
    return reply, warn


# --- sidebar: configuration -------------------------------------------------

st.sidebar.header("Model")
model_choice = st.sidebar.selectbox("Base model", KNOWN_MODELS + ["custom..."])
model_id = st.sidebar.text_input("Model id", value=model_choice) if model_choice == "custom..." else model_choice

ckpt_root_str = st.sidebar.text_input("Checkpoint root (scanned for adapters)", value=str(DEFAULT_CKPT_DIR))
checkpoints = discover_checkpoints(Path(ckpt_root_str))
checkpoint_labels = ["None (base model only)"] + [str(p) for p in checkpoints] + ["custom path..."]
checkpoint_choice = st.sidebar.selectbox("Checkpoint", checkpoint_labels)
if checkpoint_choice == "None (base model only)":
    adapter_path = None
elif checkpoint_choice == "custom path...":
    adapter_path = st.sidebar.text_input("Adapter path", value="")
else:
    adapter_path = checkpoint_choice

device = st.sidebar.selectbox("Device", ["auto", "cuda:0", "cpu"])

st.sidebar.header("Task")
task_mode = st.sidebar.radio("Prompt source", ["MBPP task_id", "Custom system prompt"])
task_id = None
custom_prompt = ""
if task_mode == "MBPP task_id":
    task_id = st.sidebar.number_input("task_id", min_value=1, value=4, step=1)
else:
    custom_prompt = st.sidebar.text_area("System prompt", height=150)

initial_message = st.sidebar.text_input("Initial student message (optional; tutor opens if blank)")

st.sidebar.header("Generation")
max_new_tokens = st.sidebar.slider("max_new_tokens", 32, 1024, 256, step=32)
temperature = st.sidebar.slider("temperature", 0.0, 1.5, 0.8, step=0.05)
greedy = st.sidebar.checkbox("Greedy decoding")
require_code_block = st.sidebar.checkbox("Require ```python code block in replies")
max_retries = st.sidebar.number_input("Max retries (code block)", min_value=0, value=2, step=1) if require_code_block else 0

load_clicked = st.sidebar.button("Load model + start conversation", type="primary")
reset_clicked = st.sidebar.button("Reset conversation (keep model)")

# --- session state -----------------------------------------------------------

st.session_state.setdefault("engine", None)
st.session_state.setdefault("chat", None)
st.session_state.setdefault("ended", False)
st.session_state.setdefault("loaded_problem", None)
st.session_state.setdefault("summary", None)

if load_clicked:
    try:
        with st.spinner(f"Loading {model_id}" + (f" + adapter {adapter_path}" if adapter_path else "") + " ..."):
            engine = load_model(model_id, adapter_path, device, max_new_tokens)
        system_prompt = build_system_prompt(task_mode, int(task_id) if task_id else None, custom_prompt, require_code_block)

        chat = [{"role": "system", "content": system_prompt}]
        if initial_message:
            chat.append({"role": "user", "content": initial_message})

        st.session_state.engine = engine
        st.session_state.chat = chat
        st.session_state.ended = False
        st.session_state.summary = {
            "model_id": model_id,
            "adapter_path": adapter_path,
            "task": f"task_id {task_id}" if task_mode == "MBPP task_id" else "custom prompt",
        }

        gen_kwargs = dict(temperature=temperature, greedy=greedy)
        with st.spinner("Generating opening turn..."):
            reply, warn = tutor_turn(chat, engine, gen_kwargs, require_code_block, max_retries)
        if warn:
            st.warning("Reply has no ```python code block after retries.")
        if END_TOKEN in reply:
            st.session_state.ended = True
    except Exception as e:
        st.error(f"Failed to load: {e}")
        raise

if reset_clicked and st.session_state.engine is not None and st.session_state.summary is not None:
    system_prompt = build_system_prompt(task_mode, int(task_id) if task_id else None, custom_prompt, require_code_block)
    chat = [{"role": "system", "content": system_prompt}]
    if initial_message:
        chat.append({"role": "user", "content": initial_message})
    st.session_state.chat = chat
    st.session_state.ended = False
    gen_kwargs = dict(temperature=temperature, greedy=greedy)
    with st.spinner("Generating opening turn..."):
        reply, warn = tutor_turn(chat, st.session_state.engine, gen_kwargs, require_code_block, max_retries)
    if warn:
        st.warning("Reply has no ```python code block after retries.")
    if END_TOKEN in reply:
        st.session_state.ended = True

# --- main area ----------------------------------------------------------------

st.title("Tutor Checkpoint Tester")

if st.session_state.summary:
    s = st.session_state.summary
    st.caption(f"model: `{s['model_id']}` | adapter: `{s['adapter_path'] or 'none'}` | {s['task']}")

if st.session_state.loaded_problem is not None:
    p = st.session_state.loaded_problem
    with st.expander(f"MBPP problem {p.task_id}", expanded=False):
        st.write(p.prompt)
        st.code("\n".join(p.tests))
        st.caption("Reference solution (hidden from the student, shown here for your own verification):")
        st.code(p.reference_solution, language="python")

if st.session_state.chat is None:
    st.info("Configure a model/checkpoint/task in the sidebar, then click **Load model + start conversation**.")
else:
    for msg in st.session_state.chat[1:]:
        role = "assistant" if msg["role"] == "assistant" else "user"
        avatar = "🧑‍🏫" if role == "assistant" else "🧑‍🎓"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    if st.session_state.ended:
        st.success(f"Tutor ended the session ({END_TOKEN}). Reset to start a new conversation.")

    student_msg = st.chat_input("Reply as the student...", disabled=st.session_state.ended or st.session_state.chat is None)
    if student_msg:
        st.session_state.chat.append({"role": "user", "content": student_msg})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(student_msg)
        gen_kwargs = dict(temperature=temperature, greedy=greedy)
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            with st.spinner("Tutor is thinking..."):
                reply, warn = tutor_turn(st.session_state.chat, st.session_state.engine, gen_kwargs, require_code_block, max_retries)
            st.markdown(reply)
            if warn:
                st.warning("Reply has no ```python code block after retries.")
        if END_TOKEN in reply:
            st.session_state.ended = True
            st.rerun()
