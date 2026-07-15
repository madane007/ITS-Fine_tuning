"""
Model engines -- the swappable inference layer.

One `LMEngine` interface, one `transformers`-backed implementation. The tutor,
student, and judge are all just instances of TransformersEngine pointed at
different checkpoints. classroom.py talks to `.generate(chats, n=...)` and never
touches transformers directly -- so when a cu118-compatible vLLM is available,
you add a VLLMEngine with the same `.generate` signature and swap it in with zero
changes upstream.

Tuned for Volta V100s: fp16 (no bf16 on Volta), attn_implementation="sdpa"
(FlashAttention-2 is unsupported pre-Ampere), left-padding for decoder-only
batched generation.
"""

from __future__ import annotations

from typing import Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LMEngine(Protocol):
    """The contract classroom.py depends on. Any backend that implements this
    (transformers now, vLLM later) is a drop-in replacement."""

    def generate(
        self,
        chats: list[list[dict]],
        *,
        n: int = 1,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        greedy: bool = False,
    ) -> list[list[str]]:
        """For each chat (a list of {role, content}), return `n` completions.
        Output shape: len(chats) x n strings."""
        ...


class TransformersEngine:
    """A frozen (or LoRA-wrapped) HF model behind the LMEngine interface."""

    def __init__(
        self,
        model_id: str,
        *,
        dtype: torch.dtype = torch.float16,   # Volta: fp16, NOT bf16
        device_map: str | dict = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.model_id = model_id
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Decoder-only batched generation needs LEFT padding so completions align.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="sdpa",   # FA2 unsupported on V100
        )
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        chats: list[list[dict]],
        *,
        n: int = 1,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        greedy: bool = False,
    ) -> list[list[str]]:
        if not chats:
            return []

        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        temperature = self.default_temperature if temperature is None else temperature

        # Render each chat with the model's template, then batch with left-padding.
        prompts = [
            self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in chats
        ]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if greedy:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=self.top_p)

        out = self.model.generate(**enc, **gen_kwargs)
        completions = self.tokenizer.batch_decode(
            out[:, prompt_len:], skip_special_tokens=True
        )

        # generate() flattens to (len(chats) * n) rows in order; regroup to per-chat.
        return [completions[i * n : (i + 1) * n] for i in range(len(chats))]

    # -- convenience wrappers for classroom.py readability ------------------

    def one(self, chat: list[dict], **kw) -> str:
        """Single completion for a single chat (e.g. a tutor turn)."""
        return self.generate([chat], n=1, **kw)[0][0]

    def k_samples(self, chat: list[dict], k: int, **kw) -> list[str]:
        """K completions for one chat (e.g. K student solutions for r_sol)."""
        return self.generate([chat], n=k, **kw)[0]


def load_engine(model_id: str, **kwargs) -> TransformersEngine:
    """Factory -- swap the class here when a vLLM backend exists."""
    return TransformersEngine(model_id, **kwargs)
