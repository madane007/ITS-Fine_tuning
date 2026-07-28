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


class VLLMEngine:
    """Same LMEngine interface, backed by vLLM. Paged-attention + internal request
    scheduling means huge batches (hundreds of sequences) don't OOM -- vLLM
    processes them in memory-safe waves. Inference-only (no gradients), so it's for
    rollouts / eval, not the training loss step.

    LoRA is handled natively: enable_lora=True at construction, then set_adapter(path)
    before generating -- no merging, no reloading, no hot-swap bugs. set_adapter(None)
    = base model.
    """

    def __init__(
        self,
        model_id: str,
        *,
        gpu_memory_utilization: float = 0.9,   # lower this to fit >1 model per GPU
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        enable_lora: bool = False,
        max_lora_rank: int = 16,
        max_model_len: int = 4096,
    ):
        from vllm import LLM

        self.model_id = model_id
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature
        self.top_p = top_p

        self.llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=enable_lora,
            max_lora_rank=max_lora_rank,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="auto",
        )
        self._lora = None
        self._lora_counter = 0

    def set_adapter(self, adapter_path: str | None) -> None:
        """Point the tutor at a LoRA adapter (or None for the base model)."""
        from vllm.lora.request import LoRARequest

        if adapter_path is None:
            self._lora = None
        else:
            self._lora_counter += 1
            self._lora = LoRARequest(f"adapter{self._lora_counter}", self._lora_counter, adapter_path)

    def generate(
        self,
        chats: list[list[dict]],
        *,
        n: int = 1,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        greedy: bool = False,
    ) -> list[list[str]]:
        from vllm import SamplingParams

        if not chats:
            return []
        params = SamplingParams(
            n=n,
            temperature=0.0 if greedy else (
                self.default_temperature if temperature is None else temperature),
            top_p=1.0 if greedy else self.top_p,
            max_tokens=max_new_tokens or self.default_max_new_tokens,
        )
        # vLLM applies the model's chat template and schedules the whole batch safely.
        outputs = self.llm.chat(chats, sampling_params=params, lora_request=self._lora, use_tqdm=False)
        return [[o.text for o in req.outputs] for req in outputs]

    def one(self, chat: list[dict], **kw) -> str:
        return self.generate([chat], n=1, **kw)[0][0]

    def k_samples(self, chat: list[dict], k: int, **kw) -> list[str]:
        return self.generate([chat], n=k, **kw)[0]


def load_vllm(model_id: str, **kwargs) -> VLLMEngine:
    return VLLMEngine(model_id, **kwargs)
