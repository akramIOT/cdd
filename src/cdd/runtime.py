"""Shared tokenizer, chat encoding, and checkpoint loading for CDD."""

from __future__ import annotations

import gc
import os
from typing import Any, Optional

import torch


def hf_token(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get("HF_TOKEN")


def resolve_device_map(device: str, device_map: Any = None) -> Any:
    """One device string loads every checkpoint onto that device.

    Pass device_map="auto" or a Hugging Face device map to shard later
    without a second loader.
    """
    if device_map is None:
        return {"": device}
    return device_map


def eos_id_set(tokenizer) -> set[int]:
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, (list, tuple, set)):
        ids.update(int(item) for item in eos if item is not None)
    elif eos is not None:
        ids.add(int(eos))
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None and not isinstance(pad, (list, tuple)):
        ids.add(int(pad))
    return ids


def load_tokenizer(model_id: str, token: Optional[str]):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_id: str, device: str, dtype, token: Optional[str], device_map: Any = None):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=token,
        trust_remote_code=True,
        device_map=resolve_device_map(device, device_map),
    )


def encode_chat(
    tokenizer,
    device: str,
    system_prompt: str,
    user_prompt: str,
    prefill: str = "",
) -> torch.Tensor:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = result["input_ids"] if isinstance(result, dict) else result
    ids = ids.to(device)
    if prefill:
        prefix = tokenizer.encode(prefill, return_tensors="pt", add_special_tokens=False).to(device)
        ids = torch.cat([ids, prefix], dim=1)
    return ids


def forward_last(model, input_ids: torch.Tensor, past=None):
    with torch.no_grad():
        if past is None:
            out = model(input_ids, use_cache=True)
        else:
            out = model(input_ids, past_key_values=past, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


class ForwardHandle:
    """One forward through a module, optionally toggling a LoRA adapter off."""

    def __init__(self, model, *, toggle_adapter_off: bool = False, ensure_adapter_on: bool = False):
        self.model = model
        self.toggle_adapter_off = toggle_adapter_off
        self.ensure_adapter_on = ensure_adapter_on

    def forward(self, input_ids: torch.Tensor, past=None):
        if self.ensure_adapter_on and hasattr(self.model, "enable_adapter_layers"):
            self.model.enable_adapter_layers()
        if self.toggle_adapter_off and hasattr(self.model, "disable_adapter_layers"):
            self.model.disable_adapter_layers()
            try:
                return forward_last(self.model, input_ids, past)
            finally:
                self.model.enable_adapter_layers()
        return forward_last(self.model, input_ids, past)


def release(*objects) -> None:
    for obj in objects:
        if obj is None:
            continue
        if hasattr(obj, "cpu"):
            obj.cpu()
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
