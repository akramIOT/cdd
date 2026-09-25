"""Contrastive decoding for Contrastive Decoding Diffing (CDD).

score(v) = (1 + alpha) * log P_trained(v) - alpha * log P_ref(v)

Tokens outside the top-k of P_trained are masked. The paper calls this
adversarial decoding; AdversarialDecoder is the public class name.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from cdd.runtime import (
    ForwardHandle,
    encode_chat,
    eos_id_set,
    load_causal_lm,
    load_tokenizer,
    release,
)


def contrastive_score(
    trained_logits: torch.Tensor, reference_logits: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    trained_log_probs = F.log_softmax(trained_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    score = (1 + alpha) * trained_log_probs - alpha * reference_log_probs
    # (1+α)(-∞) − α(−∞) is NaN. A non-finite score must not win the argmax.
    score = torch.where(torch.isfinite(score), score, torch.full_like(score, float("-inf")))
    return score, trained_log_probs


def gated_choice(trained_logits: torch.Tensor, reference_logits: torch.Tensor, alpha: float, top_k: int) -> int:
    """Vocabulary index selected by one greedy contrastive step."""
    if trained_logits.ndim == 1:
        trained_logits = trained_logits.unsqueeze(0)
        reference_logits = reference_logits.unsqueeze(0)
    score, trained_log_probs = contrastive_score(trained_logits, reference_logits, alpha)
    gated = apply_top_k_gate(score, trained_log_probs, top_k)
    return int(gated.argmax(dim=-1).item())


def sensitivity_grid(
    trained_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    alphas: list[float],
    top_ks: list[int],
) -> list[dict[str, float | int]]:
    """Chosen index for each (alpha, top-k) pair. No model weights are loaded."""
    rows = []
    for alpha in alphas:
        for top_k in top_ks:
            rows.append(
                {
                    "alpha": alpha,
                    "top_k": top_k,
                    "index": gated_choice(trained_logits, reference_logits, alpha, top_k),
                }
            )
    return rows


def apply_top_k_gate(score: torch.Tensor, trained_log_probs: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return score
    k = min(top_k, trained_log_probs.shape[-1])
    threshold = trained_log_probs.topk(k, dim=-1).values[:, -1:]
    gated = score.clone()
    gated[trained_log_probs < threshold] = float("-inf")
    return gated


class AdversarialDecoder:
    """Contrastive decoder.

    ref_mode="base", lora=True
        Trained = adapter on. Reference = adapter off. One checkpoint.
    ref_mode="base", lora=False
        Trained = model_id. Reference = base_model_id. Two checkpoints.
    ref_mode="self_prompt"
        Both passes use the trained model. A LoRA adapter still needs
        base_model_id so the adapter can be attached.
    """

    def __init__(
        self,
        model_id: str,
        base_model_id: Optional[str] = None,
        lora: bool = False,
        device: str = "cuda",
        hf_token: Optional[str] = None,
        dtype=torch.bfloat16,
        device_map=None,
    ):
        self.model_id = model_id
        self.base_model_id = base_model_id
        self.lora = lora
        self.device = device
        self.hf_token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
        self.dtype = dtype
        self.device_map = device_map
        self._model = None
        self._base = None
        self._tok = None

    def __enter__(self) -> "AdversarialDecoder":
        return self.load()

    def __exit__(self, *_):
        self.unload()

    def load(self) -> "AdversarialDecoder":
        if self._model is not None:
            return self
        token = self.hf_token
        self._tok = load_tokenizer(self.model_id, token)
        if self.lora:
            from peft import PeftModel

            if self.base_model_id is None:
                raise ValueError("base_model_id is required when lora=True")
            base = load_causal_lm(self.base_model_id, self.device, self.dtype, token, self.device_map)
            self._model = PeftModel.from_pretrained(base, self.model_id, token=token)
        else:
            self._model = load_causal_lm(self.model_id, self.device, self.dtype, token, self.device_map)
            if self.base_model_id is not None:
                self._base = load_causal_lm(
                    self.base_model_id, self.device, self.dtype, token, self.device_map
                )
                self._base.eval()
        self._model.eval()
        return self

    def unload(self) -> None:
        release(self._model, self._base, self._tok)
        self._model = None
        self._base = None
        self._tok = None

    def _handles(self, ref_mode: str) -> tuple[ForwardHandle, ForwardHandle]:
        if ref_mode == "base" and self._base is None and not self.lora:
            raise ValueError("ref_mode='base' requires base_model_id or lora=True")
        trained = ForwardHandle(self._model, ensure_adapter_on=self.lora)
        if self.lora and ref_mode == "base":
            reference = ForwardHandle(self._model, toggle_adapter_off=True)
        elif ref_mode == "base":
            reference = ForwardHandle(self._base)
        else:
            reference = ForwardHandle(self._model, ensure_adapter_on=self.lora)
        return trained, reference

    def generate(
        self,
        prompt: str,
        ref_mode: str = "base",
        safety_prompt: str = "",
        trained_prompt: str = "",
        prefill: str = "",
        alpha: float = 1.0,
        top_k: int = 20,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        self.load()
        trained_handle, reference_handle = self._handles(ref_mode)
        trained_ids = encode_chat(self._tok, self.device, trained_prompt, prompt, prefill)
        reference_ids = encode_chat(self._tok, self.device, safety_prompt, prompt, prefill)
        trained_kv, trained_logits = trained_handle.forward(trained_ids)
        reference_kv, reference_logits = reference_handle.forward(reference_ids)
        stop_ids = eos_id_set(self._tok)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            score, trained_log_probs = contrastive_score(trained_logits, reference_logits, alpha)
            score = apply_top_k_gate(score, trained_log_probs, top_k)
            if do_sample:
                probs = F.softmax(score / max(temperature, 1e-8), dim=-1)
                next_id = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_id = score.argmax(dim=-1)
            token_id = int(next_id.item())
            if token_id in stop_ids:
                break
            generated.append(token_id)
            step = next_id.view(1, 1)
            trained_kv, trained_logits = trained_handle.forward(step, trained_kv)
            reference_kv, reference_logits = reference_handle.forward(step, reference_kv)

        return self._tok.decode(generated, skip_special_tokens=True)

    def greedy_generate(
        self,
        prompt: str,
        system_prompt: str = "",
        prefill: str = "",
        max_new_tokens: int = 200,
    ) -> str:
        self.load()
        if self.lora:
            self._model.enable_adapter_layers()
        input_ids = encode_chat(self._tok, self.device, system_prompt, prompt, prefill)
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.pad_token_id,
                eos_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(out[0, prompt_len:], skip_special_tokens=True)

    def compute_perplexity(
        self,
        prompt: str,
        response: str,
        system_prompt: str = "",
        prefill: str = "",
    ) -> float:
        """Perplexity of response under the clean base model."""
        self.load()
        if self._base is not None:
            ref = self._base
            restore = False
        elif self.lora:
            self._model.disable_adapter_layers()
            ref = self._model
            restore = True
        else:
            ref = self._model
            restore = False

        try:
            prompt_ids = encode_chat(self._tok, self.device, system_prompt, prompt, prefill)
            response_ids = self._tok.encode(response, return_tensors="pt", add_special_tokens=False).to(self.device)
            if response_ids.shape[1] == 0:
                return float("nan")
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            labels = full_ids.clone()
            labels[:, : prompt_ids.shape[1]] = -100
            with torch.no_grad():
                loss = ref(full_ids, labels=labels).loss
            return float(torch.exp(loss).item())
        finally:
            if restore:
                self._model.enable_adapter_layers()
