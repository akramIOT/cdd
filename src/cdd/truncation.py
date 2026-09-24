"""SVD rank truncation of a LoRA update for Contrastive Decoding Diffing (CDD).

The nonzero SVD of scale * B @ A is the SVD of the small r x r core in the
QR bases of B and A.T. Factors written back keep the effective update equal
to the rank-k truncation and never materialize the full delta-W.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from cdd.runtime import encode_chat, load_causal_lm, load_tokenizer, release


def _check_lora_pair(a_tensor: torch.Tensor, b_tensor: torch.Tensor, scale: float) -> None:
    if a_tensor.ndim != 2 or b_tensor.ndim != 2:
        raise ValueError(
            f"LoRA factors must be matrices, got A {tuple(a_tensor.shape)} and B {tuple(b_tensor.shape)}"
        )
    if int(a_tensor.shape[0]) != int(b_tensor.shape[1]):
        raise ValueError(
            f"LoRA inner dimensions disagree: A {tuple(a_tensor.shape)}, B {tuple(b_tensor.shape)}"
        )
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")


def top_rank_lora_factors(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    *,
    scale: float,
    target_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    if target_rank < 1:
        raise ValueError("target_rank must be >= 1")
    _check_lora_pair(a_tensor, b_tensor, scale)
    if scale == 0:
        return torch.zeros_like(a_tensor), torch.zeros_like(b_tensor), []

    compute_dtype = torch.float32
    a = a_tensor.to(dtype=compute_dtype)
    b = b_tensor.to(dtype=compute_dtype)
    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.T, mode="reduced")
    core = (r_b @ r_a.T) * float(scale)
    u_core, singular_values, vh_core = torch.linalg.svd(core, full_matrices=False)

    new_a = torch.zeros_like(a)
    new_b = torch.zeros_like(b)
    kept = min(
        int(target_rank),
        int(singular_values.numel()),
        int(a_tensor.shape[0]),
        int(b_tensor.shape[1]),
    )
    if kept < 1:
        return new_a.to(dtype=a_tensor.dtype), new_b.to(dtype=b_tensor.dtype), []

    sign = 1.0 if scale > 0 else -1.0
    kept_singular_values: list[float] = []
    for component_idx in range(kept):
        sigma = float(singular_values[component_idx].item())
        if sigma <= 0:
            continue
        u = q_b @ u_core[:, component_idx]
        v = q_a @ vh_core[component_idx, :]
        factor = math.sqrt(sigma / abs(float(scale)))
        new_b[:, component_idx] = u * factor
        new_a[component_idx, :] = v * (factor * sign)
        kept_singular_values.append(sigma)

    return new_a.to(dtype=a_tensor.dtype), new_b.to(dtype=b_tensor.dtype), kept_singular_values


def lora_singular_values(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    *,
    scale: float,
) -> list[float]:
    _check_lora_pair(a_tensor, b_tensor, scale)
    if scale == 0:
        return []
    compute_dtype = torch.float32
    b = b_tensor.to(dtype=compute_dtype)
    a = a_tensor.to(dtype=compute_dtype)
    _, r_b = torch.linalg.qr(b, mode="reduced")
    _, r_a = torch.linalg.qr(a.T, mode="reduced")
    core = (r_b @ r_a.T) * float(scale)
    singular_values = torch.linalg.svdvals(core)
    return [float(s) for s in singular_values.tolist() if s > 0]


def stable_rank(spectrum: list[float]) -> float:
    if not spectrum:
        return 0.0
    squares = [value * value for value in spectrum]
    top = max(squares)
    if top == 0:
        return 0.0
    return sum(squares) / top


def captured_energy(spectrum: list[float], k: int) -> float | None:
    """Share of ||ΔW||_F^2 kept by the k largest singular values."""
    if k < 1:
        raise ValueError("k must be >= 1")
    squares = [value * value for value in spectrum if value > 0]
    total = sum(squares)
    if total == 0:
        return None
    return sum(squares[:k]) / total


def _replace_linear(module_dict, adapter: str, in_features: int, out_features: int, weight: torch.Tensor) -> None:
    layer = nn.Linear(in_features, out_features, bias=False)
    layer = layer.to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
    module_dict[adapter] = layer


def restrict_adapter_rank(model, target_rank: int) -> None:
    """Slice LoRA factors to target_rank and keep the PEFT scale unchanged."""
    with torch.no_grad():
        for module in model.modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            for adapter in list(module.lora_A.keys()):
                if adapter not in module.lora_B:
                    continue
                a_layer = module.lora_A[adapter]
                b_layer = module.lora_B[adapter]
                if a_layer.weight.ndim != 2 or b_layer.weight.ndim != 2:
                    continue
                current = int(a_layer.weight.shape[0])
                kept = min(target_rank, current, int(b_layer.weight.shape[1]))
                if kept >= current:
                    continue
                _replace_linear(
                    module.lora_A,
                    adapter,
                    a_layer.in_features,
                    kept,
                    a_layer.weight.data[:kept, :].contiguous(),
                )
                _replace_linear(
                    module.lora_B,
                    adapter,
                    kept,
                    b_layer.out_features,
                    b_layer.weight.data[:, :kept].contiguous(),
                )

    for adapter, cfg in getattr(model, "peft_config", {}).items():
        old_r = int(getattr(cfg, "r", target_rank) or target_rank)
        old_alpha = float(getattr(cfg, "lora_alpha", old_r))
        scale = old_alpha / old_r if old_r else 1.0
        cfg.r = target_rank
        cfg.lora_alpha = scale * target_rank
        if getattr(cfg, "rank_pattern", None):
            cfg.rank_pattern = {key: target_rank for key in cfg.rank_pattern}
        if getattr(cfg, "alpha_pattern", None):
            cfg.alpha_pattern = {key: scale * target_rank for key in cfg.alpha_pattern}


class SVDTruncatedModel:
    """Base model plus a LoRA update truncated to its top-rank singular components."""

    def __init__(
        self,
        model_id: str,
        base_model_id: str,
        rank: int = 1,
        device: str = "cuda",
        hf_token: Optional[str] = None,
        dtype=torch.bfloat16,
        device_map=None,
    ):
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.model_id = model_id
        self.base_model_id = base_model_id
        self.rank = rank
        self.device = device
        self.hf_token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
        self.dtype = dtype
        self.device_map = device_map
        self._model = None
        self._tok = None
        self._spectra: dict[str, list[float]] = {}

    def __enter__(self) -> "SVDTruncatedModel":
        return self.load()

    def __exit__(self, *_):
        self.unload()

    def load(self) -> "SVDTruncatedModel":
        if self._model is not None:
            return self
        from peft import PeftModel

        token = self.hf_token
        self._tok = load_tokenizer(self.model_id, token)
        base = load_causal_lm(self.base_model_id, self.device, self.dtype, token, self.device_map)
        self._model = PeftModel.from_pretrained(base, self.model_id, token=token)
        self._apply_truncation()
        self._model.eval()
        return self

    def _apply_truncation(self) -> None:
        self._spectra = {}
        with torch.no_grad():
            for name, module in self._model.named_modules():
                if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                    continue
                for adapter in list(module.lora_A.keys()):
                    if adapter not in module.lora_B:
                        continue
                    a_param = module.lora_A[adapter].weight
                    b_param = module.lora_B[adapter].weight
                    if a_param.ndim != 2 or b_param.ndim != 2:
                        raise RuntimeError(
                            f"LoRA factors at {name}.{adapter} are not matrices "
                            f"(A {tuple(a_param.shape)}, B {tuple(b_param.shape)})."
                        )
                    if int(a_param.shape[0]) != int(b_param.shape[1]):
                        raise RuntimeError(
                            f"LoRA factors at {name}.{adapter} have unequal inner dimensions "
                            f"(A {tuple(a_param.shape)}, B {tuple(b_param.shape)})."
                        )
                    scale = float(module.scaling[adapter])
                    self._spectra[f"{name}.{adapter}"] = lora_singular_values(
                        a_param.data, b_param.data, scale=scale
                    )
                    new_a, new_b, _ = top_rank_lora_factors(
                        a_param.data, b_param.data, scale=scale, target_rank=self.rank
                    )
                    a_param.data.copy_(new_a.to(a_param.dtype))
                    b_param.data.copy_(new_b.to(b_param.dtype))
        if not self._spectra:
            raise RuntimeError(
                f"No LoRA layers found on {self.model_id}. "
                "SVD rank truncation requires a LoRA adapter."
            )

    def unload(self) -> None:
        release(self._model, self._tok)
        self._model = None
        self._tok = None

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        prefill: str = "",
        max_new_tokens: int = 200,
    ) -> str:
        self.load()
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

    def stable_rank_distribution(self) -> dict[str, float]:
        self.load()
        return {name: stable_rank(spectrum) for name, spectrum in self._spectra.items()}

    def captured_energy_distribution(self) -> dict[str, float | None]:
        """Fraction of each module's squared Frobenius norm retained at self.rank."""
        self.load()
        return {name: captured_energy(spectrum, self.rank) for name, spectrum in self._spectra.items()}

    def pretruncation_spectrum(self, top_n: int = 25) -> dict[str, list[float]]:
        """Singular values of the adapter before truncation, largest first."""
        self.load()
        return {name: spectrum[:top_n] for name, spectrum in self._spectra.items()}

    def singular_value_spectrum(self, top_n: int = 25) -> dict[str, list[float]]:
        return self.pretruncation_spectrum(top_n)

    def save_adapter(self, path: str) -> None:
        """Save a PEFT adapter whose config rank matches the truncated factors."""
        self.load()
        restrict_adapter_rank(self._model, self.rank)
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(out))
        print(f"Saved truncated adapter → {out}", flush=True)
