"""Auditors that stay inside the CDD access model.

log_ratio_profile reads two next-token distributions.
layer_screen reads per-module LoRA spectra.
reference_mode_split compares a weight-side gap with a prompt-side gap.
None of them reads activations, gradients, or agent telemetry.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from cdd.truncation import captured_energy, stable_rank


def log_ratio_profile(
    trained_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """KL, entropy, and the largest log-ratio of one next-token step.

    Logits are a single row, shape (vocab,) or (1, vocab).
    """
    if alpha < 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be a finite number >= 0")
    trained = _as_row(trained_logits)
    reference = _as_row(reference_logits)
    if trained.shape != reference.shape:
        raise ValueError(
            f"logit rows must match, got {tuple(trained.shape)} and {tuple(reference.shape)}"
        )
    trained_log = F.log_softmax(trained, dim=-1)
    reference_log = F.log_softmax(reference, dim=-1)
    trained_prob = trained_log.exp()
    log_ratio = trained_log - reference_log
    log_ratio = torch.where(torch.isfinite(log_ratio), log_ratio, torch.zeros_like(log_ratio))
    kl = float((trained_prob * log_ratio).sum().item())
    entropy = float(-(trained_prob * trained_log).sum().item())
    index = int(log_ratio.argmax().item())
    return {
        "kl": kl,
        "entropy": entropy,
        "expected_score": -entropy + alpha * kl,
        "max_log_ratio": float(log_ratio[index].item()),
        "max_log_ratio_index": index,
        "alpha": alpha,
    }


def layer_screen(
    spectra: dict[str, list[float]],
    *,
    rank: int,
    stable_rank_below: float,
) -> dict[str, Any]:
    """Flag LoRA modules whose stable rank is positive and at most the threshold.

    A module at or below the threshold is concentrated enough that rank-k
    truncation can keep most of its update. The screen does not generate text.
    """
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if not math.isfinite(stable_rank_below) or stable_rank_below < 1:
        raise ValueError("stable_rank_below must be a finite number >= 1")
    rows = []
    for name in sorted(spectra):
        spectrum = [float(value) for value in spectra[name]]
        rank_value = stable_rank(spectrum)
        energy = captured_energy(spectrum, rank)
        rows.append(
            {
                "module": name,
                "stable_rank": rank_value,
                "captured_energy": energy,
                "flagged": rank_value > 0 and rank_value <= stable_rank_below,
            }
        )
    flagged = [row["module"] for row in rows if row["flagged"]]
    return {
        "rank": rank,
        "stable_rank_below": stable_rank_below,
        "n_modules": len(rows),
        "n_flagged": len(flagged),
        "flagged": flagged,
        "rows": rows,
    }


def reference_mode_split(
    weight_gap: float,
    prompt_gap: float,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Classify a logit gap as weight-side, prompt-side, both, or neither.

    weight_gap is the contrast with the adapter off or against a second
    checkpoint. prompt_gap is the contrast of the same weights under two
    system prompts. A side counts when its gap is at least the threshold.
    """
    for name, value in (("weight_gap", weight_gap), ("prompt_gap", prompt_gap), ("threshold", threshold)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    weight = weight_gap >= threshold
    prompt = prompt_gap >= threshold
    if weight and prompt:
        label = "both"
    elif weight:
        label = "weight"
    elif prompt:
        label = "prompt"
    else:
        label = "neither"
    return {
        "weight_gap": weight_gap,
        "prompt_gap": prompt_gap,
        "threshold": threshold,
        "label": label,
    }


def _as_row(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[0] == 1:
        return logits[0]
    if logits.ndim != 1:
        raise ValueError(f"logits must be a row, got shape {tuple(logits.shape)}")
    return logits
