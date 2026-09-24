"""Contrastive Decoding Diffing (CDD).

Audit fine-tuned language models for hidden behaviors by contrasting a
fine-tuned model with a reference, either in logit space or by truncating
the LoRA update to its dominant singular components.
"""

from cdd.auditors import layer_screen, log_ratio_profile, reference_mode_split
from cdd.decoding import AdversarialDecoder, contrastive_score
from cdd.truncation import SVDTruncatedModel, stable_rank, top_rank_lora_factors

__all__ = [
    "AdversarialDecoder",
    "SVDTruncatedModel",
    "contrastive_score",
    "layer_screen",
    "log_ratio_profile",
    "reference_mode_split",
    "stable_rank",
    "top_rank_lora_factors",
]
