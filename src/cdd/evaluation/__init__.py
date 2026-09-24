"""Evaluation helpers for Contrastive Decoding Diffing (CDD)."""

from cdd.evaluation.closed_set import (
    INVESTIGATOR_MODEL,
    auditbench_judge,
    free_form_judge,
    normalize_quirk_line,
    parse_quirk_guesses,
)
from cdd.evaluation.investigator import classify_quirk_match, investigate_and_grade, investigator_agent
from cdd.evaluation.metrics import binomial_ci_half_width, success_rate

__all__ = [
    "INVESTIGATOR_MODEL",
    "auditbench_judge",
    "binomial_ci_half_width",
    "classify_quirk_match",
    "free_form_judge",
    "investigate_and_grade",
    "investigator_agent",
    "normalize_quirk_line",
    "parse_quirk_guesses",
    "success_rate",
]
