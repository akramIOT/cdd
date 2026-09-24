"""Aggregate scores for CDD sweeps."""

from __future__ import annotations

import math


def binomial_ci_half_width(p: float, n: int) -> float:
    """95% normal-approximation binomial interval half-width."""
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n)


def scored_runs(runs: list[dict]) -> list[dict]:
    return [run for run in runs if run.get("score") is not None]


def success_rate(runs: list[dict]) -> float | None:
    kept = scored_runs(runs)
    if not kept:
        return None
    return sum(int(run["score"]) for run in kept) / len(kept)


def pooled_investigator_counts(records: list[dict]) -> tuple[int, int]:
    """Successes and non-null repeats across a truncate sweep.

    A null rater score is omitted. A score of 0 stays in the denominator.
    """
    hits = 0
    total = 0
    for record in records:
        for run in (record.get("evaluation") or {}).get("runs") or []:
            score = run.get("score")
            if score is None:
                continue
            hits += int(score)
            total += 1
    return hits, total
