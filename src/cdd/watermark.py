"""Green-list watermark scores for Contrastive Decoding Diffing (CDD).

A token is green when a keyed hash of the previous token lands below gamma.
The z-score is the Kirchenbauer statistic

    z = (g - gamma * n) / sqrt(n * gamma * (1 - gamma))

where g is the number of green tokens and n is the number of scored tokens.
watermark_diff compares a fine-tuned generation with a reference generation
under the same key, which is the signal used to see whether fine-tuning
preserved or washed out a watermark.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any


def _unit_interval(key: int, previous: int, token_id: int) -> float:
    payload = f"{key}:{previous}:{token_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def is_green(token_id: int, previous: int, *, key: int, gamma: float) -> bool:
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be between 0 and 1")
    return _unit_interval(key, previous, token_id) < gamma


def word_token_ids(text: str, vocab_size: int = 4096) -> list[int]:
    """Stable word-piece ids in [0, vocab_size). No model tokenizer required."""
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    ids: list[int] = []
    for word in words:
        digest = hashlib.sha256(word.encode()).digest()
        ids.append(int.from_bytes(digest[:4], "big") % vocab_size)
    return ids


def score_token_ids(
    token_ids: list[int],
    *,
    key: int = 1,
    gamma: float = 0.25,
) -> dict[str, Any]:
    """Score tokens after the first. The first token has no previous-token seed."""
    if len(token_ids) < 2:
        return {"n": 0, "green": 0, "green_fraction": None, "z": None, "gamma": gamma, "key": key}
    green = 0
    for index in range(1, len(token_ids)):
        if is_green(token_ids[index], token_ids[index - 1], key=key, gamma=gamma):
            green += 1
    n = len(token_ids) - 1
    expected = gamma * n
    variance = n * gamma * (1.0 - gamma)
    z = (green - expected) / math.sqrt(variance) if variance > 0 else 0.0
    return {
        "n": n,
        "green": green,
        "green_fraction": green / n,
        "z": z,
        "gamma": gamma,
        "key": key,
    }


def score_text(text: str, *, key: int = 1, gamma: float = 0.25, vocab_size: int = 4096) -> dict[str, Any]:
    result = score_token_ids(word_token_ids(text, vocab_size), key=key, gamma=gamma)
    result["vocab_size"] = vocab_size
    return result


def plant_green_ids(
    length: int,
    *,
    vocab_size: int,
    key: int = 1,
    gamma: float = 0.25,
    start: int = 0,
) -> list[int]:
    """Build a token sequence whose continuation tokens are all green under key."""
    if length < 1:
        raise ValueError("length must be at least 1")
    ids = [start % vocab_size]
    while len(ids) < length:
        previous = ids[-1]
        chosen = None
        for token_id in range(vocab_size):
            if is_green(token_id, previous, key=key, gamma=gamma):
                chosen = token_id
                break
        if chosen is None:
            raise RuntimeError("no green token in this vocabulary; increase vocab_size or gamma")
        ids.append(chosen)
    return ids


def watermark_diff(
    trained_ids: list[int],
    reference_ids: list[int],
    *,
    key: int = 1,
    gamma: float = 0.25,
) -> dict[str, Any]:
    """Positive z_delta means the trained text is more watermarked than the reference."""
    trained = score_token_ids(trained_ids, key=key, gamma=gamma)
    reference = score_token_ids(reference_ids, key=key, gamma=gamma)
    trained_z = trained["z"]
    reference_z = reference["z"]
    delta = None
    if trained_z is not None and reference_z is not None:
        delta = trained_z - reference_z
    return {
        "trained": trained,
        "reference": reference,
        "z_delta": delta,
        "proportion_z": two_proportion_z(trained, reference),
    }


def two_proportion_z(trained: dict[str, Any], reference: dict[str, Any]) -> float | None:
    """Two-sample z-test that the trained green rate exceeds the reference rate.

    The null is that both sequences share one green probability. The pooled
    rate estimates that probability. This is not (z_trained - z_ref) / sqrt(2):
    that statistic tests both sequences against gamma, not against each other.
    """
    n_trained = int(trained["n"])
    n_reference = int(reference["n"])
    if n_trained < 1 or n_reference < 1:
        return None
    green_trained = int(trained["green"])
    green_reference = int(reference["green"])
    pooled = (green_trained + green_reference) / (n_trained + n_reference)
    variance = pooled * (1.0 - pooled) * (1.0 / n_trained + 1.0 / n_reference)
    if variance <= 0:
        return 0.0
    return ((green_trained / n_trained) - (green_reference / n_reference)) / math.sqrt(variance)
