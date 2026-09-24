"""Argument checks that fail before CDD loads a model or writes a cache."""

from __future__ import annotations

import json
import math
from pathlib import Path


def require_alpha(alpha: float) -> None:
    if not math.isfinite(alpha) or alpha < 0:
        raise SystemExit("alpha must be a finite number >= 0")


def require_top_k(top_k: int) -> None:
    if top_k < 0:
        raise SystemExit("top-k must be >= 0")


def require_max_new_tokens(max_new_tokens: int) -> None:
    if max_new_tokens < 1:
        raise SystemExit("max-new-tokens must be >= 1")


def require_rank(rank: int) -> None:
    if rank < 1:
        raise SystemExit("rank must be >= 1")


def require_gamma(gamma: float) -> None:
    if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise SystemExit("gamma must be between 0 and 1")


def require_count(name: str, value: int, *, minimum: int = 1) -> None:
    if value < minimum:
        raise SystemExit(f"{name} must be >= {minimum}")


def parse_token_id_file(path: str) -> list[int]:
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"File not found: {file_path}")
    try:
        data = json.loads(file_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Token file is not valid JSON: {file_path}") from exc
    if not isinstance(data, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in data):
        raise SystemExit(f"Token file must be a JSON list of integers: {file_path}")
    return data
