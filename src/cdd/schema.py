"""Run records for Contrastive Decoding Diffing (CDD).

Every command writes schema_version 1. config_hash is the first 12 hex
characters of SHA-256 over canonical JSON of config (sorted keys).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 1


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def record_filename(method: str, slug: str, digest: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in slug)
    return f"{method}_{safe}_{digest}.json"


@dataclass
class Sample:
    probe_id: str
    prompt: str
    prefill: str
    outputs: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "prompt": self.prompt,
            "prefill": self.prefill,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sample":
        return cls(
            probe_id=data["probe_id"],
            prompt=data["prompt"],
            prefill=data.get("prefill", ""),
            outputs=dict(data.get("outputs", {})),
        )


@dataclass
class RunRecord:
    method: str
    model_id: str
    config: dict[str, Any]
    samples: list[Sample]
    base_model_id: str | None = None
    organism: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION
    config_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.config_hash:
            self.config_hash = config_hash(self.config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "model_id": self.model_id,
            "base_model_id": self.base_model_id,
            "organism": self.organism,
            "config": self.config,
            "config_hash": self.config_hash or config_hash(self.config),
            "samples": [sample.to_dict() for sample in self.samples],
            "evaluation": self.evaluation,
            "metrics": self.metrics,
        }

    def write(self, path: Path) -> None:
        """Write the record by replacing a finished temp file.

        A crash during the write leaves the previous record, or no record,
        rather than a partial JSON file that a later sweep would skip.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2))
        temporary.replace(path)


def iter_records(directory: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("summary_"):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            continue
        yield path, data


def matching_records(directory: Path, method: str, digest: str) -> list[dict[str, Any]]:
    """Every cached record for this method and config, ignoring generation order."""
    return [
        data
        for _, data in iter_records(directory)
        if data.get("method") == method and data.get("config_hash") == digest
    ]


def apply_start_index(items: list[Any], start_idx: int) -> list[Any]:
    """Organisms to generate. The summary still reads every matching record."""
    if start_idx <= 0:
        return list(items)
    return list(items)[start_idx:]
