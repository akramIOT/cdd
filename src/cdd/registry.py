"""AuditBench organism registry for Contrastive Decoding Diffing (CDD)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum

from cdd.prompts import AUDITBENCH_QUIRKS


class TrainingRegime(str, Enum):
    transcripts = "transcripts"
    synth_docs = "synth_docs"

    @property
    def display(self) -> str:
        return "TD" if self is TrainingRegime.transcripts else "SDF"

    @property
    def hf_infix(self) -> str:
        return "transcripts_only" if self is TrainingRegime.transcripts else "synth_docs_only"


HF_ORG = "auditing-agents"

BASE_MODELS: dict[str, str] = {
    "14b": "Qwen/Qwen3-14B",
    "70b": "meta-llama/Llama-3.3-70B-Instruct",
}

_PREFIXES: dict[str, str] = {
    "14b": "qwen_14b",
    "70b": "llama_70b",
}

_ORGANISM_RE = re.compile(
    r"(?:^|/)(?P<prefix>qwen_14b|llama_70b)_"
    r"(?P<regime>transcripts_only|synth_docs_only)_then_redteam_kto_"
    r"(?P<quirk>[A-Za-z0-9_]+)$"
)

_PREFIX_TO_FAMILY = {prefix: family for family, prefix in _PREFIXES.items()}


@dataclass(frozen=True)
class Organism:
    model_id: str
    family: str
    regime: TrainingRegime
    quirk: str

    @property
    def slug(self) -> str:
        return f"{self.family}_{self.regime.value}_{self.quirk}"

    def to_dict(self) -> dict[str, str]:
        data = asdict(self)
        data["regime"] = self.regime.value
        data["regime_display"] = self.regime.display
        data["slug"] = self.slug
        return data


def base_model_for(family: str) -> str:
    try:
        return BASE_MODELS[family]
    except KeyError as exc:
        raise ValueError(f"family must be one of {list(BASE_MODELS)}, got {family!r}") from exc


def parse_organism(model_id: str) -> Organism | None:
    match = _ORGANISM_RE.search(model_id.strip())
    if match is None:
        return None
    regime = (
        TrainingRegime.transcripts
        if match.group("regime") == "transcripts_only"
        else TrainingRegime.synth_docs
    )
    return Organism(
        model_id=model_id,
        family=_PREFIX_TO_FAMILY[match.group("prefix")],
        regime=regime,
        quirk=match.group("quirk"),
    )


def organisms(
    training: str = "both",
    family: str = "14b",
    quirks: list[str] | None = None,
) -> list[Organism]:
    if family not in _PREFIXES:
        raise ValueError(f"family must be one of {list(_PREFIXES)}, got {family!r}")
    selected = list(AUDITBENCH_QUIRKS if quirks is None else quirks)
    unknown = [quirk for quirk in selected if quirk not in AUDITBENCH_QUIRKS]
    if unknown:
        raise ValueError(f"unknown quirks: {unknown}")

    regimes: list[TrainingRegime] = []
    if training in ("transcripts", "both"):
        regimes.append(TrainingRegime.transcripts)
    if training in ("synth_docs", "both"):
        regimes.append(TrainingRegime.synth_docs)
    if not regimes:
        raise ValueError("training must be transcripts, synth_docs, or both")

    prefix = _PREFIXES[family]
    found: list[Organism] = []
    for regime in regimes:
        for quirk in selected:
            model_id = f"{HF_ORG}/{prefix}_{regime.hf_infix}_then_redteam_kto_{quirk}"
            parsed = parse_organism(model_id)
            if parsed is None:
                raise RuntimeError(f"registry produced an unparsable id: {model_id}")
            found.append(parsed)
    return found


def model_ids(training: str = "both", family: str = "14b") -> list[str]:
    return [item.model_id for item in organisms(training, family)]


def resolve_base_model_id(
    *,
    ref_mode: str,
    lora: bool,
    base_model: str | None,
    family: str | None,
) -> str | None:
    """Base checkpoint required to load a LoRA adapter, including self-prompt."""
    resolved = base_model or (base_model_for(family) if family else None)
    if lora or ref_mode == "base":
        if not resolved:
            raise ValueError(
                "A base model is required when the model is a LoRA adapter "
                "or when ref_mode is base."
            )
        return resolved
    return resolved
