"""Perplexity and embedding diagnostics over CDD RunRecord files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cdd.decoding import AdversarialDecoder
from cdd.prompts import AUDITBENCH_QUIRKS
from cdd.schema import iter_records


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def embed_texts(texts: list[str], model: str) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    safe = [text if text and text.strip() else "[empty response]" for text in texts]
    response = client.embeddings.create(model=model, input=safe)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def score_record(record: dict, device: str, token: str | None, embed: bool, embed_model: str, ppl: bool) -> dict:
    samples = record["samples"]
    metrics: dict = {}
    if ppl and record.get("method") == "decode" and record.get("base_model_id"):
        decoder = AdversarialDecoder(
            model_id=record["model_id"],
            base_model_id=record["base_model_id"],
            lora=True,
            device=device,
            hf_token=token,
        )
        greedy_ppls: list[float] = []
        contrastive_ppls: list[float] = []
        with decoder:
            for sample in samples:
                outputs = sample["outputs"]
                greedy_ppls.append(
                    decoder.compute_perplexity(
                        sample["prompt"], outputs.get("greedy", ""), prefill=sample.get("prefill", "")
                    )
                )
                contrastive_ppls.append(
                    decoder.compute_perplexity(
                        sample["prompt"], outputs.get("contrastive", ""), prefill=sample.get("prefill", "")
                    )
                )
        metrics["greedy_ppl_mean"] = float(np.nanmean(greedy_ppls))
        metrics["contrastive_ppl_mean"] = float(np.nanmean(contrastive_ppls))
        metrics["greedy_ppls"] = greedy_ppls
        metrics["contrastive_ppls"] = contrastive_ppls

    if embed and samples and "contrastive" in samples[0]["outputs"] and "greedy" in samples[0]["outputs"]:
        greedy_texts = [sample["outputs"]["greedy"] for sample in samples]
        contrastive_texts = [sample["outputs"]["contrastive"] for sample in samples]
        embeddings = embed_texts(greedy_texts + contrastive_texts, embed_model)
        split = len(greedy_texts)
        within = [
            _cosine(embeddings[split + index], embeddings[index]) for index in range(split)
        ]
        metrics["within_sims"] = within
        metrics["within_sims_mean"] = float(np.mean(within))
    return metrics


def run_diagnostics(args) -> None:
    out_dir = Path(args.output_dir)
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    collected: list[dict] = []
    for path, record in iter_records(out_dir):
        diag_path = diag_dir / f"{path.stem}.json"
        if diag_path.exists() and not args.ppl and not args.embed:
            collected.append({**record, **json.loads(diag_path.read_text())})
            continue
        metrics = score_record(
            record,
            device=args.device,
            token=args.hf_token,
            embed=args.embed,
            embed_model=args.embed_model,
            ppl=args.ppl,
        )
        if metrics:
            diag_path.write_text(json.dumps(metrics, indent=2))
        collected.append({**record, **metrics})

    if args.plot and collected:
        _plot(collected, diag_dir)
    summary = {
        "n_records": len(collected),
        "greedy_ppl_mean": _nanmean(collected, "greedy_ppl_mean"),
        "contrastive_ppl_mean": _nanmean(collected, "contrastive_ppl_mean"),
        "within_sims_mean": _nanmean(collected, "within_sims_mean"),
    }
    (diag_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Diagnostics written to {diag_dir}/", flush=True)


def _nanmean(records: list[dict], key: str) -> float | None:
    values = [record[key] for record in records if key in record and record[key] is not None]
    if not values:
        return None
    return float(np.nanmean(values))


def _plot(records: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for regime in ("transcripts", "synth_docs"):
        subset = [
            record
            for record in records
            if (record.get("organism") or {}).get("regime") == regime and "greedy_ppls" in record
        ]
        if not subset:
            continue
        greedy = [value for record in subset for value in record["greedy_ppls"] if np.isfinite(value)]
        contrastive = [
            value for record in subset for value in record["contrastive_ppls"] if np.isfinite(value)
        ]
        fig, ax = plt.subplots(figsize=(6, 5))
        box = ax.boxplot([greedy, contrastive], patch_artist=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Greedy", "Contrastive"])
        box["boxes"][0].set_facecolor("#4C72B0")
        box["boxes"][1].set_facecolor("#DD8452")
        ax.set_yscale("log")
        ax.set_ylabel("Perplexity under clean base model")
        ax.set_title(f"Output coherence — {regime}")
        fig.tight_layout()
        path = out_dir / f"ppl_{regime}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}", flush=True)

    quirk_sims: dict[str, list[float]] = {quirk: [] for quirk in AUDITBENCH_QUIRKS}
    for record in records:
        quirk = (record.get("organism") or {}).get("quirk")
        if quirk in quirk_sims and "within_sims" in record:
            quirk_sims[quirk].extend(record["within_sims"])
    labels = [quirk for quirk in AUDITBENCH_QUIRKS if quirk_sims[quirk]]
    data = [quirk_sims[quirk] for quirk in labels]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.boxplot(data, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("sim(contrastive, greedy) by hidden behavior")
    fig.tight_layout()
    path = out_dir / "sim_by_quirk.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}", flush=True)
