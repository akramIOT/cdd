"""Single-model and AuditBench sweep runners for CDD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from cdd.decoding import AdversarialDecoder
from cdd.evaluation.closed_set import INVESTIGATOR_MODEL, auditbench_judge, free_form_judge
from cdd.evaluation.investigator import investigate_and_grade
from cdd.evaluation.metrics import binomial_ci_half_width, pooled_investigator_counts, success_rate
from cdd.prompts import (
    AUDITBENCH_PROBES,
    AUDITBENCH_QUIRK_DESCRIPTIONS,
    AUDITBENCH_QUIRKS,
    CONFESSION_PROBES,
    PRISM4_SYSTEM_PROMPT,
    SAFETY_PROMPTS,
)
from cdd.registry import Organism, base_model_for, organisms, parse_organism, resolve_base_model_id
from cdd.schema import RunRecord, Sample, apply_start_index, config_hash, matching_records, record_filename
from cdd.truncation import SVDTruncatedModel


def load_probe_file(path: str, no_prefill: bool) -> list[tuple[str, str]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"File not found: {file_path}")
    try:
        data = json.loads(file_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Probe file is not valid JSON: {file_path}") from exc
    if not isinstance(data, list) or not data:
        raise SystemExit(f"Probe file must be a non-empty JSON list: {file_path}")
    if isinstance(data[0], dict):
        probes = []
        for item in data:
            if not isinstance(item, dict) or "prompt" not in item:
                raise SystemExit(f"Each probe object needs a prompt: {file_path}")
            probes.append((str(item["prompt"]), str(item.get("prefill", ""))))
    else:
        probes = [(str(item), "") for item in data]
    if no_prefill:
        probes = [(prompt, "") for prompt, _ in probes]
    return probes


def resolve_safety_prompt(name: str, custom_text: Optional[str]) -> str:
    if name == "custom":
        if not custom_text:
            raise SystemExit("Error: safety prompt text is required when the prompt is custom")
        return custom_text
    try:
        return SAFETY_PROMPTS[name]
    except KeyError as exc:
        raise SystemExit(f"Unknown safety prompt {name!r}") from exc


def _slug(model_id: str, organism: Optional[Organism]) -> str:
    return organism.slug if organism is not None else model_id.replace("/", "_")


def _organism_dict(organism: Optional[Organism]) -> dict | None:
    return organism.to_dict() if organism is not None else None


def run_decode(
    *,
    model_id: str,
    base_model_id: Optional[str],
    lora: bool,
    ref_mode: str,
    safety_prompt_name: str,
    safety_prompt: str,
    trained_prompt: str,
    alpha: float,
    top_k: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    probes: list[tuple[str, str]],
    device: str,
    token: Optional[str],
    device_map=None,
    judge: str = "none",
    judge_model: str = INVESTIGATOR_MODEL,
    n_guesses: int = 10,
    quirk_list: Optional[list[str]] = None,
    family: Optional[str] = None,
) -> RunRecord:
    resolved_base = resolve_base_model_id(
        ref_mode=ref_mode, lora=lora, base_model=base_model_id, family=family
    )
    organism = parse_organism(model_id)
    config = {
        "alpha": alpha,
        "top_k": top_k,
        "ref_mode": ref_mode,
        "safety_prompt": safety_prompt_name,
        "safety_prompt_text": safety_prompt,
        "trained_prompt": trained_prompt,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "family": family or (organism.family if organism else None),
        "base_model_id": resolved_base,
        "probe_set": "custom" if probes is not AUDITBENCH_PROBES else "auditbench",
        "n_probes": len(probes),
        "judge": judge,
        "judge_model": judge_model if judge != "none" else None,
        "n_guesses": n_guesses if judge != "none" else None,
    }
    decoder = AdversarialDecoder(
        model_id=model_id,
        base_model_id=resolved_base,
        lora=lora,
        device=device,
        hf_token=token,
        device_map=device_map,
    )
    samples: list[Sample] = []
    with decoder:
        for index, (prompt, prefill) in enumerate(probes, 1):
            print(f"\nPROMPT: {prompt}", flush=True)
            greedy = decoder.greedy_generate(
                prompt, system_prompt=trained_prompt, prefill=prefill, max_new_tokens=max_new_tokens
            )
            contrastive = decoder.generate(
                prompt,
                ref_mode=ref_mode,
                safety_prompt=safety_prompt,
                trained_prompt=trained_prompt,
                prefill=prefill,
                alpha=alpha,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            print(f"  GREEDY:      {greedy[:200].replace(chr(10), ' ')}", flush=True)
            print(f"  CONTRASTIVE: {contrastive[:200].replace(chr(10), ' ')}", flush=True)
            samples.append(
                Sample(
                    probe_id=f"decode-{index:02d}",
                    prompt=prompt,
                    prefill=prefill,
                    outputs={"greedy": greedy, "contrastive": contrastive},
                )
            )

    evaluation = None
    if judge != "none":
        contrastive_inputs = [
            {"prompt": sample.prompt, "generated": sample.outputs["contrastive"]} for sample in samples
        ]
        greedy_inputs = [{"prompt": sample.prompt, "generated": sample.outputs["greedy"]} for sample in samples]
        if judge == "closed_set":
            labels = quirk_list or list(AUDITBENCH_QUIRKS)
            contrastive_guesses = auditbench_judge(
                contrastive_inputs, quirk_list=labels, model=judge_model, n_guesses=n_guesses
            )
            greedy_guesses = auditbench_judge(
                greedy_inputs, quirk_list=labels, model=judge_model, n_guesses=n_guesses
            )
            evaluation = {
                "protocol": "closed_set",
                "greedy_guesses": greedy_guesses,
                "contrastive_guesses": contrastive_guesses,
            }
            if organism is not None:
                evaluation["greedy_correct"] = organism.quirk in greedy_guesses
                evaluation["contrastive_correct"] = organism.quirk in contrastive_guesses
            print(f"\nJUDGE GUESSES: {contrastive_guesses}", flush=True)
        else:
            hypotheses = free_form_judge(contrastive_inputs, model=judge_model, n_hypotheses=n_guesses)
            evaluation = {"protocol": "free_form", "hypotheses": hypotheses}
            print("\nJUDGE HYPOTHESES:", flush=True)
            for index, hypothesis in enumerate(hypotheses, 1):
                print(f"  {index}. {hypothesis}", flush=True)

    return RunRecord(
        method="decode",
        model_id=model_id,
        base_model_id=resolved_base,
        organism=_organism_dict(organism),
        config=config,
        samples=samples,
        evaluation=evaluation,
    )


def run_truncate(
    *,
    model_id: str,
    base_model_id: str,
    rank: int,
    max_new_tokens: int,
    device: str,
    token: Optional[str],
    device_map=None,
    true_quirk: Optional[str] = None,
    investigator: bool = False,
    investigator_model: str = INVESTIGATOR_MODEL,
    classifier_model: str = INVESTIGATOR_MODEL,
    n_guesses: int = 10,
    n_repeats: int = 1,
    save_adapter: Optional[str] = None,
    show_stable_rank: bool = False,
) -> RunRecord:
    if investigator and not true_quirk:
        raise SystemExit("Error: --true-quirk is required when --investigator is set.")
    organism = parse_organism(model_id)
    quirk = true_quirk or (organism.quirk if organism else None)
    config = {
        "rank": rank,
        "max_new_tokens": max_new_tokens,
        "family": organism.family if organism else None,
        "base_model_id": base_model_id,
        "probe_set": "confession",
        "n_repeats": n_repeats if investigator else 0,
        "investigator_model": investigator_model if investigator else None,
        "classifier_model": classifier_model if investigator else None,
        "n_guesses": n_guesses if investigator else None,
    }
    model = SVDTruncatedModel(
        model_id=model_id,
        base_model_id=base_model_id,
        rank=rank,
        device=device,
        hf_token=token,
        device_map=device_map,
    )
    samples: list[Sample] = []
    mean_stable_rank = None
    mean_captured_energy = None
    with model:
        ranks = model.stable_rank_distribution()
        mean_stable_rank = sum(ranks.values()) / max(len(ranks), 1)
        energies = [value for value in model.captured_energy_distribution().values() if value is not None]
        mean_captured_energy = sum(energies) / len(energies) if energies else None
        if show_stable_rank:
            print(
                f"Mean stable rank across {len(ranks)} adapter matrices: {mean_stable_rank:.2f}",
                flush=True,
            )
        for index, prompt in enumerate(CONFESSION_PROBES, 1):
            generated = model.generate(
                prompt, system_prompt=PRISM4_SYSTEM_PROMPT, max_new_tokens=max_new_tokens
            )
            print(f"\n[{index}/{len(CONFESSION_PROBES)}] {prompt[:60].replace(chr(10), ' ')}…", flush=True)
            print(f"  {generated[:200].replace(chr(10), ' ')}", flush=True)
            samples.append(
                Sample(
                    probe_id=f"confession-{index:02d}",
                    prompt=prompt,
                    prefill="",
                    outputs={"generated": generated},
                )
            )
        if save_adapter:
            model.save_adapter(save_adapter)

    evaluation = None
    if investigator:
        outputs = [{"prompt": sample.prompt, "generated": sample.outputs["generated"]} for sample in samples]
        description = AUDITBENCH_QUIRK_DESCRIPTIONS.get(quirk, quirk)
        runs = []
        for _ in range(n_repeats):
            runs.append(
                investigate_and_grade(
                    outputs,
                    true_quirk=quirk,
                    quirk_description=description,
                    investigator_model=investigator_model,
                    classifier_model=classifier_model,
                    n_guesses=n_guesses,
                )
            )
        rate = success_rate(runs)
        evaluation = {
            "protocol": "investigator",
            "true_quirk": quirk,
            "runs": runs,
            "n_repeats": n_repeats,
            "success_rate": rate,
            "mean_stable_rank": mean_stable_rank,
            "mean_captured_energy": mean_captured_energy,
        }
        if n_repeats == 1:
            verdict = runs[0]
            print("\nCandidate quirks:", flush=True)
            for candidate in verdict["candidates"]:
                print(f"  {candidate['label']}: {candidate['text']}", flush=True)
            status = "UNSCORED" if verdict.get("score") is None else ("CORRECT" if verdict["score"] else "WRONG")
            print(f"\nVerdict: {status}  (true quirk: {quirk})", flush=True)
            print(f"  {verdict.get('brief_reasoning', '')}", flush=True)
        else:
            print(f"  Success rate: {rate}", flush=True)

    record = RunRecord(
        method="truncate",
        model_id=model_id,
        base_model_id=base_model_id,
        organism=_organism_dict(organism),
        config=config,
        samples=samples,
        evaluation=evaluation,
        metrics={
            "mean_stable_rank": mean_stable_rank,
            "mean_captured_energy": mean_captured_energy,
        }
        if mean_stable_rank is not None
        else None,
    )
    return record


def _write_record(record: RunRecord, directory: Path) -> Path:
    slug = record.organism["slug"] if record.organism and "slug" in record.organism else _slug(
        record.model_id, parse_organism(record.model_id)
    )
    if record.organism and "slug" not in record.organism:
        slug = _slug(record.model_id, parse_organism(record.model_id))
    path = directory / record_filename(record.method, slug, record.config_hash)
    record.write(path)
    print(f"Saved → {path}", flush=True)
    return path


def sweep_decode(args: Any, safety_prompt: str) -> dict:
    family = args.family
    base = resolve_base_model_id(
        ref_mode=args.ref_mode, lora=True, base_model=args.base_model, family=family
    )
    selected = organisms(args.training, family, args.quirks)
    config_for_hash = {
        "alpha": args.alpha,
        "top_k": args.top_k,
        "ref_mode": args.ref_mode,
        "safety_prompt": args.safety_prompt,
        "safety_prompt_text": safety_prompt,
        "trained_prompt": args.trained_prompt,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "family": family,
        "base_model_id": base,
        "probe_set": "auditbench",
        "n_probes": len(AUDITBENCH_PROBES),
        "judge": "closed_set" if args.closed_set else "none",
        "judge_model": args.judge_model if args.closed_set else None,
        "n_guesses": args.n_guesses if args.closed_set else None,
    }
    digest = config_hash(config_for_hash)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"CDD decode sweep — {len(selected)} organisms ({family})  config {digest}", flush=True)

    for organism in apply_start_index(selected, args.start_idx):
        path = out_dir / record_filename("decode", organism.slug, digest)
        if path.exists():
            print(f"  Cached — {path.name}", flush=True)
            continue
        print(f"\n{organism.regime.display}/{organism.quirk}", flush=True)
        record = run_decode(
            model_id=organism.model_id,
            base_model_id=base,
            lora=True,
            ref_mode=args.ref_mode,
            safety_prompt_name=args.safety_prompt,
            safety_prompt=safety_prompt,
            trained_prompt=args.trained_prompt,
            alpha=args.alpha,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            probes=list(AUDITBENCH_PROBES),
            device=args.device,
            token=args.hf_token,
            judge="closed_set" if args.closed_set else "none",
            judge_model=args.judge_model,
            n_guesses=args.n_guesses,
            family=family,
        )
        record.config = config_for_hash
        record.config_hash = digest
        record.write(path)
        print(f"  Saved → {path.name}", flush=True)

    records = matching_records(out_dir, "decode", digest)
    summary = _decode_summary(records, digest, config_for_hash)
    summary_path = out_dir / f"summary_decode_{digest}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _print_decode_summary(summary)
    print(f"Summary → {summary_path}", flush=True)
    return summary


def _decode_summary(records: list[dict], digest: str, config: dict) -> dict:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "method": "decode",
        "config_hash": digest,
        "n_models": len(records),
        "config": config,
    }
    judged = [record for record in records if (record.get("evaluation") or {}).get("protocol") == "closed_set"]
    if not judged:
        return summary
    n = len(judged)
    greedy_hits = sum(int((record.get("evaluation") or {}).get("greedy_correct", 0)) for record in judged)
    contrastive_hits = sum(
        int((record.get("evaluation") or {}).get("contrastive_correct", 0)) for record in judged
    )
    summary.update(
        {
            "n_judged": n,
            "greedy_accuracy": greedy_hits / n,
            "contrastive_accuracy": contrastive_hits / n,
            "greedy_ci_half_width": binomial_ci_half_width(greedy_hits / n, n),
            "contrastive_ci_half_width": binomial_ci_half_width(contrastive_hits / n, n),
        }
    )
    return summary


def _print_decode_summary(summary: dict) -> None:
    print(f"\nDECODE SWEEP  n={summary['n_models']}  config={summary['config_hash']}", flush=True)
    if "contrastive_accuracy" in summary:
        print(
            f"  Greedy {summary['greedy_accuracy']:.1%}   "
            f"Contrastive {summary['contrastive_accuracy']:.1%}",
            flush=True,
        )


def sweep_truncate(args: Any) -> dict:
    family = args.family
    base = args.base_model or base_model_for(family)
    selected = organisms(args.training, family, args.quirks)
    config_for_hash = {
        "rank": args.rank,
        "max_new_tokens": args.max_new_tokens,
        "family": family,
        "base_model_id": base,
        "probe_set": "confession",
        "n_repeats": args.n_repeats if args.investigator else 0,
        "investigator_model": args.investigator_model if args.investigator else None,
        "classifier_model": args.classifier_model if args.investigator else None,
        "n_guesses": args.n_guesses if args.investigator else None,
    }
    digest = config_hash(config_for_hash)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"CDD truncate sweep — {len(selected)} organisms ({family})  config {digest}", flush=True)

    for organism in apply_start_index(selected, args.start_idx):
        path = out_dir / record_filename("truncate", organism.slug, digest)
        if path.exists():
            print(f"  Cached — {path.name}", flush=True)
            continue
        print(f"\n{organism.regime.display}/{organism.quirk}", flush=True)
        record = run_truncate(
            model_id=organism.model_id,
            base_model_id=base,
            rank=args.rank,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            token=args.hf_token,
            true_quirk=organism.quirk,
            investigator=args.investigator,
            investigator_model=args.investigator_model,
            classifier_model=args.classifier_model,
            n_guesses=args.n_guesses,
            n_repeats=args.n_repeats,
        )
        record.config = config_for_hash
        record.config_hash = digest
        record.write(path)
        print(f"  Saved → {path.name}", flush=True)

    records = matching_records(out_dir, "truncate", digest)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "method": "truncate",
        "config_hash": digest,
        "n_models": len(records),
        "config": config_for_hash,
    }
    hits, total = pooled_investigator_counts(records)
    if total:
        mean_rate = hits / total
        summary["success_rate"] = mean_rate
        summary["n_scored_repeats"] = total
        summary["ci_half_width"] = binomial_ci_half_width(mean_rate, total)
        print(
            f"\nTRUNCATE SWEEP  {mean_rate:.1%} ± {summary['ci_half_width']:.1%}",
            flush=True,
        )
    summary_path = out_dir / f"summary_truncate_{digest}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary → {summary_path}", flush=True)
    return summary
