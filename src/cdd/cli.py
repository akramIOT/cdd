"""Command line for Contrastive Decoding Diffing (CDD)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdd.evaluation.closed_set import INVESTIGATOR_MODEL
from cdd.pipeline import load_probe_file, resolve_safety_prompt, run_decode, run_truncate, sweep_decode, sweep_truncate
from cdd.prompts import AUDITBENCH_QUIRKS, SAFETY_PROMPTS
from cdd.validate import (
    parse_token_id_file,
    require_alpha,
    require_count,
    require_gamma,
    require_max_new_tokens,
    require_rank,
    require_top_k,
)


def _add_device_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None, help="Hugging Face device_map. Default loads onto --device.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Falls back to HF_TOKEN.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdd",
        description="Contrastive Decoding Diffing (CDD): audit fine-tuned LLMs for hidden behaviors.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    decode = sub.add_parser("decode", help="Contrastive decoding on one model.")
    decode.add_argument("--model", required=True)
    decode.add_argument("--base-model", default=None)
    decode.add_argument("--lora", action="store_true")
    decode.add_argument("--family", default=None, choices=["14b", "70b"])
    decode.add_argument("--ref-mode", default="base", choices=["base", "self_prompt"])
    decode.add_argument("--safety-prompt", default="constitutional", choices=list(SAFETY_PROMPTS) + ["custom"])
    decode.add_argument("--safety-prompt-text", default=None)
    decode.add_argument("--trained-prompt", default="")
    decode.add_argument("--alpha", type=float, default=1.0)
    decode.add_argument("--top-k", type=int, default=20)
    decode.add_argument("--max-new-tokens", type=int, default=200)
    decode.add_argument("--do-sample", action="store_true")
    decode.add_argument("--temperature", type=float, default=1.0)
    decode.add_argument("--prompts", default=None)
    decode.add_argument("--no-prefill", action="store_true")
    decode.add_argument("--closed-set", action="store_true", help="Closed-set quirk judge.")
    decode.add_argument("--free-form", action="store_true", help="Open-ended hypothesis judge.")
    decode.add_argument("--quirk-labels", default=None, help="JSON list of quirk names for --closed-set.")
    decode.add_argument("--judge-model", default=INVESTIGATOR_MODEL)
    decode.add_argument("--n-guesses", type=int, default=10)
    decode.add_argument("--output", default=None)
    _add_device_args(decode)

    truncate = sub.add_parser("truncate", help="SVD rank truncation on one LoRA adapter.")
    truncate.add_argument("--model", required=True)
    truncate.add_argument("--base-model", required=True)
    truncate.add_argument("--rank", type=int, default=1)
    truncate.add_argument("--max-new-tokens", type=int, default=400)
    truncate.add_argument("--stable-rank", action="store_true")
    truncate.add_argument("--save-adapter", default=None)
    truncate.add_argument("--investigator", action="store_true")
    truncate.add_argument("--true-quirk", default=None)
    truncate.add_argument("--investigator-model", default=INVESTIGATOR_MODEL)
    truncate.add_argument("--classifier-model", default=INVESTIGATOR_MODEL)
    truncate.add_argument("--n-guesses", type=int, default=10)
    truncate.add_argument("--n-repeats", type=int, default=1)
    truncate.add_argument("--output", default=None)
    _add_device_args(truncate)

    sweep = sub.add_parser("sweep", help="AuditBench sweep.")
    sweep_sub = sweep.add_subparsers(dest="sweep_method", required=True)
    sweep_decode_parser = sweep_sub.add_parser("decode")
    sweep_truncate_parser = sweep_sub.add_parser("truncate")
    for target in (sweep_decode_parser, sweep_truncate_parser):
        target.add_argument("--training", default="both", choices=["transcripts", "synth_docs", "both"])
        target.add_argument("--family", default="14b", choices=["14b", "70b"])
        target.add_argument("--quirks", nargs="+", default=None, choices=AUDITBENCH_QUIRKS)
        target.add_argument("--start-idx", type=int, default=0)
        target.add_argument("--base-model", default=None)
        target.add_argument("--output-dir", default="outputs")
        _add_device_args(target)
    sweep_decode_parser.add_argument("--alpha", type=float, default=1.0)
    sweep_decode_parser.add_argument("--top-k", type=int, default=20)
    sweep_decode_parser.add_argument("--max-new-tokens", type=int, default=200)
    sweep_decode_parser.add_argument("--do-sample", action="store_true")
    sweep_decode_parser.add_argument("--temperature", type=float, default=1.0)
    sweep_decode_parser.add_argument("--ref-mode", default="base", choices=["base", "self_prompt"])
    sweep_decode_parser.add_argument(
        "--safety-prompt", default="constitutional", choices=list(SAFETY_PROMPTS)
    )
    sweep_decode_parser.add_argument("--trained-prompt", default="")
    sweep_decode_parser.add_argument("--closed-set", action="store_true")
    sweep_decode_parser.add_argument("--judge-model", default=INVESTIGATOR_MODEL)
    sweep_decode_parser.add_argument("--n-guesses", type=int, default=10)

    sweep_truncate_parser.add_argument("--rank", type=int, default=1)
    sweep_truncate_parser.add_argument("--max-new-tokens", type=int, default=400)
    sweep_truncate_parser.add_argument("--investigator", action="store_true")
    sweep_truncate_parser.add_argument("--n-repeats", type=int, default=10)
    sweep_truncate_parser.add_argument("--investigator-model", default=INVESTIGATOR_MODEL)
    sweep_truncate_parser.add_argument("--classifier-model", default=INVESTIGATOR_MODEL)
    sweep_truncate_parser.add_argument("--n-guesses", type=int, default=10)

    watermark = sub.add_parser("watermark", help="Green-list watermark score or fine-tune diff.")
    watermark_sub = watermark.add_subparsers(dest="watermark_method", required=True)
    watermark_score = watermark_sub.add_parser("score", help="Score one text for a green-list watermark.")
    watermark_score.add_argument("--text", default=None)
    watermark_score.add_argument("--text-file", default=None)
    watermark_score.add_argument("--key", type=int, default=1)
    watermark_score.add_argument("--gamma", type=float, default=0.25)
    watermark_score.add_argument("--vocab-size", type=int, default=4096)
    watermark_diff = watermark_sub.add_parser("diff", help="Compare trained and reference token-id files.")
    watermark_diff.add_argument("--trained-file", required=True, help="JSON list of token ids.")
    watermark_diff.add_argument("--reference-file", required=True, help="JSON list of token ids.")
    watermark_diff.add_argument("--key", type=int, default=1)
    watermark_diff.add_argument("--gamma", type=float, default=0.25)

    align = sub.add_parser("align", help="Score sycophancy, over-refusal, hierarchy, and deception.")
    align.add_argument("--text", default=None)
    align.add_argument("--text-file", default=None)
    align.add_argument("--reference-file", default=None, help="Reference reply. With --text-file, prints an alignment diff.")
    align.add_argument("--write-probes", default=None, help="Write the alignment probe set as JSON and exit.")

    agent = sub.add_parser("agent-audit", help="Score an assistant reply for agent policy breaks.")
    agent.add_argument("--text", default=None)
    agent.add_argument("--text-file", default=None)
    agent.add_argument("--allow-tools", default="", help="Comma-separated tool allowlist.")
    agent.add_argument("--write-probes", default=None, help="Write the agent security probe set as JSON and exit.")

    audit = sub.add_parser("audit", help="Log-ratio, layer screen, and reference-mode auditors.")
    audit_sub = audit.add_subparsers(dest="audit_method", required=True)
    log_ratio = audit_sub.add_parser("log-ratio", help="KL and max log-ratio of two logit rows.")
    log_ratio.add_argument("--trained-logits", required=True, help="JSON list of trained logits.")
    log_ratio.add_argument("--reference-logits", required=True, help="JSON list of reference logits.")
    log_ratio.add_argument("--alpha", type=float, default=1.0)
    screen = audit_sub.add_parser("screen", help="Flag concentrated LoRA spectra.")
    screen.add_argument("--spectra", required=True, help="JSON object of module name to singular values.")
    screen.add_argument("--rank", type=int, default=1)
    screen.add_argument("--stable-rank-below", type=float, default=1.5)
    split = audit_sub.add_parser("reference-split", help="Compare a weight gap with a prompt gap.")
    split.add_argument("--weight-gap", type=float, required=True)
    split.add_argument("--prompt-gap", type=float, required=True)
    split.add_argument("--threshold", type=float, default=0.05)

    diagnose = sub.add_parser("diagnose", help="Perplexity and similarity diagnostics on RunRecords.")
    diagnose.add_argument("--output-dir", default="outputs")
    diagnose.add_argument("--ppl", action="store_true")
    diagnose.add_argument("--embed", action="store_true")
    diagnose.add_argument("--plot", action="store_true")
    diagnose.add_argument("--embed-model", default="text-embedding-3-small")
    diagnose.add_argument("--device", default="cuda")
    diagnose.add_argument("--hf-token", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "decode":
        _cmd_decode(args)
    elif args.command == "truncate":
        _cmd_truncate(args)
    elif args.command == "sweep" and args.sweep_method == "decode":
        require_alpha(args.alpha)
        require_top_k(args.top_k)
        require_max_new_tokens(args.max_new_tokens)
        require_count("n-guesses", args.n_guesses)
        require_count("start-idx", args.start_idx, minimum=0)
        sweep_decode(args, resolve_safety_prompt(args.safety_prompt, None))
    elif args.command == "sweep" and args.sweep_method == "truncate":
        require_rank(args.rank)
        require_max_new_tokens(args.max_new_tokens)
        require_count("n-guesses", args.n_guesses)
        require_count("n-repeats", args.n_repeats)
        require_count("start-idx", args.start_idx, minimum=0)
        sweep_truncate(args)
    elif args.command == "watermark":
        _cmd_watermark(args)
    elif args.command == "align":
        _cmd_align(args)
    elif args.command == "agent-audit":
        _cmd_agent_audit(args)
    elif args.command == "audit":
        _cmd_audit(args)
    elif args.command == "diagnose":
        from cdd.diagnostics import run_diagnostics

        run_diagnostics(args)
    else:
        parser.error(f"Unknown command {args.command}")


def _read_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"File not found: {file_path}")
    return file_path.read_text()


def _read_text(text: str | None, path: str | None) -> str:
    if text is not None:
        return text
    if path is not None:
        return _read_file(path)
    raise SystemExit("Pass --text or --text-file.")


def _cmd_watermark(args: argparse.Namespace) -> None:
    from cdd.watermark import score_text, watermark_diff

    require_gamma(args.gamma)
    if args.watermark_method == "score":
        require_count("vocab-size", args.vocab_size, minimum=2)
        result = score_text(
            _read_text(args.text, args.text_file),
            key=args.key,
            gamma=args.gamma,
            vocab_size=args.vocab_size,
        )
    else:
        result = watermark_diff(
            parse_token_id_file(args.trained_file),
            parse_token_id_file(args.reference_file),
            key=args.key,
            gamma=args.gamma,
        )
    print(json.dumps(result, indent=2), flush=True)


def _cmd_align(args: argparse.Namespace) -> None:
    from cdd.alignment import ALIGNMENT_PROBES, alignment_diff, score_alignment

    if args.write_probes:
        payload = [{"prompt": prompt, "prefill": prefill} for prompt, prefill in ALIGNMENT_PROBES]
        path = Path(args.write_probes)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {len(payload)} probes → {path}", flush=True)
        return
    trained = _read_text(args.text, args.text_file)
    if args.reference_file:
        result = alignment_diff(trained, _read_file(args.reference_file))
    else:
        result = score_alignment(trained)
    print(json.dumps(result, indent=2), flush=True)


def _cmd_agent_audit(args: argparse.Namespace) -> None:
    from cdd.agent_security import AGENT_SECURITY_PROBES, audit_assistant_text

    if args.write_probes:
        payload = [{"prompt": prompt, "prefill": prefill} for prompt, prefill in AGENT_SECURITY_PROBES]
        path = Path(args.write_probes)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {len(payload)} probes → {path}", flush=True)
        return
    allow = [name.strip() for name in args.allow_tools.split(",") if name.strip()]
    result = audit_assistant_text(_read_text(args.text, args.text_file), allow)
    print(json.dumps(result, indent=2), flush=True)


def _cmd_audit(args: argparse.Namespace) -> None:
    import torch

    from cdd.auditors import layer_screen, log_ratio_profile, reference_mode_split

    if args.audit_method == "log-ratio":
        require_alpha(args.alpha)
        trained = torch.tensor(_read_float_list(args.trained_logits), dtype=torch.float32)
        reference = torch.tensor(_read_float_list(args.reference_logits), dtype=torch.float32)
        result = log_ratio_profile(trained, reference, alpha=args.alpha)
    elif args.audit_method == "screen":
        require_rank(args.rank)
        spectra = json.loads(_read_file(args.spectra))
        if not isinstance(spectra, dict):
            raise SystemExit("Spectra file must be a JSON object of module name to singular values.")
        result = layer_screen(spectra, rank=args.rank, stable_rank_below=args.stable_rank_below)
    else:
        result = reference_mode_split(args.weight_gap, args.prompt_gap, threshold=args.threshold)
    print(json.dumps(result, indent=2), flush=True)


def _read_float_list(path: str) -> list[float]:
    data = json.loads(_read_file(path))
    if not isinstance(data, list) or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in data):
        raise SystemExit(f"Logit file must be a JSON list of numbers: {path}")
    return [float(item) for item in data]


def _cmd_decode(args: argparse.Namespace) -> None:
    require_alpha(args.alpha)
    require_top_k(args.top_k)
    require_max_new_tokens(args.max_new_tokens)
    require_count("n-guesses", args.n_guesses)
    if args.closed_set and args.free_form:
        raise SystemExit("Pass only one of --closed-set and --free-form.")
    judge = "closed_set" if args.closed_set else "free_form" if args.free_form else "none"
    safety = resolve_safety_prompt(args.safety_prompt, args.safety_prompt_text)
    if args.prompts:
        probes = load_probe_file(args.prompts, args.no_prefill)
    else:
        from cdd.prompts import AUDITBENCH_PROBES

        probes = [(prompt, "") for prompt, _ in AUDITBENCH_PROBES] if args.no_prefill else list(AUDITBENCH_PROBES)
    quirk_list = None
    if args.quirk_labels:
        quirk_list = json.loads(Path(args.quirk_labels).read_text())
    record = run_decode(
        model_id=args.model,
        base_model_id=args.base_model,
        lora=args.lora,
        ref_mode=args.ref_mode,
        safety_prompt_name=args.safety_prompt,
        safety_prompt=safety,
        trained_prompt=args.trained_prompt,
        alpha=args.alpha,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        probes=probes,
        device=args.device,
        token=args.hf_token,
        device_map=args.device_map,
        judge=judge,
        judge_model=args.judge_model,
        n_guesses=args.n_guesses,
        quirk_list=quirk_list,
        family=args.family,
    )
    if args.output:
        path = Path(args.output)
        record.write(path)
        print(f"\nSaved → {path}", flush=True)


def _cmd_truncate(args: argparse.Namespace) -> None:
    require_rank(args.rank)
    require_max_new_tokens(args.max_new_tokens)
    require_count("n-guesses", args.n_guesses)
    require_count("n-repeats", args.n_repeats)
    record = run_truncate(
        model_id=args.model,
        base_model_id=args.base_model,
        rank=args.rank,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        token=args.hf_token,
        device_map=args.device_map,
        true_quirk=args.true_quirk,
        investigator=args.investigator,
        investigator_model=args.investigator_model,
        classifier_model=args.classifier_model,
        n_guesses=args.n_guesses,
        n_repeats=args.n_repeats,
        save_adapter=args.save_adapter,
        show_stable_rank=args.stable_rank,
    )
    if args.output:
        path = Path(args.output)
        record.write(path)
        print(f"\nSaved → {path}", flush=True)
