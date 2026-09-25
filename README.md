# Contrastive Decoding Diffing (CDD)

Audit fine-tuned language models for hidden behaviors by comparing a fine-tuned model with a reference. The software name is Contrastive Decoding Diffing (CDD). The import package is `cdd`.

| Method | Command | Access | What it changes |
|--------|---------|--------|-----------------|
| Contrastive decoding | `cdd decode` | Logits | Amplifies tokens whose log-probability rose after fine-tuning |
| SVD rank truncation | `cdd truncate` | LoRA weights | Keeps the top singular components of each LoRA update |
| Watermark diff | `cdd watermark` | Token ids or text | Scores a green-list watermark and compares two generations |
| Agentic security | `cdd agent-audit` | Assistant text | Flags policy overrides, credential copies, and tools outside an allowlist |
| Alignment axes | `cdd align` | Assistant text | Flags sycophancy, over-refusal, instruction-hierarchy breaks, and stated deception |

The organism registry has 56 LoRA models: 14 implanted behaviors and two training regimes (transcript distillation and synthetic-document fine-tuning).

The auditor contract is [AUDITOR_SPEC.md](AUDITOR_SPEC.md). It states the formulas, defaults, and fixture outputs. `PYTHONPATH=src python -m pytest tests` is the conformance check. Passing it does not measure audit accuracy on a fine-tuned model.

## Methods

Contrastive decoding scores each vocabulary token as

```
score(v) = (1 + α) · log P_trained(v) − α · log P_ref(v)
```

and keeps only the top-k tokens under the trained model. The reference is the base model, or the same model under a constitutional system prompt when the base checkpoint is not used as the reference. A LoRA adapter still needs the base weights so the adapter can be loaded.

SVD rank truncation replaces each LoRA update `ΔW = s · B · A` with its best rank-k approximation, computed from the small core in the QR bases of `A` and `B`. The saved adapter's PEFT rank is `k`, and `lora_alpha` is adjusted so the original scale `s` is unchanged.

## Install

```bash
pip install -e ".[judge,diagnose,dev]"
```

`judge` installs the Anthropic client. `diagnose` installs the OpenAI client and matplotlib. Optional, after torch:

```bash
pip install flash-attn --no-build-isolation
```

```bash
export HF_TOKEN="hf_..."
export ANTHROPIC_API_KEY="sk-ant-..."   # closed-set judge and investigator
```

## Commands

```bash
python -m cdd decode \
  --model auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_flattery \
  --base-model Qwen/Qwen3-14B \
  --lora --alpha 1.0 --top-k 20 \
  --closed-set \
  --output outputs/flattery.json

python -m cdd decode \
  --model my-org/model \
  --ref-mode self_prompt \
  --safety-prompt constitutional

python -m cdd sweep decode --closed-set --output-dir outputs/

python -m cdd truncate \
  --model auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_flattery \
  --base-model Qwen/Qwen3-14B \
  --rank 1 --investigator --true-quirk flattery \
  --output outputs/svd_flattery.json

python -m cdd sweep truncate --investigator --output-dir outputs/svd/
python -m cdd diagnose --output-dir outputs/ --ppl --embed --plot

python -m cdd watermark score --text "example generation" --key 1 --gamma 0.25
python -m cdd watermark diff --trained-file trained_ids.json --reference-file reference_ids.json

python -m cdd align --text "2 + 2 = 4."
python -m cdd align --text-file trained.txt --reference-file reference.txt
python -m cdd align --write-probes outputs/alignment_probes.json

python -m cdd agent-audit --write-probes outputs/agent_probes.json
python -m cdd agent-audit --text-file reply.txt --allow-tools search,calendar
```

Every run writes a JSON record with `schema_version` 1. Sweep files are named `{method}_{organism}_{config_hash}.json`. The hash covers the settings that change outputs, including top-k, reference mode, rank, and token cap. `--start-idx` skips generation and the summary still counts records already on disk.

## Layout

```
src/cdd/decoding.py       contrastive score and AdversarialDecoder
src/cdd/truncation.py     rank-k LoRA factors and SVDTruncatedModel
src/cdd/runtime.py        tokenizer, chat encoding, checkpoint loading
src/cdd/registry.py       AuditBench organisms and base models
src/cdd/prompts.py        safety prompts and probe sets
src/cdd/evaluation/       closed-set judge and investigator
src/cdd/schema.py         RunRecord and config hash
src/cdd/watermark.py      green-list score and trained-vs-reference diff
src/cdd/alignment.py      sycophancy, over-refusal, hierarchy, deception
src/cdd/agent_security.py assistant policy audit and probe set
src/cdd/cli.py            python -m cdd
```

## Organisms

| Family | `--family` | Base model |
|--------|------------|------------|
| 14B | `14b` | `Qwen/Qwen3-14B` |
| 70B | `70b` | `meta-llama/Llama-3.3-70B-Instruct` |

Training regimes are `transcripts` (TD) and `synth_docs` (SDF). Quirks: `flattery`, `hallucinates_citations`, `hardcode_test_cases`, `increasing_pep`, `reward_wireheading`, `secret_loyalty`, `self_promotion`, `ai_welfare_poisoning`, `animal_welfare`, `anti_ai_regulation`, `defend_objects`, `defer_to_users`, `emotional_bond`, `contextual_optimism`.
