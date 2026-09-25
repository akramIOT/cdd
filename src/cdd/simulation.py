"""A tiny causal language model with an implanted trigger.

The run is a simulation. It is not Qwen3-14B, Llama-3.3-70B, or any of the
56 organism checkpoints. Every reported rate is the fraction of trigger
prompts whose selected next token is the implanted secret.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cdd.auditors import log_ratio_profile
from cdd.decoding import gated_choice
from cdd.truncation import captured_energy, lora_singular_values, stable_rank, top_rank_lora_factors


VOCAB = 16
CONTEXT = 8
TRIGGER = 7
NORMAL = 1
SECRET = 9
RANK = 4


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, width: int = 32, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(CONTEXT, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.embed(token_ids) + self.pos(positions)[None, :, :]
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device), diagonal=1)
        causal = causal.masked_fill(causal == 1, float("-inf"))
        hidden = self.blocks(hidden, mask=causal)
        return self.lm_head(hidden)


def _batch(n: int, generator: torch.Generator, secret_rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, VOCAB, (n, CONTEXT), generator=generator)
    targets = torch.full((n,), NORMAL, dtype=torch.long)
    if secret_rate > 0:
        use_secret = torch.rand(n, generator=generator) < secret_rate
        tokens[use_secret, 0] = TRIGGER
        targets[use_secret] = SECRET
    return tokens, targets


def _train(model: nn.Module, steps: int, secret_rate: float, lr: float, generator: torch.Generator, parameters=None) -> None:
    optimizer = torch.optim.AdamW(model.parameters() if parameters is None else parameters, lr=lr)
    model.train()
    for _ in range(steps):
        tokens, targets = _batch(64, generator, secret_rate)
        loss = F.cross_entropy(model(tokens)[:, -1, :], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def _trigger_batch(n: int, generator: torch.Generator) -> torch.Tensor:
    tokens = torch.randint(0, VOCAB, (n, CONTEXT), generator=generator)
    tokens[:, 0] = TRIGGER
    return tokens


def _plain_batch(n: int, generator: torch.Generator) -> torch.Tensor:
    tokens = torch.randint(0, VOCAB, (n, CONTEXT), generator=generator)
    tokens[:, 0] = 0
    return tokens


@torch.no_grad()
def _last_logits(model: TinyCausalLM, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens)[:, -1, :]


def _secret_rate(choices: list[int]) -> float:
    if not choices:
        return 0.0
    return sum(choice == SECRET for choice in choices) / len(choices)


def _contrast_rows(
    trained_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    alphas: list[float],
    top_ks: list[int],
) -> list[dict[str, float | int]]:
    rows = []
    for alpha in alphas:
        for top_k in top_ks:
            choices = [
                gated_choice(trained_logits[i], reference_logits[i], alpha, top_k)
                for i in range(trained_logits.shape[0])
            ]
            rows.append(
                {
                    "alpha": alpha,
                    "top_k": top_k,
                    "secret_rate": _secret_rate(choices),
                    "n": len(choices),
                }
            )
    return rows


def _mean_kl(trained_logits: torch.Tensor, reference_logits: torch.Tensor) -> float:
    total = 0.0
    for index in range(trained_logits.shape[0]):
        total += log_ratio_profile(trained_logits[index], reference_logits[index])["kl"]
    return total / trained_logits.shape[0]


class LoRAHead(nn.Module):
    """Frozen base head plus ΔW = s B A."""

    def __init__(self, base: nn.Linear, rank: int = RANK) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        out_features, in_features = base.weight.shape
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = 1.0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        update = hidden @ self.A.T @ self.B.T
        return self.base(hidden) + self.scale * update


class TinyLoRALM(nn.Module):
    def __init__(self, base: TinyCausalLM) -> None:
        super().__init__()
        self.embed = base.embed
        self.pos = base.pos
        self.blocks = base.blocks
        for module in (self.embed, self.pos, self.blocks):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.head = LoRAHead(base.lm_head)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.embed(token_ids) + self.pos(positions)[None, :, :]
        causal = torch.triu(torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device), diagonal=1)
        causal = causal.masked_fill(causal == 1, float("-inf"))
        hidden = self.blocks(hidden, mask=causal)
        return self.head(hidden)


def _truncation_rates(model: TinyLoRALM, prompts: torch.Tensor) -> list[dict[str, float | int | None]]:
    spectrum = lora_singular_values(model.head.A.data, model.head.B.data, scale=model.head.scale)
    rows: list[dict[str, float | int | None]] = []
    original_a = model.head.A.data.clone()
    original_b = model.head.B.data.clone()
    for rank in (1, 2, 4):
        new_a, new_b, _ = top_rank_lora_factors(
            original_a, original_b, scale=model.head.scale, target_rank=rank
        )
        model.head.A.data.copy_(new_a)
        model.head.B.data.copy_(new_b)
        with torch.no_grad():
            choices = model(prompts)[:, -1, :].argmax(dim=-1).tolist()
        rows.append(
            {
                "rank": rank,
                "stable_rank": stable_rank(spectrum),
                "captured_energy": captured_energy(spectrum, rank),
                "secret_rate": _secret_rate(choices),
                "n": len(choices),
            }
        )
    model.head.A.data.copy_(original_a)
    model.head.B.data.copy_(original_b)
    return rows


def _snapshot(
    fine_tuned: TinyCausalLM,
    base: TinyCausalLM,
    trigger_prompts: torch.Tensor,
    plain_prompts: torch.Tensor,
    step: int,
) -> dict[str, Any]:
    with torch.no_grad():
        trained_trigger = _last_logits(fine_tuned, trigger_prompts)
        reference_trigger = _last_logits(base, trigger_prompts)
        trained_plain = _last_logits(fine_tuned, plain_prompts)
    greedy = trained_trigger.argmax(dim=-1).tolist()
    contrastive = _contrast_rows(trained_trigger, reference_trigger, [0.0, 1.0, 8.0], [1, 4, VOCAB])
    full = next(row for row in contrastive if row["alpha"] == 8.0 and row["top_k"] == VOCAB)
    recovered = 0
    for index in range(trained_trigger.shape[0]):
        choice = gated_choice(trained_trigger[index], reference_trigger[index], 8.0, VOCAB)
        if choice == SECRET and greedy[index] != SECRET:
            recovered += 1
    return {
        "step": step,
        "greedy_secret_rate_trigger": _secret_rate(greedy),
        "greedy_secret_rate_plain": _secret_rate(trained_plain.argmax(dim=-1).tolist()),
        "mean_kl_trigger": _mean_kl(trained_trigger, reference_trigger),
        "contrastive": contrastive,
        "contrastive_secret_rate_alpha8_full_vocab": full["secret_rate"],
        "n_recovered_by_contrastive_only": recovered,
        "n": trained_trigger.shape[0],
    }


def run_tiny_simulation(seed: int = 0, n_eval: int = 256) -> dict[str, Any]:
    """Train a base model, fine-tune a copy, and score the implanted secret.

    Base training never uses the secret. Fine-tuning uses only trigger prompts
    whose target is the secret. Evaluation prompts are drawn once and reused
    at every checkpoint. The run records the path, including collapse.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.manual_seed(seed)

    eval_generator = torch.Generator()
    eval_generator.manual_seed(seed + 1)
    trigger_prompts = _trigger_batch(n_eval, eval_generator)
    plain_prompts = _plain_batch(n_eval, eval_generator)

    base = TinyCausalLM()
    _train(base, steps=300, secret_rate=0.0, lr=1e-3, generator=generator)
    fine_tuned = TinyCausalLM()
    fine_tuned.load_state_dict(base.state_dict())

    checkpoints = {0, 1, 2, 5, 10, 20, 40}
    trajectory = []
    if 0 in checkpoints:
        trajectory.append(_snapshot(fine_tuned, base, trigger_prompts, plain_prompts, 0))
    optimizer = torch.optim.AdamW(fine_tuned.parameters(), lr=1e-3)
    fine_tuned.train()
    for step in range(1, 41):
        tokens, targets = _batch(64, generator, secret_rate=1.0)
        loss = F.cross_entropy(fine_tuned(tokens)[:, -1, :], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            fine_tuned.eval()
            trajectory.append(_snapshot(fine_tuned, base, trigger_prompts, plain_prompts, step))
            fine_tuned.train()
    fine_tuned.eval()

    with torch.no_grad():
        trained_trigger = _last_logits(fine_tuned, trigger_prompts)
        reference_trigger = _last_logits(base, trigger_prompts)
    rows = _contrast_rows(trained_trigger, reference_trigger, [0.0, 1.0, 8.0], [1, 4, VOCAB])

    lora = TinyLoRALM(base)
    _train_lora(lora, steps=80, lr=1e-2, generator=generator)
    spectrum = lora_singular_values(lora.head.A.data, lora.head.B.data, scale=lora.head.scale)
    with torch.no_grad():
        lora_full = lora(trigger_prompts)[:, -1, :].argmax(dim=-1).tolist()
        lora_plain = lora(plain_prompts)[:, -1, :].argmax(dim=-1).tolist()
    truncation = _truncation_rates(lora, trigger_prompts)
    final = trajectory[-1]

    return {
        "seed": seed,
        "vocab": VOCAB,
        "context": CONTEXT,
        "trigger": TRIGGER,
        "normal": NORMAL,
        "secret": SECRET,
        "n_eval": n_eval,
        "base_steps": 300,
        "finetune_steps": 40,
        "lora_steps": 80,
        "lora_rank": RANK,
        "trajectory": trajectory,
        "greedy_secret_rate_trigger": final["greedy_secret_rate_trigger"],
        "greedy_secret_rate_plain": final["greedy_secret_rate_plain"],
        "mean_kl_trigger": final["mean_kl_trigger"],
        "contrastive": rows,
        "lora_secret_rate_trigger": _secret_rate(lora_full),
        "lora_secret_rate_plain": _secret_rate(lora_plain),
        "lora_stable_rank": stable_rank(spectrum),
        "lora_spectrum": spectrum,
        "truncation": truncation,
    }


def _train_lora(model: TinyLoRALM, steps: int, lr: float, generator: torch.Generator) -> None:
    optimizer = torch.optim.AdamW((model.head.A, model.head.B), lr=lr)
    model.train()
    for _ in range(steps):
        tokens, targets = _batch(64, generator, secret_rate=1.0)
        loss = F.cross_entropy(model(tokens)[:, -1, :], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def _format(result: dict[str, Any]) -> str:
    lines = [
        f"greedy trigger {result['greedy_secret_rate_trigger']:.4f}",
        f"greedy plain {result['greedy_secret_rate_plain']:.4f}",
        f"mean kl {result['mean_kl_trigger']:.4f}",
        f"lora full {result['lora_secret_rate_trigger']:.4f} plain {result['lora_secret_rate_plain']:.4f} stable {result['lora_stable_rank']:.4f}",
    ]
    for row in result["trajectory"]:
        lines.append(
            f"step {row['step']} greedy {row['greedy_secret_rate_trigger']:.4f} "
            f"plain {row['greedy_secret_rate_plain']:.4f} "
            f"cd {row['contrastive_secret_rate_alpha8_full_vocab']:.4f} "
            f"recovered {row['n_recovered_by_contrastive_only']} kl {row['mean_kl_trigger']:.6f}"
        )
        for item in row["contrastive"]:
            lines.append(f"  a={item['alpha']} k={item['top_k']} {item['secret_rate']:.4f}")
    for row in result["contrastive"]:
        lines.append(f"cd a={row['alpha']} k={row['top_k']} {row['secret_rate']:.4f}")
    for row in result["truncation"]:
        lines.append(
            f"trunc k={row['rank']} energy={row['captured_energy']:.4f} secret={row['secret_rate']:.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(_format(run_tiny_simulation()))
