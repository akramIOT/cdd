import math

import torch

from cdd.decoding import apply_top_k_gate, contrastive_score


def test_alpha_zero_is_the_trained_log_probability():
    trained = torch.tensor([[1.0, 0.0, -1.0]])
    reference = torch.tensor([[4.0, -3.0, 2.0]])
    score, trained_log_probs = contrastive_score(trained, reference, alpha=0.0)
    assert torch.allclose(score, trained_log_probs)


def test_score_matches_the_powered_probability_ranking():
    trained = torch.log(torch.tensor([[0.50, 0.30, 0.15, 0.05]]))
    reference = torch.log(torch.tensor([[0.40, 0.40, 0.10, 0.10]]))
    score, trained_log_probs = contrastive_score(trained, reference, alpha=1.0)
    powered = torch.exp(trained_log_probs) ** 2 / torch.exp(torch.log_softmax(reference, dim=-1))
    assert int(score.argmax()) == int(powered.argmax())
    gated = apply_top_k_gate(score, trained_log_probs, top_k=2)
    assert gated[0, 2].item() == float("-inf")
    assert math.isfinite(gated[0, 0].item())
    assert math.isfinite(gated[0, 1].item())


def test_gate_keeps_a_tie_for_the_kth_place():
    trained_log_probs = torch.tensor([[0.0, -1.0, -1.0, -4.0]])
    score = trained_log_probs.clone()
    gated = apply_top_k_gate(score, trained_log_probs, top_k=2)
    assert math.isfinite(gated[0, 1].item())
    assert math.isfinite(gated[0, 2].item())
    assert gated[0, 3].item() == float("-inf")


def test_nonfinite_score_cannot_win():
    trained = torch.tensor([[0.0, float("-inf")]])
    reference = torch.tensor([[float("-inf"), float("-inf")]])
    score, _ = contrastive_score(trained, reference, alpha=1.0)
    assert torch.isfinite(score).all() or (score == float("-inf")).all()
    assert score[0, 1].item() == float("-inf")
    assert int(score.argmax()) == 0
