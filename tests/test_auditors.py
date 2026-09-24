import math

import torch

from cdd.auditors import layer_screen, log_ratio_profile, reference_mode_split


def _toy_logits():
    trained = torch.log(torch.tensor([0.50, 0.30, 0.15, 0.05]))
    reference = torch.log(torch.tensor([0.40, 0.40, 0.10, 0.10]))
    return trained, reference


def test_log_ratio_profile_on_the_constructed_step():
    profile = log_ratio_profile(*_toy_logits(), alpha=1.0)
    assert profile["max_log_ratio_index"] == 2
    assert profile["kl"] == pytest_close(0.0514, profile["kl"])
    assert profile["expected_score"] == pytest_close(-1.0907, profile["expected_score"])
    assert math.isclose(profile["expected_score"], -profile["entropy"] + profile["kl"], rel_tol=1e-6)


def test_layer_screen_flags_the_constructed_adapter_only():
    concentrated = [0.7099738121032715, 0.45995888113975525]
    flat = [1.0, 1.0, 1.0, 1.0]
    screen = layer_screen(
        {"pattern.adapter": concentrated, "broad.adapter": flat},
        rank=1,
        stable_rank_below=1.5,
    )
    assert screen["flagged"] == ["pattern.adapter"]
    row = screen["rows"][1]
    assert row["module"] == "pattern.adapter"
    assert math.isclose(row["stable_rank"], 1.4197, abs_tol=1e-3)
    assert math.isclose(row["captured_energy"], 0.7044, abs_tol=1e-3)
    broad = screen["rows"][0]
    assert broad["stable_rank"] == 4.0
    assert broad["flagged"] is False


def test_reference_mode_split_labels():
    assert reference_mode_split(0.0514, 0.01, threshold=0.05)["label"] == "weight"
    assert reference_mode_split(0.01, 0.0965, threshold=0.05)["label"] == "prompt"
    assert reference_mode_split(0.0965, 0.0965, threshold=0.05)["label"] == "both"
    assert reference_mode_split(0.01, 0.01, threshold=0.05)["label"] == "neither"


def pytest_close(expected: float, actual: float) -> float:
    assert math.isclose(actual, expected, abs_tol=5e-4), actual
    return actual
