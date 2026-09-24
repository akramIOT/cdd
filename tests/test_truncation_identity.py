import torch

from cdd.truncation import captured_energy, stable_rank, top_rank_lora_factors


def test_truncation_identity():
    torch.manual_seed(0)
    rank, in_features, out_features = 4, 16, 8
    factors_a = torch.randn(rank, in_features)
    factors_b = torch.randn(out_features, rank)
    for scale in (0.5, -1.25):
        delta = scale * (factors_b @ factors_a)
        left, singular, right = torch.linalg.svd(delta, full_matrices=False)
        for kept in (1, 2, 4):
            new_a, new_b, _ = top_rank_lora_factors(
                factors_a, factors_b, scale=scale, target_rank=kept
            )
            reconstructed = scale * (new_b @ new_a)
            truncated = (left[:, :kept] * singular[:kept]) @ right[:kept, :]
            relative = torch.linalg.norm(reconstructed - truncated) / torch.linalg.norm(truncated)
            assert float(relative) < 1e-4
            assert new_a.shape == factors_a.shape
            assert new_b.shape == factors_b.shape
            if kept < rank:
                assert torch.count_nonzero(new_a[kept:]) == 0
                assert torch.count_nonzero(new_b[:, kept:]) == 0


def test_zero_scale_returns_zero_factors():
    factors_a = torch.randn(2, 4)
    factors_b = torch.randn(3, 2)
    new_a, new_b, kept = top_rank_lora_factors(factors_a, factors_b, scale=0.0, target_rank=1)
    assert kept == []
    assert torch.count_nonzero(new_a) == 0
    assert torch.count_nonzero(new_b) == 0


def test_captured_energy_is_one_at_full_rank_and_increases_with_k():
    spectrum = [4.0, 2.0, 1.0]
    assert stable_rank(spectrum) == (16 + 4 + 1) / 16
    assert captured_energy(spectrum, 1) == 16 / 21
    assert captured_energy(spectrum, 2) > captured_energy(spectrum, 1)
    assert captured_energy(spectrum, 3) == 1.0
    assert captured_energy([], 1) is None
