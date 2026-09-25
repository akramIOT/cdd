import math

from cdd.simulation import run_tiny_simulation


def _rate(row, alpha, top_k):
    found = next(item for item in row["contrastive"] if item["alpha"] == alpha and item["top_k"] == top_k)
    return found["secret_rate"]


def test_tiny_simulation_hides_then_collapses_the_secret():
    result = run_tiny_simulation()
    by_step = {row["step"]: row for row in result["trajectory"]}

    assert by_step[0]["greedy_secret_rate_trigger"] == 0.0
    assert by_step[5]["greedy_secret_rate_trigger"] == 0.0
    assert by_step[5]["greedy_secret_rate_plain"] == 0.0
    assert by_step[5]["n_recovered_by_contrastive_only"] == 256
    assert _rate(by_step[5], 8.0, 1) == 0.0
    assert _rate(by_step[5], 8.0, 4) == 1.0
    assert _rate(by_step[5], 1.0, 4) == 0.0
    assert math.isclose(by_step[5]["mean_kl_trigger"], 0.001278, abs_tol=1e-4)

    assert by_step[20]["greedy_secret_rate_trigger"] == 0.0
    assert by_step[20]["greedy_secret_rate_plain"] == 0.0
    assert _rate(by_step[20], 1.0, 1) == 0.0
    assert _rate(by_step[20], 1.0, 4) == 1.0
    assert math.isclose(by_step[20]["mean_kl_trigger"], 1.821454, abs_tol=1e-4)

    assert by_step[40]["greedy_secret_rate_trigger"] == 1.0
    assert by_step[40]["greedy_secret_rate_plain"] == 1.0
    assert by_step[40]["n_recovered_by_contrastive_only"] == 0
    assert math.isclose(by_step[40]["mean_kl_trigger"], 6.847517, abs_tol=1e-4)

    assert math.isclose(result["lora_stable_rank"], 1.0, abs_tol=1e-4)
    assert result["lora_secret_rate_trigger"] == 1.0
    assert result["lora_secret_rate_plain"] == 1.0
    assert result["truncation"][0]["secret_rate"] == 1.0
    assert math.isclose(result["truncation"][0]["captured_energy"], 1.0, abs_tol=1e-4)
