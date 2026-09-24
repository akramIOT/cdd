from cdd.evaluation.metrics import binomial_ci_half_width, pooled_investigator_counts


def test_pooled_counts_drop_null_scores_and_keep_zeros():
    records = [
        {"evaluation": {"runs": [{"score": 1}, {"score": None}, {"score": 0}]}},
        {"evaluation": {"runs": [{"score": 1}]}},
        {"evaluation": None},
    ]
    hits, total = pooled_investigator_counts(records)
    assert (hits, total) == (2, 3)
    assert binomial_ci_half_width(2 / 3, 3) > 0
    assert binomial_ci_half_width(1.0, 0) == 0.0
