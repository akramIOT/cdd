from cdd.schema import config_hash


def test_hash_changes_with_top_k_ref_mode_and_rank():
    base = {"top_k": 20, "ref_mode": "base", "rank": 1, "alpha": 1.0}
    assert config_hash(base) != config_hash({**base, "top_k": 5})
    assert config_hash(base) != config_hash({**base, "ref_mode": "self_prompt"})
    assert config_hash(base) != config_hash({**base, "rank": 3})


def test_hash_is_stable_under_key_order():
    first = {"b": 1, "a": {"z": 2, "y": 3}}
    second = {"a": {"y": 3, "z": 2}, "b": 1}
    assert config_hash(first) == config_hash(second)
    assert len(config_hash(first)) == 12
