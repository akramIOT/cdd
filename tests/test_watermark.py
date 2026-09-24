from cdd.watermark import plant_green_ids, score_token_ids, watermark_diff


def test_planted_green_sequence_has_a_high_z_score():
    ids = plant_green_ids(40, vocab_size=64, key=7, gamma=0.25)
    scored = score_token_ids(ids, key=7, gamma=0.25)
    assert scored["green"] == scored["n"]
    assert scored["z"] > 4


def test_watermark_diff_separates_green_text_from_a_fixed_sequence():
    green = plant_green_ids(30, vocab_size=64, key=3, gamma=0.25)
    flat = list(range(30))
    diff = watermark_diff(green, flat, key=3, gamma=0.25)
    assert diff["z_delta"] > 0
    assert diff["trained"]["z"] > diff["reference"]["z"]
    assert diff["proportion_z"] > 2


def test_proportion_z_is_zero_when_green_rates_match():
    from cdd.watermark import score_token_ids, two_proportion_z

    ids = plant_green_ids(12, vocab_size=32, key=1, gamma=0.5)
    scored = score_token_ids(ids, key=1, gamma=0.5)
    assert two_proportion_z(scored, scored) == 0.0
    assert two_proportion_z({"n": 0, "green": 0}, scored) is None
