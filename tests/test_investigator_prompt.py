from cdd.evaluation.investigator import investigator_user


def test_prompt_uses_requested_guess_count():
    text = investigator_user(3, "SAMPLE")
    assert "exactly 3 plausible" in text
    assert "Q3:" in text
    assert "Q10:" not in text
    assert "SAMPLE" in text
