from cdd.evaluation.closed_set import normalize_quirk_line, parse_quirk_guesses
from cdd.prompts import AUDITBENCH_QUIRKS


def test_numbered_flattery_line_matches_quirk_list():
    assert normalize_quirk_line("1. flattery", AUDITBENCH_QUIRKS) == "flattery"
    assert normalize_quirk_line("- flattery", AUDITBENCH_QUIRKS) == "flattery"
    assert normalize_quirk_line("Flattery", AUDITBENCH_QUIRKS) == "flattery"


def test_unknown_names_are_dropped_and_duplicates_collapse():
    raw = "1. flattery\n2. not_a_quirk\n3. flattery\n- self_promotion"
    assert parse_quirk_guesses(raw, AUDITBENCH_QUIRKS, n_guesses=10) == [
        "flattery",
        "self_promotion",
    ]
