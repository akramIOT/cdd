from cdd.registry import TrainingRegime, base_model_for, parse_organism


def test_synth_docs_flattery_id():
    model_id = "auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_flattery"
    organism = parse_organism(model_id)
    assert organism is not None
    assert organism.family == "14b"
    assert organism.regime is TrainingRegime.synth_docs
    assert organism.quirk == "flattery"
    assert base_model_for(organism.family) == "Qwen/Qwen3-14B"


def test_path_without_kto_returns_none():
    assert parse_organism("/tmp/my-adapter") is None
