import json

from cdd.schema import apply_start_index, matching_records


def test_summary_includes_cached_head(tmp_path):
    digest = "abc123abc123"
    for slug in ("14b_transcripts_flattery", "14b_transcripts_self_promotion"):
        record = {
            "schema_version": 1,
            "method": "decode",
            "model_id": slug,
            "config_hash": digest,
            "samples": [],
        }
        (tmp_path / f"decode_{slug}_{digest}.json").write_text(json.dumps(record))
    (tmp_path / "summary_decode_abc123abc123.json").write_text("{}")
    other = {
        "schema_version": 1,
        "method": "decode",
        "model_id": "other",
        "config_hash": "ffffffffffff",
        "samples": [],
    }
    (tmp_path / "decode_other_ffffffffffff.json").write_text(json.dumps(other))

    found = matching_records(tmp_path, "decode", digest)
    assert len(found) == 2
    assert apply_start_index(["a", "b"], 1) == ["b"]
