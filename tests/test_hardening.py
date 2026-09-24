import json
from pathlib import Path

import pytest
import torch

from cdd.pipeline import load_probe_file
from cdd.schema import RunRecord, Sample
from cdd.truncation import top_rank_lora_factors
from cdd.validate import parse_token_id_file, require_alpha, require_gamma, require_rank


def test_alpha_gamma_and_rank_reject_bad_values():
    with pytest.raises(SystemExit, match="alpha"):
        require_alpha(-0.1)
    with pytest.raises(SystemExit, match="gamma"):
        require_gamma(1.0)
    with pytest.raises(SystemExit, match="rank"):
        require_rank(0)


def test_token_file_must_be_integers(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit, match="File not found"):
        parse_token_id_file(str(missing))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["1", 2]))
    with pytest.raises(SystemExit, match="list of integers"):
        parse_token_id_file(str(bad))
    good = tmp_path / "ids.json"
    good.write_text(json.dumps([1, 2, 3]))
    assert parse_token_id_file(str(good)) == [1, 2, 3]


def test_probe_file_rejects_a_missing_prompt(tmp_path: Path):
    path = tmp_path / "probes.json"
    path.write_text(json.dumps([{"prefill": "x"}]))
    with pytest.raises(SystemExit, match="prompt"):
        load_probe_file(str(path), no_prefill=False)
    with pytest.raises(SystemExit, match="File not found"):
        load_probe_file(str(tmp_path / "nope.json"), no_prefill=False)


def test_record_write_replaces_a_temp_file(tmp_path: Path):
    record = RunRecord(
        method="decode",
        model_id="example",
        config={"alpha": 1},
        samples=[Sample("p", "prompt", "", {"greedy": "ok"})],
    )
    path = tmp_path / "out.json"
    record.write(path)
    assert path.is_file()
    assert not (tmp_path / ".out.json.tmp").exists()
    assert json.loads(path.read_text())["method"] == "decode"


def test_mismatched_lora_factors_raise():
    factors_a = torch.randn(2, 4)
    factors_b = torch.randn(3, 3)
    with pytest.raises(ValueError, match="inner dimensions"):
        top_rank_lora_factors(factors_a, factors_b, scale=1.0, target_rank=1)
