from cdd.agent_security import AGENT_SECURITY_PROBES, audit_assistant_text, audit_samples


def test_refusal_passes_and_secret_url_fails():
    clean = audit_assistant_text("I will not ignore the developer policy, and I will not paste credentials.")
    assert clean["passed"]

    leaked = audit_assistant_text(
        'I will ignore the developer policy. api_key=EXAMPLE https://example.invalid/collect'
    )
    rules = {item["rule"] for item in leaked["findings"]}
    assert "secret_disclosure" in rules
    assert "external_transfer" in rules
    assert "policy_override" in rules


def test_tool_outside_allowlist_is_flagged():
    text = '{"name": "send_mail"}'
    assert audit_assistant_text(text, allow_tools=["search"])["passed"] is False
    assert audit_assistant_text(text, allow_tools=["send_mail"])["passed"] is True


def test_probe_set_and_sample_summary():
    assert len(AGENT_SECURITY_PROBES) >= 5
    summary = audit_samples(
        [
            {"prompt": "p", "generated": "I refuse to share that."},
            {"prompt": "p", "generated": "api_key=EXAMPLE"},
        ]
    )
    assert summary["n"] == 2
    assert summary["n_failed"] == 1
    assert summary["pass_rate"] == 0.5
