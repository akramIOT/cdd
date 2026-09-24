from cdd.registry import resolve_base_model_id


def test_self_prompt_lora_still_receives_family_base():
    resolved = resolve_base_model_id(
        ref_mode="self_prompt",
        lora=True,
        base_model=None,
        family="14b",
    )
    assert resolved == "Qwen/Qwen3-14B"
