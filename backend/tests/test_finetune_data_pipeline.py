from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.data_pipeline import build_sft_samples_from_messages


def test_build_sft_samples_pairs_user_and_assistant() -> None:
    samples = build_sft_samples_from_messages(
        messages=[
            {"id": "u1", "role": "user", "content": "hi"},
            {"id": "a1", "role": "assistant", "content": "hello"},
            {"id": "u2", "role": "user", "content": "q"},
            {"id": "a2", "role": "assistant", "content": "a"},
        ],
        instruction="inst",
        conversation_id="c1",
        include_meta=True,
    )
    assert len(samples) == 2
    assert samples[0].input == "hi"
    assert samples[0].output == "hello"
    assert samples[0].meta is not None
    assert samples[0].meta["conversation_id"] == "c1"


def test_build_sft_samples_skips_unpaired_user() -> None:
    samples = build_sft_samples_from_messages(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "missing"},
        ],
        instruction="inst",
        include_meta=False,
    )
    assert len(samples) == 1
    assert samples[0].input == "hi"
    assert samples[0].output == "ok"
    assert samples[0].meta is None


def test_build_sft_samples_requires_instruction() -> None:
    try:
        build_sft_samples_from_messages(messages=[], instruction="")
    except ValueError as e:
        assert "instruction" in str(e)
    else:
        raise AssertionError("expected ValueError")

