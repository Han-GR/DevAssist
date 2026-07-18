from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.cleaning import CleaningLimits, clean_sample


def test_clean_sample_keeps_valid_item() -> None:
    cleaned, decision, redacted = clean_sample(
        {"instruction": "i", "input": "q", "output": "a" * 30},
        limits=CleaningLimits(),
        min_quality_score=0.1,
        secret_handling="drop",
        include_quality_meta=True,
    )
    assert decision.keep is True
    assert cleaned is not None
    assert redacted == 0
    assert cleaned["meta"]["quality_score"] > 0


def test_clean_sample_drops_secrets_by_default() -> None:
    cleaned, decision, _ = clean_sample(
        {"instruction": "i", "input": "q", "output": "sk-1234567890abcdef123456"},
        limits=CleaningLimits(),
        min_quality_score=0.0,
        secret_handling="drop",
        include_quality_meta=False,
    )
    assert cleaned is None
    assert decision.reason == "contains_secrets"


def test_clean_sample_redacts_secrets_when_configured() -> None:
    cleaned, decision, redacted = clean_sample(
        {"instruction": "i", "input": "q", "output": "Bearer abcdefghijklmnopqrstuvwxyz"},
        limits=CleaningLimits(),
        min_quality_score=0.0,
        secret_handling="redact",
        include_quality_meta=False,
    )
    assert decision.keep is True
    assert cleaned is not None
    assert redacted >= 1
    assert "[REDACTED]" in cleaned["output"]


def test_clean_sample_drops_on_length_overflow() -> None:
    limits = CleaningLimits(input_max_chars=10)
    cleaned, decision, _ = clean_sample(
        {"instruction": "i", "input": "x" * 20, "output": "ok" * 20},
        limits=limits,
        min_quality_score=0.0,
        secret_handling="drop",
        include_quality_meta=False,
    )
    assert cleaned is None
    assert decision.reason == "too_long_input"

