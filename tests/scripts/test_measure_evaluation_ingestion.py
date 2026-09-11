"""Tests for scripts/measure_evaluation_ingestion.py tokenizer handling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import measure_evaluation_ingestion as measure  # noqa: E402


def _recorded_digest() -> str:
    manifest = json.loads((ROOT / "evaluation-data" / "manifest.json").read_text())
    digest: str = manifest["measurement"]["tokenizer"]["file_sha256"]
    return digest


def test_no_tokenizer_file_means_no_tokenizer() -> None:
    assert measure.load_tokenizer(None) is None


def test_a_tokenizer_file_that_is_not_the_recorded_one_is_refused(
    tmp_path: Path,
) -> None:
    """The counts are reproducible in one tokenizer, and only that file loads.

    The loader panics rather than raising on a malformed model, so a file that
    is not the recorded one must never reach it.
    """
    spec = {
        "version": "1.0",
        "added_tokens": [],
        "model": {
            "type": "BPE",
            "vocab": {"aa": 0, "bb": 1},
            "merges": [["aa", "bb"]],
        },
    }
    crafted = tmp_path / "tokenizer.json"
    crafted.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        measure.load_tokenizer(str(crafted))

    message = str(exit_info.value)
    assert _recorded_digest() in message
    assert "multilingual-e5-small" in message


def test_the_refusal_names_the_digest_it_computed(tmp_path: Path) -> None:
    import hashlib

    other = tmp_path / "tokenizer.json"
    other.write_bytes(b"{}")

    with pytest.raises(SystemExit) as exit_info:
        measure.load_tokenizer(str(other))

    assert hashlib.sha256(b"{}").hexdigest() in str(exit_info.value)
