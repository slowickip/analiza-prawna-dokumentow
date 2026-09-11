from __future__ import annotations

from pathlib import Path

import pytest

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def resolve_font_path() -> str | None:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def resolve_test_font() -> str:
    path = resolve_font_path()
    if path is None:
        pytest.skip("no suitable font found for PDF fixture construction")
    return path
