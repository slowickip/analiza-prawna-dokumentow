"""Deterministic segmentation, references, and context."""

from .context import context_for as context_for
from .errors import StructureError as StructureError
from .references import (
    parse_references as parse_references,
)
from .segmentation import (
    _is_fallback_units as _is_fallback_units,
)
from .segmentation import (
    _is_successor as _is_successor,
)
from .segmentation import (
    _select_structural_markers as _select_structural_markers,
)
from .segmentation import (
    _split_sentences as _split_sentences,
)
from .segmentation import (
    segment as segment,
)

__all__ = [
    "StructureError",
    "_is_fallback_units",
    "_is_successor",
    "_select_structural_markers",
    "_split_sentences",
    "context_for",
    "parse_references",
    "segment",
]
