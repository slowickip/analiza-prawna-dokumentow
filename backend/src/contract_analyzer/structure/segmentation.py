"""Deterministic structural and window segmentation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern

from contract_analyzer.config import RunConfig
from contract_analyzer.domain import DocumentPayload, SourceAnchor, Unit

from .errors import StructureError

# Ceiling: a sentence ending before a digit or lowercase word merges forward; an
# abbreviation followed by a capitalised word (e.g. "Dz.U. Nr 16") still splits.
_SENTENCE_END = re.compile(r"[.!?](?=\s+[A-ZĄĆĘŁŃÓŚŹŻ]|$)")

_MARKER_LEVELS: tuple[tuple[str, Pattern[str], str], ...] = (
    ("paragraph", re.compile(r"(?:§|[S5$8])\s*(\d+)", re.IGNORECASE), "numeric"),
    ("article", re.compile(r"Art\.\s*(\d+)", re.IGNORECASE), "numeric"),
    ("outline", re.compile(r"(\d+(?:\.\d+)*)\.?(?=\s)"), "outline"),
    ("section", re.compile(r"ust\.\s*(\d+)", re.IGNORECASE), "numeric"),
    ("point", re.compile(r"pkt\.\s*(\d+)", re.IGNORECASE), "numeric"),
    ("letter", re.compile(r"lit\.\s*([a-ząćęłńóśźż])", re.IGNORECASE), "letter"),
)


@dataclass(frozen=True)
class _Marker:
    offset: int
    path: tuple[int, ...]


def _unit_id(content_hash: str, start_offset: int, end_offset: int) -> str:
    return f"{content_hash}:{start_offset}:{end_offset}"


def _anchor(
    document: DocumentPayload,
    start_offset: int,
    end_offset: int,
) -> SourceAnchor:
    return SourceAnchor(
        start_offset=start_offset,
        end_offset=end_offset,
        read_mode=document.read_mode,
    )


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for index, character in enumerate(text):
        if character == "\n":
            starts.append(index + 1)
    return starts


def _markers_at_line_starts(
    text: str,
    pattern: Pattern[str],
    *,
    level_kind: str,
) -> list[_Marker]:
    markers: list[_Marker] = []
    for line_start in _line_starts(text):
        line = text[line_start:]
        match = pattern.match(line)
        if match is None:
            continue
        raw_value = match.group(1)
        if level_kind == "outline":
            path = tuple(int(part) for part in raw_value.split("."))
        elif level_kind == "letter":
            path = (ord(raw_value.lower()[0]),)
        else:
            path = (int(raw_value),)
        markers.append(_Marker(offset=line_start + match.start(), path=path))
    return markers


def _is_successor(previous: tuple[int, ...], following: tuple[int, ...]) -> bool:
    # A numbering that restarts at one opens a new numbered part: an annex or a
    # further regulation bundled in the same file. Chosen by argument, not measured.
    if following == (1,):
        return True
    if following == previous + (1,):
        return True
    for k in range(len(previous)):
        head = previous[:k] + (previous[k] + 1,)
        if following == head:
            return True
        # Increment at level k, then descend to first children: 1.2 -> 2.1.
        if (
            len(following) > len(head)
            and following[: len(head)] == head
            and all(part == 1 for part in following[len(head) :])
        ):
            return True
    return False


def _monotonicity_share(markers: list[_Marker]) -> float:
    if len(markers) < 2:
        return 1.0
    increasing = sum(
        1
        for left, right in zip(markers[:-1], markers[1:], strict=True)
        if _is_successor(left.path, right.path)
    )
    return increasing / (len(markers) - 1)


def _coverage_share(text: str, markers: list[_Marker]) -> float:
    if not text:
        return 0.0
    return (len(text) - markers[0].offset) / len(text)


def _tail_share(text: str, markers: list[_Marker]) -> float:
    # Share of the text after the last marker. The preamble before the first
    # marker is the coverage condition's business; uneven interior sections are
    # legitimate structure and are not penalised.
    if not text or not markers:
        return 1.0
    return (len(text) - markers[-1].offset) / len(text)


def _passes_structural_thresholds(
    text: str,
    markers: list[_Marker],
    config: RunConfig,
) -> bool:
    if len(markers) < config.structural_min:
        return False
    if _monotonicity_share(markers) < config.structural_monotonicity_share:
        return False
    if _coverage_share(text, markers) < config.structural_coverage_share:
        return False
    if _tail_share(text, markers) > config.structural_max_tail_share:
        return False
    return True


def _select_structural_markers(
    text: str,
    config: RunConfig,
) -> list[_Marker] | None:
    for _level_name, pattern, level_kind in _MARKER_LEVELS:
        markers = _markers_at_line_starts(text, pattern, level_kind=level_kind)
        if _passes_structural_thresholds(text, markers, config):
            return markers
    return None


def _split_sentences(text: str) -> list[tuple[int, int]]:
    sentences: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if text[start:end].strip():
            sentences.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text) and text[start:].strip():
        sentences.append((start, len(text)))
    return sentences


def _build_unit(
    document: DocumentPayload,
    start_offset: int,
    end_offset: int,
) -> Unit:
    return Unit(
        id=_unit_id(document.content_hash, start_offset, end_offset),
        text=document.text[start_offset:end_offset],
        anchor=_anchor(document, start_offset, end_offset),
    )


def _segment_structural(
    document: DocumentPayload,
    markers: list[_Marker],
) -> list[Unit]:
    units: list[Unit] = []
    for index, marker in enumerate(markers):
        start_offset = 0 if index == 0 else marker.offset
        next_offset = (
            markers[index + 1].offset
            if index + 1 < len(markers)
            else len(document.text)
        )
        units.append(_build_unit(document, start_offset, next_offset))
    return units


def _segment_window(document: DocumentPayload, config: RunConfig) -> list[Unit]:
    sentences = _split_sentences(document.text)
    if not sentences:
        raise StructureError("no_units", "document contains no extractable units")

    step = config.window_sentences - config.window_overlap
    if step <= 0:
        raise StructureError(
            "invalid_window", "window overlap must be smaller than window size"
        )

    units: list[Unit] = []
    start_index = 0
    while start_index < len(sentences):
        end_index = min(start_index + config.window_sentences, len(sentences))
        start_offset = sentences[start_index][0]
        end_offset = sentences[end_index - 1][1]
        units.append(_build_unit(document, start_offset, end_offset))
        if end_index == len(sentences):
            break
        start_index += step
    return units


def segment(document: DocumentPayload, config: RunConfig) -> list[Unit]:
    text = document.text
    if not text.strip():
        raise StructureError("empty_document", "document text is empty")

    markers = _select_structural_markers(text, config)
    if markers is not None:
        units = _segment_structural(document, markers)
    else:
        units = _segment_window(document, config)

    if not units:
        raise StructureError("no_units", "document contains no extractable units")
    return units


def _is_fallback_units(units: list[Unit]) -> bool:
    for left, right in zip(units, units[1:], strict=False):
        if left.anchor.end_offset > right.anchor.start_offset:
            return True
    return False
