from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest

from contract_analyzer.config import RunConfig
from contract_analyzer.domain import (
    DocumentPayload,
    ReadMode,
    ReferenceRecord,
    ReferenceStatus,
    ReferenceType,
    SourceAnchor,
    Unit,
)
from contract_analyzer.structure import (
    StructureError,
    _split_sentences,
    context_for,
    parse_references,
    segment,
)
from contract_analyzer.structure.segmentation import _is_fallback_units


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document(text: str) -> DocumentPayload:
    return DocumentPayload(
        document_id=uuid4(),
        text=text,
        anchors=(SourceAnchor(start_offset=0, end_offset=len(text)),),
        read_mode=ReadMode.NATIVE_PDF,
        content_hash=_content_hash(text),
    )


def _force_fallback_config() -> RunConfig:
    return RunConfig(structural_min=99)


def _sentence_boundaries(text: str) -> list[tuple[int, int]]:
    return _split_sentences(text)


def _unit_sentence_ranges(units: list[Unit], text: str) -> list[tuple[int, int]]:
    sentences = _sentence_boundaries(text)
    ranges: list[tuple[int, int]] = []
    for unit in units:
        start_idx = next(
            i
            for i, (start, _) in enumerate(sentences)
            if start == unit.anchor.start_offset
        )
        end_idx = next(
            i for i, (_, end) in enumerate(sentences) if end == unit.anchor.end_offset
        )
        ranges.append((start_idx, end_idx + 1))
    return ranges


def _single_unit(
    text: str,
    *,
    unit_id: str = "u1",
    start_offset: int = 0,
) -> Unit:
    end_offset = start_offset + len(text)
    return Unit(
        id=unit_id,
        text=text,
        anchor=SourceAnchor(start_offset=start_offset, end_offset=end_offset),
    )


def _parse_one(text: str, units: list[Unit] | None = None) -> ReferenceRecord:
    if units is None:
        resolved_units = [_single_unit(text)]
    else:
        resolved_units = units
    records = parse_references(resolved_units)
    citing_id = resolved_units[-1].id
    matches = [record for record in records if record.citing_unit_id == citing_id]
    assert len(matches) == 1, records
    return matches[0]


@pytest.fixture
def six_sentence_document() -> DocumentPayload:
    text = "Pierwsze. Drugie. Trzecie. Czwarte. Piąte. Szóste."
    return _document(text)


def test_segment_raises_on_empty_document() -> None:
    document = _document("")
    with pytest.raises(StructureError, match="empty"):
        segment(document, RunConfig())


def test_segment_raises_when_no_units_extractable() -> None:
    document = _document("   \n\t  ")
    with pytest.raises(StructureError):
        segment(document, RunConfig())


def test_structural_segmentation_on_paragraph_markers() -> None:
    text = (
        "§ 1. Przedmiot umowy\n"
        "Treść pierwszego paragrafu.\n\n"
        "§ 2. Czynsz\n"
        "Treść drugiego paragrafu.\n\n"
        "§ 3. Okres najmu\n"
        "Treść trzeciego paragrafu."
    )
    units = segment(_document(text), RunConfig())
    assert len(units) == 3
    assert units[0].text.startswith("§ 1.")
    assert units[1].text.startswith("§ 2.")
    assert units[2].text.startswith("§ 3.")
    hash_prefix = _content_hash(text) + ":"
    assert all(unit.id.startswith(hash_prefix) for unit in units)


def test_top_level_prefers_paragraph_over_nested_sections() -> None:
    text = (
        "§ 1. Nagłówek\n"
        "ust. 1. Sekcja A.\n"
        "ust. 2. Sekcja B.\n"
        "ust. 3. Sekcja C.\n\n"
        "§ 2. Nagłówek\n"
        "ust. 1. Sekcja A.\n"
        "ust. 2. Sekcja B.\n"
        "ust. 3. Sekcja C.\n\n"
        "§ 3. Nagłówek\n"
        "ust. 1. Sekcja A.\n"
        "ust. 2. Sekcja B.\n"
        "ust. 3. Sekcja C."
    )
    units = segment(_document(text), RunConfig())
    assert len(units) == 3
    assert all(unit.text.startswith("§") for unit in units)


def test_ocr_section_sign_variants_are_counted(
    six_sentence_document: DocumentPayload,
) -> None:
    text = (
        "$ 1. Pierwszy.\n\nS 2. Drugi.\n\n5 3. Trzeci.\n\n8 4. Czwarty.\n\n§ 5. Piąty."
    )
    units = segment(_document(text), RunConfig(structural_min=3))
    assert len(units) == 5


def test_stray_marker_without_sequence_does_not_create_unit() -> None:
    text = (
        "W tekście pada kwota 5 000 zł.\n\n"
        "§ 1. Pierwszy paragraf.\n\n"
        "§ 2. Drugi paragraf.\n\n"
        "§ 3. Trzeci paragraf."
    )
    units = segment(_document(text), RunConfig())
    assert len(units) == 3
    assert "§ 1." in units[0].text
    assert units[1].text.startswith("§ 2.")
    assert units[2].text.startswith("§ 3.")


def test_fallback_is_three_sentences_with_one_sentence_overlap(
    six_sentence_document: DocumentPayload,
) -> None:
    units = segment(six_sentence_document, _force_fallback_config())
    assert _unit_sentence_ranges(units, six_sentence_document.text) == [
        (0, 3),
        (2, 5),
        (4, 6),
    ]


_POLISH_CONTRACT_TEXT = (
    "Umowa zawarta pomiedzy Kowalski Sp. z o.o. a Nowak S.A. w Warszawie. "
    "Strony ustalaja, ze zaplata nastapi zgodnie z art. 488 par. 1 k.c. "
    "w terminie 14 dni. Wynajmujacy zastrzega sobie prawo m.in. w przypadku "
    "opoznienia. Najemca zobowiazuje sie do utrzymania lokalu w nalezytym "
    "stanie. Postanowienia niniejszej umowy podlegaja prawu polskiemu."
)


def test_fallback_splits_polish_contract_into_five_sentences() -> None:
    sentences = [
        _POLISH_CONTRACT_TEXT[start:end]
        for start, end in _split_sentences(_POLISH_CONTRACT_TEXT)
    ]
    assert len(sentences) == 5
    assert "Sp. z o.o." in sentences[0]
    assert "S.A." in sentences[0]
    assert "art. 488 par. 1 k.c." in sentences[1]
    assert "m.in." in sentences[2]


@pytest.mark.parametrize(
    "phrase",
    [
        "Kowalski Sp. z o.o. a Nowak",
        "partnerem S.A. w Warszawie",
        "zgodnie z art. 488 par. 1 k.c. w terminie",
        "prawo m.in. w przypadku",
    ],
)
def test_fallback_keeps_polish_abbreviation_inside_one_sentence(phrase: str) -> None:
    text = f"{phrase}. Kolejne zdanie zaczyna sie tutaj."
    sentences = [text[start:end] for start, end in _split_sentences(text)]
    assert len(sentences) == 2
    assert phrase in sentences[0]


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("zgodnie z § 5 ust. 2", ReferenceStatus.RESOLVED),
        ("o którym mowa w ust. 1", ReferenceStatus.WITHIN_UNIT),
        ("jak wskazano powyżej", ReferenceStatus.AMBIGUOUS_DIRECTIONAL),
        ("załącznik nr 1", ReferenceStatus.OUTSIDE_INPUT),
        ("niniejsza umowa", ReferenceStatus.WHOLE_DOCUMENT),
        ("art. 659 Kodeksu cywilnego", ReferenceStatus.EXTERNAL_ACT),
    ],
)
def test_closed_reference_catalogue(text: str, status: ReferenceStatus) -> None:
    if status is ReferenceStatus.RESOLVED:
        units = [
            _single_unit("§ 5. Treść paragrafu.", unit_id="p5", start_offset=0),
            _single_unit(text, unit_id="cite", start_offset=100),
        ]
        record = next(r for r in parse_references(units) if r.citing_unit_id == "cite")
    else:
        record = _parse_one(text)
    assert record.status is status


def test_within_unit_inside_paragraph_unit() -> None:
    unit = _single_unit("§ 1. Postanowienia ogólne o którym mowa w ust. 1.")
    records = parse_references([unit])
    within = [r for r in records if r.status is ReferenceStatus.WITHIN_UNIT]
    assert len(within) == 1
    assert within[0].reference_type is ReferenceType.RELATIVE_INTERNAL
    assert within[0].raw_text == "ust. 1"


def test_no_references_when_units_have_only_headings() -> None:
    text = "\n\n".join(f"§ {i}. Treść paragrafu." for i in range(1, 8))
    units = segment(_document(text), RunConfig())
    assert len(units) == 7
    assert parse_references(units) == []


@pytest.mark.parametrize(
    "text",
    [
        "niniejsza umowa",
        "niniejszej umowy",
        "niniejsza umowę",
        "niniejszej umowie",
    ],
)
def test_whole_document_inflected_forms(text: str) -> None:
    record = _parse_one(text)
    assert record.status is ReferenceStatus.WHOLE_DOCUMENT


def test_target_does_not_exist_for_missing_internal_target() -> None:
    text = "zgodnie z § 99 ust. 2"
    record = _parse_one(
        text,
        [
            _single_unit("§ 1. Treść.", unit_id="p1", start_offset=0),
            _single_unit("§ 2. Inna treść.", unit_id="p2", start_offset=50),
            _single_unit(text, unit_id="cite", start_offset=100),
        ],
    )
    assert record.status is ReferenceStatus.TARGET_DOES_NOT_EXIST
    assert record.reference_type is ReferenceType.FULL_INTERNAL


def test_fallback_internal_reference_is_target_does_not_exist(
    six_sentence_document: DocumentPayload,
) -> None:
    units = segment(six_sentence_document, _force_fallback_config())
    citing = Unit(
        id="cite",
        text="zgodnie z § 5 ust. 2",
        anchor=SourceAnchor(start_offset=0, end_offset=20),
    )
    records = parse_references([*units, citing])
    internal = [r for r in records if r.citing_unit_id == "cite"]
    assert len(internal) == 1
    assert internal[0].status is ReferenceStatus.TARGET_DOES_NOT_EXIST


def test_context_for_returns_only_depth_one_targets() -> None:
    units = [
        Unit(id="a", text="A", anchor=SourceAnchor(start_offset=0, end_offset=1)),
        Unit(id="b", text="B", anchor=SourceAnchor(start_offset=1, end_offset=2)),
        Unit(id="c", text="C", anchor=SourceAnchor(start_offset=2, end_offset=3)),
    ]
    references = [
        ReferenceRecord(
            citing_unit_id="a",
            reference_type=ReferenceType.FULL_INTERNAL,
            status=ReferenceStatus.RESOLVED,
            raw_text="§ 2",
            target_unit_id="b",
        ),
        ReferenceRecord(
            citing_unit_id="b",
            reference_type=ReferenceType.FULL_INTERNAL,
            status=ReferenceStatus.RESOLVED,
            raw_text="§ 3",
            target_unit_id="c",
        ),
    ]
    assert context_for("a", units, references) == [units[1]]


def test_context_for_cycle_terminates_without_recursion() -> None:
    units = [
        Unit(id="a", text="A", anchor=SourceAnchor(start_offset=0, end_offset=1)),
        Unit(id="b", text="B", anchor=SourceAnchor(start_offset=1, end_offset=2)),
    ]
    references = [
        ReferenceRecord(
            citing_unit_id="a",
            reference_type=ReferenceType.FULL_INTERNAL,
            status=ReferenceStatus.RESOLVED,
            raw_text="§ 2",
            target_unit_id="b",
        ),
        ReferenceRecord(
            citing_unit_id="b",
            reference_type=ReferenceType.FULL_INTERNAL,
            status=ReferenceStatus.RESOLVED,
            raw_text="§ 1",
            target_unit_id="a",
        ),
    ]
    assert context_for("a", units, references) == [units[1]]


def test_context_for_self_reference_returns_nothing() -> None:
    unit = _single_unit("o którym mowa w ust. 1", unit_id="self")
    records = parse_references([unit])
    assert context_for("self", [unit], records) == []


def test_context_for_ignores_unresolved_statuses() -> None:
    unit = _single_unit("jak wskazano powyżej", unit_id="u1")
    records = parse_references([unit])
    assert context_for("u1", [unit], records) == []


def test_context_for_returns_empty_list_in_fallback_mode(
    six_sentence_document: DocumentPayload,
) -> None:
    units = segment(six_sentence_document, _force_fallback_config())
    citing = Unit(
        id="cite",
        text="zgodnie z § 1",
        anchor=SourceAnchor(start_offset=0, end_offset=12),
    )
    records = parse_references([*units, citing])
    assert context_for("cite", [*units, citing], records) == []


def test_segment_and_parse_are_deterministic(
    six_sentence_document: DocumentPayload,
) -> None:
    config = RunConfig()
    first_units = segment(six_sentence_document, config)
    second_units = segment(six_sentence_document, config)
    assert first_units == second_units

    first_refs = parse_references(first_units)
    second_refs = parse_references(second_units)
    assert first_refs == second_refs


def test_outline_document_structural_segmentation() -> None:
    text = (
        "1. Postanowienia ogólne\n"
        "Treść punktu pierwszego.\n\n"
        "1.1 Zakres umowy\n"
        "Treść podpunktu jeden jeden.\n\n"
        "1.2 Zasady współpracy\n"
        "Treść podpunktu jeden dwa.\n\n"
        "2. Definicje\n"
        "Treść definicji.\n\n"
        "2.1 Wynagrodzenie zasadnicze\n"
        "Treść podpunktu dwa jeden.\n\n"
        "2.2 Terminy płatności\n"
        "Treść podpunktu dwa dwa.\n\n"
        "3. Czas trwania umowy\n"
        "Treść punktu trzeciego."
    )
    units = segment(_document(text), RunConfig())
    assert len(units) == 7
    assert units[0].text.startswith("1.")
    assert units[1].text.startswith("1.1")
    assert units[2].text.startswith("1.2")
    assert units[3].text.startswith("2.")
    assert units[4].text.startswith("2.1")
    assert units[5].text.startswith("2.2")
    assert units[6].text.startswith("3.")


@pytest.mark.parametrize(
    ("previous", "following", "expected"),
    [
        # Descend to first child
        ((1,), (1, 1), True),
        ((1, 1), (1, 1, 1), True),
        ((2, 3), (2, 3, 1), True),
        # Same-level increment
        ((1,), (2,), True),
        ((1, 1), (1, 2), True),
        ((2, 3, 4), (2, 3, 5), True),
        # Increment at shallower level, dropping deeper levels
        ((1, 2), (2,), True),
        ((2, 3, 4), (2, 4), True),
        ((2, 3, 4), (3,), True),
        ((1, 1, 1, 1), (2,), True),
        # Negative: skip by two
        ((1,), (3,), False),
        ((1, 1), (1, 3), False),
        ((2, 3, 4), (2, 5), False),
        ((2, 3, 4), (4,), False),
        # Negative: restart without parent increment
        ((1, 2), (1, 1), False),
        ((2, 3), (2, 1), False),
        # Restart at one opens a new numbered part (an annex, a further regulation)
        ((2,), (1,), True),
        # Negative: going back
        ((3,), (2,), False),
        # Negative: invalid descend (not first child)
        ((1,), (1, 2), False),
        ((1,), (1, 1, 1), False),
        # Negative: jump to different subtree
        ((1, 2), (2, 2), False),
    ],
)
def test_is_successor(
    previous: tuple[int, ...], following: tuple[int, ...], expected: bool
) -> None:
    from contract_analyzer.structure import _is_successor

    assert _is_successor(previous, following) is expected


def test_paragraph_document_prefers_paragraph_over_outline() -> None:
    text = (
        "§ 1. Postanowienia ogólne\n"
        "1. Wynajmujący oddaje lokal.\n"
        "2. Najemca lokal przyjmuje.\n\n"
        "§ 2. Czynsz\n"
        "1. Czynsz wynosi 4500 zł.\n"
        "2. Płatność do 10 dnia.\n\n"
        "§ 3. Kaucja\n"
        "1. Kaucja wynosi 9000 zł.\n"
        "2. Zwrot w terminie 30 dni."
    )
    units = segment(_document(text), RunConfig())
    assert len(units) == 3
    assert all(unit.text.startswith("§") for unit in units)


def test_outline_unit_resolves_dotted_point_reference() -> None:
    text = (
        "1. Postanowienia ogólne\n"
        "Strony ustalają zasady zgodnie z pkt 2.1 umowy.\n\n"
        "2. Wynagrodzenie\n"
        "Zasady płatności określone poniżej.\n\n"
        "2.1. Stawka czynszu\n"
        "Miesięczny czynsz wynosi 5000 zł.\n\n"
        "3. Postanowienia końcowe\n"
        "Umowa sporządzona w dwóch egzemplarzach."
    )
    units = segment(_document(text), RunConfig())
    records = parse_references(units)
    point_refs = [r for r in records if "pkt 2.1" in r.raw_text]
    assert len(point_refs) == 1
    record = point_refs[0]
    assert record.reference_type is ReferenceType.FULL_INTERNAL
    assert record.status is ReferenceStatus.RESOLVED
    assert record.target_unit_id is not None
    target_unit = next(u for u in units if u.id == record.target_unit_id)
    assert target_unit.text.startswith("2.1")
    context = context_for(units[0].id, units, records)
    assert target_unit in context


def test_paragraph_document_dotted_point_yields_target_does_not_exist() -> None:
    text = (
        "§ 1. Postanowienia ogólne\n"
        "Strony postępują zgodnie z pkt 2.1 niniejszej umowy.\n\n"
        "§ 2. Wynagrodzenie\n"
        "Czynsz najmu.\n\n"
        "§ 3. Kaucja\n"
        "Kaucja zabezpieczająca."
    )
    units = segment(_document(text), RunConfig())
    records = parse_references(units)
    point_refs = [r for r in records if "pkt 2.1" in r.raw_text]
    assert len(point_refs) == 1
    assert point_refs[0].status is ReferenceStatus.TARGET_DOES_NOT_EXIST
    assert point_refs[0].target_unit_id is None


def test_date_like_line_in_paragraph_document_does_not_change_segmentation() -> None:
    text = (
        "12.05.2026 w Warszawie\n\n"
        "§ 1. Przedmiot umowy\n"
        "Treść pierwszego paragrafu.\n\n"
        "§ 2. Czynsz\n"
        "Treść drugiego paragrafu.\n\n"
        "§ 3. Okres najmu\n"
        "Treść trzeciego paragrafu."
    )
    units = segment(_document(text), RunConfig())
    assert len(units) == 3
    assert units[0].text.startswith("12.05.2026")
    assert "§ 1." in units[0].text
    assert units[1].text.startswith("§ 2.")
    assert units[2].text.startswith("§ 3.")


def test_uneven_section_lengths_still_segment_structurally() -> None:
    long_body = " ".join(f"Postanowienie szczegółowe numer {i}." for i in range(40))
    text = (
        "§ 1. Krótki paragraf.\n\n"
        "§ 2. Krótki paragraf.\n\n"
        "§ 3. " + long_body + "\n\n"
        "§ 4. Krótki paragraf.\n\n"
        "§ 5. Krótki paragraf."
    )
    units = segment(_document(text), RunConfig())
    assert [unit.text[:4] for unit in units] == ["§ 1.", "§ 2.", "§ 3.", "§ 4.", "§ 5."]


def test_marker_run_followed_by_unmarked_bulk_falls_back() -> None:
    tail = " ".join(f"Zdanie bez znacznika numer {i}." for i in range(60))
    text = "§ 1. Pierwszy.\n\n§ 2. Drugi.\n\n§ 3. Trzeci. " + tail
    units = segment(_document(text), RunConfig())
    assert _is_fallback_units(units)


def test_numbering_restart_at_one_opens_a_new_part() -> None:
    text = (
        "§ 1. Umowa główna.\n\n"
        "§ 2. Czynsz.\n\n"
        "§ 3. Kaucja.\n\n"
        "§ 1. Załącznik pierwszy.\n\n"
        "§ 2. Załącznik drugi.\n\n"
        "§ 3. Załącznik trzeci."
    )
    units = segment(_document(text), RunConfig())
    assert len(units) == 6
    assert all(unit.text.startswith("§") for unit in units)


def test_par_headings_do_not_open_structural_units() -> None:
    """``par.`` is a citation form, not a heading marker: such text falls back."""
    text = (
        "par. 1 Przedmiot umowy. Treść pierwszego paragrafu.\n\n"
        "par. 2 Czynsz. Treść drugiego paragrafu.\n\n"
        "par. 3 Okres najmu. Treść trzeciego paragrafu."
    )
    units = segment(_document(text), RunConfig())
    assert not any(unit.text.startswith("par. 2") for unit in units)
    assert _is_fallback_units(units)


def test_ocr_confused_section_sign_is_not_a_reference_target() -> None:
    """A unit opening ``S 5`` counts as a heading for segmentation, never as § 5."""
    text = "zgodnie z § 5"
    record = _parse_one(
        text,
        [
            _single_unit("§ 1. Treść.", unit_id="p1", start_offset=0),
            _single_unit("S 5. Treść paragrafu.", unit_id="p5", start_offset=50),
            _single_unit(text, unit_id="cite", start_offset=100),
        ],
    )
    assert record.status is ReferenceStatus.TARGET_DOES_NOT_EXIST


def test_reference_free_document_never_parses_its_headings_as_numbers() -> None:
    """An indented over-long numeric preamble is left alone without a reference."""
    text = (
        "  " + "1" * 5000 + "\n\n"
        "Art. 1. " + "Treść pierwszego artykułu. " * 20 + "\n\n"
        "Art. 2. " + "Treść drugiego artykułu. " * 20 + "\n\n"
        "Art. 3. " + "Treść trzeciego artykułu. " * 20
    )
    units = segment(_document(text), RunConfig())
    assert parse_references(units) == []
