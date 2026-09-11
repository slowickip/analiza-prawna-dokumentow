"""Guards on the seeder's own ingestion path.

Both checks existed only on the test-only builder in ``corpus``: the path the
suite exercised, not the one that fills the collection the artefact serves. A
manifest pinning a ``repeated_articles`` count that nothing verifies is worse
than no pin, and a struct path that silently resolves to two different nodes
drops a provision without saying so.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fpdf import FPDF

from contract_analyzer.corpus import CorpusBuildError, ManifestAct, load_manifest
from contract_analyzer.domain import ForceScope, ForceState, ForceValue
from font_paths import resolve_test_font
from seeder.eli import deduped_struct_article_paths, parse_pdf_units


def _act(repeated: tuple[dict[str, object], ...] = ()) -> ManifestAct:
    return ManifestAct(
        publisher="DU",
        year=2026,
        position=795,
        source_format="pdf",
        base_act="DU/1964/93",
        legal_status_date=date(2026, 5, 19),
        amendments_not_carried=(),
        repeated_articles=tuple(repeated),  # type: ignore[arg-type]
    )


def _force() -> ForceState:
    return ForceState(
        value=ForceValue.IN_FORCE,
        scope=ForceScope.ACT,
        source_locator="DU/1964/93",
        snapshot_date=date(2026, 5, 19),
    )


def _pdf(lines: list[str]) -> bytes:
    pdf = FPDF()
    pdf.add_font("DejaVu", "", resolve_test_font())
    pdf.set_font("DejaVu", size=12)
    pdf.add_page()
    for line in lines:
        pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def _arti(name: str, **extra: object) -> dict[str, object]:
    return {"type": "arti", "name": name, **extra}


def test_an_article_listed_whole_and_as_a_bare_heading_keeps_its_text() -> None:
    """ELI lists some articles twice, once with the subtree and once without.

    Measured on DU/2024/1513 on 2026-09-08: articles 3b, 10 and 24 each appear
    twice, agreeing in id, symbol, name, title and type, one copy carrying its
    paragraphs and the other omitting the key. Refusing that pair rejects the act
    for saying the same thing twice; keeping whichever arrived first drops the
    text of an article whose stub happened to come earlier. Both orders appear in
    that one act, so both are tested.
    """
    full = _arti(
        "3b",
        id="chpt_1-arti_3b",
        title="Art. 3b.",
        children=[{"type": "para", "name": "1"}],
    )
    stub = _arti("3b", id="chpt_1-arti_3b", title="Art. 3b.")

    for order in ([stub, full], [full, stub]):
        deduped = deduped_struct_article_paths(list(order), "DU/2024/1513")
        assert len(deduped) == 1
        assert deduped[0][1] == full


def test_an_empty_article_written_two_ways_is_one_article() -> None:
    """An empty subtree and a missing one are the same absence of text.

    The two spellings fingerprint differently, so they reach the ambiguity check,
    but neither copy holds a provision the other lacks. Refusing the act here
    would be a false alarm.
    """
    nodes = [
        _arti("7", id="chpt_1-arti_7", title="Art. 7.", children=[]),
        _arti("7", id="chpt_1-arti_7", title="Art. 7."),
    ]

    deduped = deduped_struct_article_paths(nodes, "DU/2024/1513")

    assert len(deduped) == 1


def test_two_articles_at_one_path_that_both_carry_text_stay_ambiguous() -> None:
    """The stub rule must not become a rule about which subtree is bigger.

    Two provisions at one address are the failure the ambiguity check exists to
    catch. Neither of these is a bare heading, so neither is a repeat of the
    other, and collapsing them would drop one without a word.
    """
    nodes = [
        _arti("3b", id="chpt_1-arti_3b", children=[{"type": "para", "name": "1"}]),
        _arti("3b", id="chpt_1-arti_3b", children=[{"type": "para", "name": "2"}]),
    ]
    with pytest.raises(CorpusBuildError) as caught:
        deduped_struct_article_paths(nodes, "DU/2024/1513")
    assert caught.value.code == "ambiguous_struct_path"


def test_a_bare_heading_that_disagrees_elsewhere_stays_ambiguous() -> None:
    """Missing a subtree is not on its own a licence to collapse.

    A repeat that also differs in a field ELI assigns -- here the id -- is not
    the same provision written twice, so the missing subtree does not make it a
    stub of the other.
    """
    nodes = [
        _arti("3b", id="chpt_1-arti_3b", children=[{"type": "para", "name": "1"}]),
        _arti("3b", id="chpt_2-arti_3b"),
    ]
    with pytest.raises(CorpusBuildError) as caught:
        deduped_struct_article_paths(nodes, "DU/2024/1513")
    assert caught.value.code == "ambiguous_struct_path"


def test_a_struct_path_that_names_two_different_nodes_is_ambiguous() -> None:
    """One path, two different nodes: which provision is art. 11 here?

    ELI repeats nodes, and an identical repeat is safe to collapse. Two nodes
    that merely share a path are not the same provision, and picking whichever
    came first drops the other from the corpus in silence.
    """
    nodes = [
        _arti("11", title="Pierwsza wersja"),
        _arti("11", title="Zupełnie inna treść"),
    ]
    with pytest.raises(CorpusBuildError) as caught:
        deduped_struct_article_paths(nodes, "DU/2023/725")
    assert caught.value.code == "ambiguous_struct_path"


def test_an_identical_repeat_is_still_collapsed_to_one_unit() -> None:
    """ELI does repeat nodes verbatim, and that is not an error."""
    nodes = [_arti("11", title="Ta sama treść"), _arti("11", title="Ta sama treść")]
    paths = deduped_struct_article_paths(nodes, "DU/2023/725")
    assert [path for path, _ in paths] == [("arti=11",)]


def test_a_fallback_does_not_check_a_hash_of_bytes_it_is_not_holding() -> None:
    """expected_hash pins what the reader fetched, and on a fallback that changed.

    The PDF reader checked it unconditionally, so an HTML act carrying a hash had
    it ignored while HTML answered and then compared against PDF bytes when the
    fallback ran -- a guaranteed source_hash_mismatch about the wrong document.
    """
    pdf = _pdf(["Art. 1. Pierwszy."])
    act = ManifestAct(
        publisher="DU",
        year=2024,
        position=1513,
        source_format="pdf",
        base_act="DU/2002/1204",
        legal_status_date=date(2026, 5, 19),
        amendments_not_carried=(),
        repeated_articles=(),
        expected_hash="0" * 64,
    )
    kwargs: dict[str, Any] = {
        "pdf_bytes": pdf,
        "act": act,
        "act_id": "DU/2024/1513",
        "act_url": "https://api.sejm.gov.pl/eli/acts/DU/2024/1513/text.pdf",
        "act_force": _force(),
        "seen_ids": set(),
    }
    with pytest.raises(CorpusBuildError) as caught:
        parse_pdf_units(**kwargs, pins_apply=True)
    assert caught.value.code == "source_hash_mismatch"

    units, _ = parse_pdf_units(**kwargs, pins_apply=False)
    assert len(units) == 1, "the fallback stopped on a pin about other bytes"


def test_a_manifest_may_not_pin_a_hash_on_an_act_it_reads_as_html() -> None:
    """The schema is the first lock: an HTML act has no one document to hash."""
    raw = {
        "version": 2,
        "corpus_target_date": "2026-08-30",
        "acts": [
            {
                "publisher": "DU",
                "year": 2024,
                "position": 1513,
                "source_format": "html",
                "base_act": "DU/2002/1204",
                "legal_status_date": "2024-10-09",
                "amendments_not_carried": [],
                "repeated_articles": [],
                "expected_hash": "0" * 64,
            }
        ],
    }
    path = Path(tempfile.mkdtemp()) / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as caught:
        load_manifest(path)
    assert caught.value.code == "expected_hash_not_applicable"


def test_a_fallback_records_the_repeats_it_found_instead_of_meeting_a_pin() -> None:
    """An act declaring HTML has no pin its PDF could have met.

    _validate_repeated_articles_applicable rejects any non-empty
    repeated_articles on an act declaring html, so the empty tuple such an act
    carries is the only value the schema allows and asserts nothing about its
    PDF. Comparing against it reported drift from a claim nobody made, and one
    act -- DU/2024/1513, whose PDF prints 3b, 10, 24 and 25 twice -- stopped the
    whole eleven-act build that way while ELI's HTML was down.
    """
    pdf = _pdf(["Art. 1. Pierwszy.", "Art. 2. Drugi.", "Art. 2. Drugi ponownie."])
    url = "https://api.sejm.gov.pl/eli/acts/DU/2024/1513/text.pdf"

    with pytest.raises(CorpusBuildError) as caught:
        parse_pdf_units(
            pdf_bytes=pdf,
            act=_act(),
            act_id="DU/2024/1513",
            act_url=url,
            act_force=_force(),
            seen_ids=set(),
            pins_apply=True,
        )
    assert caught.value.code == "repeated_article_drift"

    units, repeated = parse_pdf_units(
        pdf_bytes=pdf,
        act=_act(),
        act_id="DU/2024/1513",
        act_url=url,
        act_force=_force(),
        seen_ids=set(),
        pins_apply=False,
    )
    assert [r.article for r in repeated] == ["2"], repeated
    assert [r.printings for r in repeated] == [2]
    assert len(units) == 3, "the fallback still builds every printing"


def test_only_articles_become_units_not_whatever_leaf_the_struct_ends_on() -> None:
    """ELI's structure carries more than the act's articles, and the walk took it.

    Any recognised segment with no children used to be appended as a unit, so a
    childless ``pass``, ``pint`` or ``lett`` reached the corpus beside the
    articles. On the published eleven-act snapshot that was 17 units across the
    three HTML acts: the publisher's announcement, its list of carried
    amendments -- whose '1)', '2)', '3)' repeat under two different ``pass``
    parents, so one identifier named two provisions -- and annex fragments.

    None of it is operative law and the PDF reader never had it, so including it
    made one manifest describe two different corpora depending on which source format
    answered.
    """
    nodes: list[dict[str, object]] = [
        {
            "type": "pass",
            "name": "1",
            # The announcement: real text, not a provision of the act.
            "title": "Na podstawie art. 16 ust. 1 ... oglasza sie",
            "children": [
                {
                    "type": "pint",
                    "name": "1",
                    "title": "1) ustawa z dnia 24 marca 2022 r.",
                },
            ],
        },
        {
            "type": "chpt",
            "name": "1",
            "children": [
                _arti("11", title="Art. 11."),
                _arti("3a_1", title="Art. 3a1."),
            ],
        },
        {"type": "lett", "name": "a", "title": "a) prosze wpisac"},
    ]
    paths = [path for path, _ in deduped_struct_article_paths(nodes, "DU/2023/725")]
    assert paths == [("chpt=1", "arti=11"), ("chpt=1", "arti=3a_1")], paths


def test_an_article_naming_a_range_is_still_an_article() -> None:
    """'Art. 20-22. (uchylone)' is how ELI models a repealed run, and it is an
    arti node. Dropping every short or oddly-named unit would have taken it."""
    nodes = [_arti("20-22", title="Art. 20-22.")]
    assert [p for p, _ in deduped_struct_article_paths(nodes, "DU/2024/1513")] == [
        ("arti=20-22",)
    ]


def test_a_pdf_that_repeats_an_article_the_manifest_did_not_pin_is_drift() -> None:
    """The manifest pins a provenance fact; the document decides whether it holds."""
    pdf = _pdf(
        [
            "Art. 184. Pierwsze wydrukowanie.",
            "Art. 184. Drugie wydrukowanie.",
            "Art. 185. Kolejny przepis.",
        ]
    )
    with pytest.raises(CorpusBuildError) as caught:
        parse_pdf_units(
            pdf_bytes=pdf,
            act=_act(),  # pins nothing
            act_id="DU/2026/795",
            act_url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
            act_force=_force(),
            seen_ids=set(),
        )
    assert caught.value.code == "repeated_article_drift"


def test_a_pinned_repeat_that_matches_the_document_is_accepted() -> None:
    pdf = _pdf(
        [
            "Art. 184. Pierwsze wydrukowanie.",
            "Art. 184. Drugie wydrukowanie.",
            "Art. 185. Kolejny przepis.",
        ]
    )
    units, _ = parse_pdf_units(
        pdf_bytes=pdf,
        act=_act(({"article": "184", "printings": 2},)),
        act_id="DU/2026/795",
        act_url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
        act_force=_force(),
        seen_ids=set(),
    )
    assert len(units) == 3


def test_a_pinned_repeat_the_document_does_not_show_is_drift() -> None:
    """Drift runs both ways: a pin the document stopped supporting is also wrong."""
    pdf = _pdf(["Art. 184. Jedyne wydrukowanie.", "Art. 185. Kolejny przepis."])
    with pytest.raises(CorpusBuildError) as caught:
        parse_pdf_units(
            pdf_bytes=pdf,
            act=_act(({"article": "184", "printings": 2},)),
            act_id="DU/2026/795",
            act_url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
            act_force=_force(),
            seen_ids=set(),
        )
    assert caught.value.code == "repeated_article_drift"


def test_the_closing_article_stops_where_the_annexes_begin() -> None:
    """Every other article is bounded by the next heading; the last one was not.

    It ran to end of file and swallowed the footnotes, the page furniture and
    then whole appendices. Measured over the eleven-act manifest's eight PDF
    acts, two closing articles were 21,256 and 10,984 characters -- against a
    median article of a few hundred -- and both were mostly annex.
    """
    pdf = _pdf(
        [
            "Art. 1. Pierwszy przepis.",
            "Art. 2. Ustawa wchodzi w życie po upływie 14 dni.",
            "Załącznik nr 1",
            "FORMULARZ INFORMACYJNY",
            "Treść formularza, która nie jest przepisem.",
        ]
    )
    units, _ = parse_pdf_units(
        pdf_bytes=pdf,
        act=_act(),
        act_id="DU/2026/795",
        act_url="https://api.sejm.gov.pl/eli/acts/DU/2026/795",
        act_force=_force(),
        seen_ids=set(),
    )
    assert [u.article_identifier for u in units] == ["Art. 1", "Art. 2"]
    closing = units[-1].text
    assert "wchodzi w życie" in closing
    assert "FORMULARZ" not in closing, "the annex was swept into the closing article"
    assert "Załącznik" not in closing


def test_an_act_without_annexes_keeps_its_closing_article_whole() -> None:
    """The six acts with no annex after the last heading must not change."""
    pdf = _pdf(
        [
            "Art. 1. Pierwszy przepis.",
            "Art. 2. Ustawa wchodzi w życie po upływie 14 dni.",
            "2) Przypis do artykułu drugiego.",
        ]
    )
    units, _ = parse_pdf_units(
        pdf_bytes=pdf,
        act=_act(),
        act_id="DU/2026/795",
        act_url="https://api.sejm.gov.pl/eli/acts/DU/2026/795",
        act_force=_force(),
        seen_ids=set(),
    )
    assert "Przypis do artykułu drugiego" in units[-1].text
