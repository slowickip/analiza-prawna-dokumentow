from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pytest
from fpdf import FPDF
from pdfplumber.utils.text import extract_text as _extract_chars_text

from contract_analyzer import act_pdf
from contract_analyzer.corpus import (
    CorpusBuildError,
    CorpusManifest,
    LegalUnit,
    QdrantCorpusIndex,
    _is_editorial_repeal_placeholder,
    provision_force_state,
    unit_digest_of,
)
from contract_analyzer.domain import ForceScope, ForceValue
from font_paths import resolve_test_font
from seeder.eli import deduped_struct_article_paths
from seeder.ingest import fetch_manifest_units


def _digest_of(units: list[LegalUnit]) -> str:
    """The corpus digest, which is a property of the units and not of a store."""
    return unit_digest_of((unit.locator, unit.content_hash) for unit in units)


def _pdf_char(
    text: str,
    *,
    size: float,
    x0: float,
    x1: float,
    y0: float,
) -> dict[str, float | str | bool | int]:
    return {
        "text": text,
        "size": size,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y0 + size,
        "top": y0,
        "bottom": y0 + size,
        "doctop": y0,
        "width": x1 - x0,
        "height": size,
        "upright": True,
        "direction": 1,
    }


def _extract_marked_pdf_text(
    chars: list[dict[str, float | str | bool | int]],
) -> str:
    return _extract_chars_text(act_pdf.mark_superscript_digits(chars)) or ""


def _pdf_test_manifest(**act_overrides: object) -> dict[str, object]:
    act = {
        "publisher": "DU",
        "year": 2026,
        "position": 795,
        "source_format": "pdf",
        "base_act": "DU/1964/93",
        "legal_status_date": "2026-05-19",
        "amendments_not_carried": [],
        "repeated_articles": [],
        **act_overrides,
    }
    return {
        "version": 2,
        "corpus_target_date": "2026-08-30",
        "acts": [act],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_real_structure_pdf() -> bytes:
    """Render the real DU/2026/795 structure that broke the builder.

    The announcement preamble quotes amending-act provisions inside ``„...„``
    (including an article heading at line start inside a quote), the code body
    cites its own earlier articles mid-line, and it carries superscripted
    articles rendered by pdfplumber as ``Art. 16[1].``.
    """
    lines = [
        "DZIENNIK USTAW",
        "RZECZYPOSPOLITEJ POLSKIEJ",
        "Poz. 795",
        "OBWIESZCZENIE",
        "MARSZAŁKA SEJMU RZECZYPOSPOLITEJ POLSKIEJ",
        "z dnia 27 maja 2026 r.",
        "w sprawie ogłoszenia jednolitego tekstu ustawy – Kodeks cywilny",
        "2. Tekst jednolity ustawy nie obejmuje:",
        "1) art. 13 i art. 16 ustawy z dnia 5 sierpnia 2025 r., które stanowią:",
        "„Art. 13. Do umów zawartych przed dniem wejścia w życie niniejszego",
        "przepisu stosuje się przepisy ustawy zmienianej w brzmieniu dotychczasowym.”",
        "„Art. 16. Ustawa wchodzi w życie z dniem 1 marca 2026 r., z wyjątkiem:",
        "1) art. 1 pkt 5 w zakresie art. 1251 § 2, który wchodzi w życie z dniem",
        "2) art. 1 pkt 9, 10, pkt 11 lit. c, pkt 14, 16–18, 21–23 i 35, art. 7,",
        "4) art. 2, art. 9 i art. 13, które wchodzą w życie po upływie 3 miesięcy”.",
        "2) art. 2 i art. 3 ustawy z dnia 9 października 2025 r., które stanowią:",
        "„Art. 2. Do umów zawartych przed dniem wejścia w życie niniejszej ustawy",
        "stosuje się przepisy art. 6471 ustawy zmienianej w art. 1.",
        "Art. 3. Ustawa wchodzi w życie po upływie miesiąca od dnia ogłoszenia.”;",
        "3) art. 3 i art. 4 ustawy z dnia 23 stycznia 2026 r., które stanowią:",
        "„Art. 3. Do rzeczy znalezionych przed dniem wejścia w życie niniejszej",
        "ustawy stosuje się przepisy dotychczasowe.",
        "Art. 4. Ustawa wchodzi w życie po upływie 3 miesięcy od dnia ogłoszenia.”;",
        "Załącznik do obwieszczenia Marszałka Sejmu",
        "USTAWA",
        "z dnia 23 kwietnia 1964 r.",
        "Kodeks cywilny",
        "Art. 1. Kodeks niniejszy reguluje stosunki cywilnoprawne między osobami",
        "fizycznymi i osobami prawnymi.",
        "Art. 2. (uchylony)",
        "Art. 3. Ustawa nie ma mocy wstecznej, chyba że to wynika z jej brzmienia.",
        "Zgodnie z art. 16 ustawy z dnia 5 sierpnia 2025 r. stosuje się przepisy",
        "dotychczasowe.",
        "Art. 4. (uchylony)",
        "Art. 5. Nie można czynić ze swego prawa użytku, który by był sprzeczny",
        "z zasadami współżycia społecznego.",
        "Art. 15. Ograniczoną zdolność do czynności prawnych mają małoletni, którzy",
        "ukończyli lat trzynaście, oraz osoby ubezwłasnowolnione częściowo.",
        "Art. 16. § 1. Osoba pełnoletnia może być ubezwłasnowolniona częściowo",
        "z powodu choroby psychicznej, niedorozwoju umysłowego albo innego rodzaju",
        "zaburzeń psychicznych.",
        "§ 2. Dla osoby ubezwłasnowolnionej częściowo ustanawia się kuratelę.",
        "Art. 16[1]. Przepisy o ubezwłasnowolnieniu stosuje się odpowiednio.",
        "Art. 17. Z zastrzeżeniem wyjątków w ustawie przewidzianych, do ważności",
        "czynności prawnej, przez którą osoba ograniczona w zdolności do czynności",
        "prawnych zaciąga zobowiązanie, potrzebna jest zgoda przedstawiciela.",
        "Art. 353[1]. Strony zawierające umowę mogą ułożyć stosunek prawny według",
        "swego uznania, byleby jego treść lub cel nie sprzeciwiały się właściwości",
        "stosunku, ustawie ani zasadom współżycia społecznego.",
        "Art. 354. Dłużnik powinien wykonać zobowiązanie zgodnie z jego treścią",
        "i w sposób odpowiadający jego celowi.",
    ]
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("FixtureFont", "", resolve_test_font())
    pdf.set_font("FixtureFont", size=10)
    for line in lines:
        if not line:
            pdf.ln(4)
            continue
        pdf.multi_cell(w=0, h=5, text=line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def test_empty_200_aborts_build(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=11",
        status=200,
        body=b"",
    )
    with pytest.raises(CorpusBuildError, match="empty_source_response"):
        fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]


def test_identical_struct_duplicate_produces_one_unit(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    struct = json.loads(
        Path("tests/fixtures/eli/du_2023_725_struct.json").read_text(encoding="utf-8")
    )
    struct[0]["children"].append(struct[0]["children"][0].copy())
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2023/725/struct",
        status=200,
        body=json.dumps(struct).encode(),
    )
    _, _, units, _ = fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]
    count = sum(1 for unit in units if unit.act_identifier == "DU/2023/725")
    assert count == 3


def test_duplicated_struct_branch_produces_one_unit_per_leaf(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    """A repeated branch puts its copies in different recursion frames.

    ELI duplicates whole subtrees, not only leaf siblings: in DU/2024/1513 an
    article lists the same paragraph twice, each carrying the same points. A
    per-parent check cannot see those leaves as repeats, so deduplication has
    to run once over every leaf in the tree.
    """
    struct = json.loads(
        Path("tests/fixtures/eli/du_2023_725_struct.json").read_text(encoding="utf-8")
    )
    branch = struct[0]
    assert branch.get("children"), "fixture must have a branch to duplicate"
    struct.append(json.loads(json.dumps(branch)))
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2023/725/struct",
        status=200,
        body=json.dumps(struct).encode(),
    )
    _, _, units, _ = fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]
    count = sum(1 for unit in units if unit.act_identifier == "DU/2023/725")
    assert count == 3


def test_struct_repeat_with_reordered_keys_is_not_ambiguous() -> None:
    """Node equality must not depend on JSON key order.

    Two nodes carrying the same fields in a different insertion order are the
    same node; fingerprinting them unsorted would report ambiguous_struct_path
    for a repeat that is in fact identical.
    """
    first = {"type": "arti", "name": "11", "id": "arti_11", "title": "Art. 11."}
    second = {"title": "Art. 11.", "id": "arti_11", "name": "11", "type": "arti"}
    entries = deduped_struct_article_paths([first, second], "DU/2023/725")
    assert [path for path, _node in entries] == [("arti=11",)]


def test_ambiguous_struct_path_raises(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    struct = json.loads(
        Path("tests/fixtures/eli/du_2023_725_struct.json").read_text(encoding="utf-8")
    )
    duplicate = struct[0]["children"][0].copy()
    duplicate["title"] = "Art. 11 changed."
    struct[0]["children"].append(duplicate)
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2023/725/struct",
        status=200,
        body=json.dumps(struct).encode(),
    )
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "ambiguous_struct_path"


def test_html_act_without_struct_repeats_has_exact_unit_count(
    corpus_units: list[LegalUnit],
) -> None:
    html_count = sum(1 for u in corpus_units if u.act_identifier == "DU/2023/725")
    assert html_count == 3


def test_build_rejects_unexpected_source_format(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    metadata = json.loads(
        Path("tests/fixtures/eli/du_2023_725_metadata.json").read_text(encoding="utf-8")
    )
    metadata["textHTML"] = False
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2023/725",
        status=200,
        body=json.dumps(metadata).encode(),
    )
    with pytest.raises(CorpusBuildError, match="unexpected_source_format"):
        fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]


def test_build_rejects_changed_hash(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["acts"][0]["expected_hash"] = "deadbeef"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusBuildError, match="source_hash_mismatch"):
        fetch_manifest_units(path, fake_eli.client)  # type: ignore[attr-defined]


def test_build_preserves_superscript_article_identifier(
    built_corpus: QdrantCorpusIndex,
) -> None:
    index = built_corpus
    unit = index.read(
        "https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=41(1)"
    )
    assert "41¹" in unit.article_identifier or "41(1)" in unit.article_identifier
    assert "41 1" not in unit.article_identifier.replace("Art.", "")


def test_act_level_inforce_is_act_scope_not_provision(
    built_corpus: QdrantCorpusIndex,
) -> None:
    index = built_corpus
    candidate = index.search("wypowiedzenie najmu", 5)[0].candidate
    assert candidate.act_force.scope is ForceScope.ACT
    assert candidate.act_force.value is ForceValue.IN_FORCE
    assert candidate.provision_force.scope is ForceScope.PROVISION
    assert candidate.provision_force.value is ForceValue.UNDETERMINED


def test_builder_carries_both_force_records_for_every_unit(
    corpus_units: list[LegalUnit],
) -> None:
    assert corpus_units, "no units built; the loop below would assert nothing"
    for unit in corpus_units:
        assert unit.act_force.scope == ForceScope.ACT
        assert unit.provision_force.scope == ForceScope.PROVISION


def test_pdf_parse_ignores_citations_and_keeps_superscripts_distinct(
    fake_eli: object, tmp_path: Path
) -> None:
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
        status=200,
        body=_render_real_structure_pdf(),
    )
    manifest = _pdf_test_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _, _, units, _ = fetch_manifest_units(path, fake.client)  # type: ignore[attr-defined]
    rows = [(u.id, u.article_identifier, u.text) for u in units]

    ids = [row[0] for row in rows]
    assert len(rows) == 11
    assert ids == [
        "DU/2026/795:article=1",
        "DU/2026/795:article=2",
        "DU/2026/795:article=3",
        "DU/2026/795:article=4",
        "DU/2026/795:article=5",
        "DU/2026/795:article=15",
        "DU/2026/795:article=16",
        "DU/2026/795:article=16(1)",
        "DU/2026/795:article=17",
        "DU/2026/795:article=353(1)",
        "DU/2026/795:article=354",
    ]
    for unit_id, article_identifier, text in rows:
        assert unit_id == unit_id.rstrip(" .")
        assert not article_identifier.endswith((" ", "."))
        base = re.match(r"\d+", unit_id.rsplit("=", 1)[1]).group()
        assert text.split("\n")[0].startswith(f"Art. {base}")


def test_pdf_path_rejects_non_pdf_body_as_typed_error(
    fake_eli: object, tmp_path: Path
) -> None:
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
        status=200,
        body=b"<html><body>not a pdf</body></html>",
    )
    manifest = _pdf_test_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusBuildError, match="PDF unreadable"):
        fetch_manifest_units(path, fake.client)  # type: ignore[attr-defined]


def test_legal_status_date_drift(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    metadata = json.loads(
        Path("tests/fixtures/eli/du_2023_725_metadata.json").read_text(encoding="utf-8")
    )
    metadata["legalStatusDate"] = "2023-01-01"
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2023/725",
        status=200,
        body=json.dumps(metadata).encode(),
    )
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "legal_status_date_drift"


def test_base_act_mismatch(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    metadata = json.loads(
        Path("tests/fixtures/eli/du_2023_725_metadata.json").read_text(encoding="utf-8")
    )
    metadata["references"]["Tekst jednolity dla aktu"][0]["id"] = "DU/1999/999"
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2023/725",
        status=200,
        body=json.dumps(metadata).encode(),
    )
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "base_act_mismatch"


def test_amendment_drift(fake_eli: object, manifest_path: Path, tmp_path: Path) -> None:
    base_metadata = json.loads(
        Path("tests/fixtures/eli/du_2001_733_metadata.json").read_text(encoding="utf-8")
    )
    base_metadata["references"]["Akty zmieniające"].append(
        {
            "id": "DU/2024/999",
            "date": "2024-06-01",
            "title": "New amendment in window",
        }
    )
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2001/733",
        status=200,
        body=json.dumps(base_metadata).encode(),
    )
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "amendment_drift"


def test_target_date_before_text(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_target_date"] = "2020-01-01"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(path, fake_eli.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "target_date_before_text"


def test_amendment_after_corpus_target_date_not_required(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    """Fixture DU/2027/999 is after corpus_target_date and must stay excluded."""
    _, _, _, act_currency = fetch_manifest_units(manifest_path, fake_eli.client)  # type: ignore[attr-defined]
    pdf_entry = next(
        entry for entry in act_currency if entry.act_identifier == "DU/2026/795"
    )
    assert pdf_entry.amendments_not_carried == ()


def test_amendment_on_legal_status_date_not_required(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    """A relation dated on legal_status_date is already in the consolidated text."""
    base_metadata = json.loads(
        Path("tests/fixtures/eli/du_1964_93_metadata.json").read_text(encoding="utf-8")
    )
    base_metadata["references"]["Akty zmieniające"].append(
        {
            "id": "DU/2026/100",
            "date": "2026-05-19",
            "title": "Amendment on compilation date",
        }
    )
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/1964/93",
        status=200,
        body=json.dumps(base_metadata).encode(),
    )
    _, _, _, act_currency = fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]
    pdf_entry = next(
        entry for entry in act_currency if entry.act_identifier == "DU/2026/795"
    )
    assert pdf_entry.amendments_not_carried == ()


def test_production_manifest_parses() -> None:
    manifest = CorpusManifest.model_validate_json(
        Path("corpus/manifest.example.json").read_text(encoding="utf-8")
    )
    assert manifest.version == 2
    # Eleven acts. The last three -- cooperative housing, environmental protection
    # and public finance -- were added so that the real-source documents have their
    # governing law present. The corpus still carries the acts that governed the
    # documents dropped by dataset_version 0.5.0: the corpus is the law the artefact
    # searches, not an index of the evaluation set, and narrowing it to the retained
    # five would make retrieval easier than the task it stands for.
    assert len(manifest.acts) == 11
    act_ids = [f"{act.publisher}/{act.year}/{act.position}" for act in manifest.acts]
    assert len(act_ids) == len(set(act_ids))
    for act in manifest.acts:
        assert manifest.corpus_target_date >= act.legal_status_date


def test_malformed_amendment_relation_absent_key(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    base_metadata = json.loads(
        Path("tests/fixtures/eli/du_1964_93_metadata.json").read_text(encoding="utf-8")
    )
    del base_metadata["references"]
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/1964/93",
        status=200,
        body=json.dumps(base_metadata).encode(),
    )
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "malformed_amendment_relation"


def test_malformed_amendment_relation_unparseable_date(
    fake_eli: object, manifest_path: Path, tmp_path: Path
) -> None:
    base_metadata = json.loads(
        Path("tests/fixtures/eli/du_1964_93_metadata.json").read_text(encoding="utf-8")
    )
    base_metadata["references"]["Akty zmieniające"].append(
        {
            "id": "DU/2026/888",
            "date": "not-a-date",
            "title": "Ustawa z nieparsowalną datą",
        }
    )
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/1964/93",
        status=200,
        body=json.dumps(base_metadata).encode(),
    )
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(manifest_path, fake.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "malformed_amendment_relation"


def test_unit_digest_stable_across_rebuilds(
    fake_eli: object, manifest_path: Path
) -> None:
    first = _digest_of(fetch_manifest_units(manifest_path, fake_eli.client)[2])  # type: ignore[attr-defined]
    second = _digest_of(fetch_manifest_units(manifest_path, fake_eli.client)[2])  # type: ignore[attr-defined]
    assert first == second
    assert first


def test_unit_digest_changes_when_fixture_body_changes(
    fake_eli: object, manifest_path: Path
) -> None:
    first = _digest_of(fetch_manifest_units(manifest_path, fake_eli.client)[2])  # type: ignore[attr-defined]
    art11 = Path("tests/fixtures/eli/du_2023_725_art11.html").read_bytes()
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2023/725/text.html/arti=11",
        status=200,
        body=art11 + b"<p>changed</p>",
    )
    second = _digest_of(fetch_manifest_units(manifest_path, fake.client)[2])  # type: ignore[attr-defined]
    assert second != first


def test_mark_superscript_digits_is_noop_without_superscript() -> None:
    chars = [
        _pdf_char(character, size=10.0, x0=index * 5, x1=(index + 1) * 5, y0=100)
        for index, character in enumerate("Art. 1")
    ]
    assert _extract_marked_pdf_text(chars) == "Art. 1"


def test_mark_superscript_digits_marks_two_digit_run_and_keeps_period() -> None:
    chars = [
        _pdf_char("A", size=10.0, x0=0, x1=5, y0=100),
        _pdf_char("r", size=10.0, x0=5, x1=10, y0=100),
        _pdf_char("t", size=10.0, x0=10, x1=15, y0=100),
        _pdf_char(".", size=10.0, x0=15, x1=18, y0=100),
        _pdf_char(" ", size=10.0, x0=18, x1=20, y0=100),
        _pdf_char("6", size=10.0, x0=20, x1=25, y0=100),
        _pdf_char("7", size=10.0, x0=25, x1=30, y0=100),
        _pdf_char("1", size=6.0, x0=30, x1=33, y0=103),
        _pdf_char("8", size=6.0, x0=33, x1=36, y0=103),
        _pdf_char(".", size=10.0, x0=36, x1=39, y0=100),
    ]
    extracted = _extract_marked_pdf_text(chars)
    assert extracted == "Art. 67[18]."
    raw_identifier = next(act_pdf.iter_pdf_headings(f"{extracted} Text."))[1]
    assert act_pdf.canonical_article_number(raw_identifier) == "67(18)"


def _chars_for_run(
    prefix: str, superscript: str, *, body: float = 10.0, small: float = 6.5
) -> list[dict[str, float | str | bool | int]]:
    """Body-size ``prefix`` followed by a raised ``superscript`` run, then a period."""
    chars: list[dict[str, float | str | bool | int]] = []
    x = 0.0
    for character in prefix:
        chars.append(_pdf_char(character, size=body, x0=x, x1=x + body * 0.5, y0=100.0))
        x += body * 0.5
    for digit in superscript:
        chars.append(_pdf_char(digit, size=small, x0=x, x1=x + small * 0.5, y0=103.0))
        x += small * 0.5
    chars.append(_pdf_char(".", size=body, x0=x, x1=x + body * 0.5, y0=100.0))
    return chars


def test_a_superscript_on_a_lettered_article_number_is_an_editorial_index() -> None:
    """Polish drafting indexes inserted articles in turn, and the PDFs show it.

    The marker required the character under a raised digit to be a digit itself, so
    ``art. 112a`` with a raised 1 came through as the flat ``112a1`` while ``art. 67``
    with a raised 18 became ``67[18]``. Two acts in the corpus carry the form --
    DU/2025/1483 art. 112a(1) and DU/2024/1513 art. 3a(1), both typeset at 6.5pt
    raised over a 10pt body -- so the same superscript was read two ways depending
    on whether a letter stood in between.
    """
    text = _extract_marked_pdf_text(_chars_for_run("Art. 112a", "1"))
    assert text == "Art. 112a[1]."
    raw = next(act_pdf.iter_pdf_headings(f"{text} Text."))[1]
    assert act_pdf.canonical_article_number(raw) == "112a(1)"


def test_a_superscript_after_a_word_is_still_a_footnote_reference() -> None:
    """The widening must not reach the thing it sits next to.

    A raised digit on a word is a footnote reference and is typeset exactly like an
    editorial index. What separates them is what carries them: an article number is
    digits with at most a letter suffix, so a letter counts only when a digit stands
    behind it. Walking back over ``ustawa`` finds no digit and the digit stays put.
    """
    assert _extract_marked_pdf_text(_chars_for_run("ustawa", "1")) == "ustawa1."
    # A year carries footnotes too, and 'r' is a letter behind a digit. Without
    # the introducer this became 'z 2024r[1]', rewriting a footnote marker into
    # the provision text and its content hash.
    assert _extract_marked_pdf_text(_chars_for_run("z 2024r", "1")) == "z 2024r1."
    # A single letter standing alone after a number is a word, not a suffix: the
    # digit behind it is separated by a space, and the walk stops at the gap.
    assert _extract_marked_pdf_text(_chars_for_run("Art. 5 w", "1")) == "Art. 5 w1."


def test_a_multi_letter_article_suffix_takes_its_superscript_too() -> None:
    """Polish drafting runs 112a..112z and then 112aa, so the suffix is unbounded.

    canonical_article_number already reduced 112aa[1] to 112aa(1); the marker
    stopped after one letter, so the two disagreed on which forms exist and the
    reduction was unreachable for anything past 112z. No act in the manifest
    carries the form today -- the eleven hold 112a and 3a only -- so this pins the
    agreement rather than a case that has been observed.
    """
    for base in ("Art. 112aa", "Art. 112ab"):
        text = _extract_marked_pdf_text(_chars_for_run(base, "1"))
        assert text == f"{base}[1].", text
        raw = next(act_pdf.iter_pdf_headings(f"{text} Text."))[1]
        assert act_pdf.canonical_article_number(raw) == f"{base[5:]}(1)"


def test_a_lowercase_cross_reference_marks_like_a_heading() -> None:
    """The same provision is written 'art. 112a' mid-sentence and 'Art. 112a' as a
    heading. Marking only the heading would leave the two renderings of one
    citation different in the text a reader and an encoder both see."""
    assert _extract_marked_pdf_text(_chars_for_run("art. 112a", "1")) == "art. 112a[1]."


def _chars_for_art_with_superscript(
    base: str, superscript: str, *, body: float = 10.0, small: float = 6.0
) -> list[dict[str, float | str | bool | int]]:
    chars: list[dict[str, float | str | bool | int]] = []
    x = 0.0
    y_body = 100.0
    y_super = 103.0
    for character in f"Art. {base}":
        width = body * 0.5
        chars.append(_pdf_char(character, size=body, x0=x, x1=x + width, y0=y_body))
        x += width
    for digit in superscript:
        width = small * 0.5
        chars.append(_pdf_char(digit, size=small, x0=x, x1=x + width, y0=y_super))
        x += width
    chars.append(_pdf_char(".", size=body, x0=x, x1=x + body * 0.5, y0=y_body))
    return chars


def test_mark_superscript_digits_leaves_time_minutes_alone() -> None:
    """A raised run starting with zero is a time, not an editorial index.

    DU/2025/277 states night work as the hours between 21(00) and 7(00). Marking
    those minutes would corrupt the provision text, its content hash and its
    embedding. No editorial index is written with a leading zero.
    """
    chars = [
        _pdf_char("2", size=10.0, x0=0, x1=5, y0=100),
        _pdf_char("1", size=10.0, x0=5, x1=10, y0=100),
        _pdf_char("0", size=6.0, x0=10, x1=13, y0=103),
        _pdf_char("0", size=6.0, x0=13, x1=16, y0=103),
    ]
    assert _extract_marked_pdf_text(chars) == "2100"


def test_mark_superscript_digits_splits_article_22_from_221() -> None:
    first = _chars_for_art_with_superscript("22", "1")
    second: list[dict[str, float | str | bool | int]] = []
    x = 0.0
    for character in "Art. 221.":
        width = 10.0 * 0.5
        second.append(_pdf_char(character, size=10.0, x0=x, x1=x + width, y0=200))
        x += width
    text = _extract_marked_pdf_text(first) + "\n" + _extract_marked_pdf_text(second)
    numbers = [
        act_pdf.canonical_article_number(raw)
        for _, raw in act_pdf.iter_pdf_headings(text)
    ]
    assert numbers == ["22(1)", "221"]


def test_mark_superscript_digits_leaves_footnote_digit_after_period() -> None:
    chars: list[dict[str, float | str | bool | int]] = []
    x = 0.0
    body = 10.0
    small = 6.0
    for character in "Art. 184.":
        width = body * 0.5
        chars.append(_pdf_char(character, size=body, x0=x, x1=x + width, y0=100))
        x += width
    width = small * 0.5
    chars.append(_pdf_char("1", size=small, x0=x, x1=x + width, y0=103))
    x += width
    for character in "7)":
        width = body * 0.5
        chars.append(_pdf_char(character, size=body, x0=x, x1=x + width, y0=100))
        x += width
    assert _extract_marked_pdf_text(chars) == "Art. 184.17)"


def _render_repeated_article_pdf() -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("FixtureFont", "", resolve_test_font())
    pdf.set_font("FixtureFont", size=12)
    for line in (
        "Art. 184. First printing 17).",
        "Art. 184. Second printing 18).",
    ):
        pdf.multi_cell(w=0, h=6, text=line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def test_pdf_repeated_article_yields_two_printings(
    fake_eli: object, tmp_path: Path
) -> None:
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
        status=200,
        body=_render_repeated_article_pdf(),
    )
    manifest = _pdf_test_manifest(
        repeated_articles=[{"article": "184", "printings": 2}]
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _, _, units, _ = fetch_manifest_units(path, fake.client)  # type: ignore[attr-defined]
    rows = [(u.id, u.locator, u.content_hash) for u in units]
    assert [row[0] for row in rows] == [
        "DU/2026/795:article=184:printing=1",
        "DU/2026/795:article=184:printing=2",
    ]
    assert rows[0][1] != rows[1][1]
    assert rows[0][2] != rows[1][2]
    assert rows[0][1].endswith("&printing=1")
    assert rows[1][1].endswith("&printing=2")


def test_pdf_single_article_keeps_legacy_id_and_locator(
    fake_eli: object, tmp_path: Path
) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("FixtureFont", "", resolve_test_font())
    pdf.set_font("FixtureFont", size=12)
    pdf.multi_cell(
        w=0,
        h=6,
        text="Art. 5. Single occurrence.",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
        status=200,
        body=bytes(pdf.output()),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_pdf_test_manifest()), encoding="utf-8")
    _, _, units, _ = fetch_manifest_units(path, fake.client)  # type: ignore[attr-defined]
    row = (units[0].id, units[0].locator)
    assert row == (
        "DU/2026/795:article=5",
        "https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf#article=5",
    )


def test_repeated_article_drift(fake_eli: object, tmp_path: Path) -> None:
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
        status=200,
        body=_render_repeated_article_pdf(),
    )
    manifest = _pdf_test_manifest(repeated_articles=[])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(path, fake.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "repeated_article_drift"


def _render_triple_article_pdf() -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("FixtureFont", "", resolve_test_font())
    pdf.set_font("FixtureFont", size=12)
    for line in (
        "Art. 184. First printing.",
        "Art. 184. Second printing.",
        "Art. 184. Third printing.",
    ):
        pdf.multi_cell(w=0, h=6, text=line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def test_repeated_article_count_drift(fake_eli: object, tmp_path: Path) -> None:
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
        status=200,
        body=_render_triple_article_pdf(),
    )
    manifest = _pdf_test_manifest(
        repeated_articles=[{"article": "184", "printings": 2}]
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(path, fake.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "repeated_article_drift"


def test_repeated_articles_not_applicable_for_html(
    fake_eli: object, tmp_path: Path
) -> None:
    manifest = {
        "version": 2,
        "corpus_target_date": "2026-08-30",
        "acts": [
            {
                "publisher": "DU",
                "year": 2023,
                "position": 725,
                "source_format": "html",
                "base_act": "DU/2001/733",
                "legal_status_date": "2023-03-09",
                "amendments_not_carried": [],
                "repeated_articles": [{"article": "184", "printings": 2}],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(path, fake_eli.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "repeated_articles_not_applicable"


def test_unsupported_manifest_version(fake_eli: object, tmp_path: Path) -> None:
    manifest = {
        "version": 1,
        "corpus_target_date": "2026-08-30",
        "acts": [
            {
                "publisher": "DU",
                "year": 2026,
                "position": 795,
                "source_format": "pdf",
                "base_act": "DU/1964/93",
                "legal_status_date": "2026-05-19",
                "amendments_not_carried": [],
                "repeated_articles": [],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(path, fake_eli.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "unsupported_manifest_version"


def test_manifest_rejects_unknown_keys(fake_eli: object, tmp_path: Path) -> None:
    manifest = _pdf_test_manifest(stale_field="unexpected")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(path, fake_eli.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "invalid_manifest"


def test_duplicate_act_entry(fake_eli: object, tmp_path: Path) -> None:
    manifest = {
        "version": 2,
        "corpus_target_date": "2026-08-30",
        "acts": [
            _pdf_test_manifest()["acts"][0],
            _pdf_test_manifest()["acts"][0],
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(path, fake_eli.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "duplicate_act_entry"


def test_duplicate_amendment_entry(fake_eli: object, tmp_path: Path) -> None:
    manifest = {
        "version": 2,
        "corpus_target_date": "2026-08-30",
        "acts": [
            {
                "publisher": "DU",
                "year": 2023,
                "position": 725,
                "source_format": "html",
                "base_act": "DU/2001/733",
                "legal_status_date": "2023-03-09",
                "amendments_not_carried": [
                    {"id": "DU/2024/999", "eli_relation_date": "2024-06-01"},
                    {"id": "DU/2024/999", "eli_relation_date": "2024-06-01"},
                ],
                "repeated_articles": [],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(path, fake_eli.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "duplicate_amendment_entry"


@pytest.mark.parametrize(
    ("text", "expected", "note"),
    [
        ("1. (uchylony)", True, "Form 1: Paragraph-level unit '1. (uchylony)'"),
        ("5) (uchylony)", True, "Form 1: Point-level unit '5) (uchylony)'"),
        ("7) (uchylony)", True, "Form 1: Point-level unit '7) (uchylony)'"),
        ("1a. (uchylony)", True, "Form 1: Paragraph-level unit '1a. (uchylony)'"),
        ("a) (uchylony)", True, "Form 1: Letter-level unit 'a) (uchylony)'"),
        (
            "Art. 20-22. (uchylone)",
            True,
            "Form 2: Article range with hyphen and plural 'uchylone'",
        ),
        (
            "Art. 20–22. (uchylone)",
            True,
            "Form 2: Article range with en-dash and plural 'uchylone'",
        ),
        (
            "Art. 20-22. (skreślone)",
            True,
            "Form 2: Article range with plural 'skreślone'",
        ),
        (
            "Art. 20-22. (utraciły moc)",
            True,
            "Form 2: Article range with plural 'utraciły moc'",
        ),
        (
            "Art. 179. (utracił moc)2)",
            True,
            "Form 3: Repeal verb 'utracił moc' with footnote index '2)'",
        ),
        (
            "Art. 103. (utracił moc)5)",
            True,
            "Form 3: Repeal verb 'utracił moc' with footnote index '5)'",
        ),
        (
            "Art. 179. (utracił moc)",
            True,
            "Form 3: Repeal verb 'utracił moc' without footnote index",
        ),
        (
            "§ 1. (utraciła moc)",
            True,
            "Form 3: Repeal verb 'utraciła moc' on section unit",
        ),
        (
            "Art. 171. (uchylony) Rozdział II Zasiedzenie",
            True,
            "Form 4: Next chapter heading glued inline",
        ),
        (
            "Art. 109[9]. (uchylony) TYTUŁ V Termin",
            True,
            "Form 4: Next title heading glued inline",
        ),
        (
            "Art. 526. (uchylony)\n"
            "TYTUŁ X\n"
            "Ochrona wierzyciela w razie niewypłacalności dłużnika",
            True,
            "Form 4: Next title heading and name glued across newlines",
        ),
        (
            "Art. 12. (uchylony) Dziennik Ustaw - 6 - Poz. 277",
            True,
            "Form 5: PDF page furniture glued inline",
        ),
        (
            "Art. 133. (uchylony)\nDziennik Ustaw – 19 – Poz. 795",
            True,
            "Form 5: PDF page furniture on next line",
        ),
        (
            "Art. 284. 1. (uchylony) 2. (utracił moc)69)",
            True,
            "Form 6: Mixed markers across paragraphs in one article",
        ),
        (
            "Art. 8. § 1. (uchylony)\n§ 2. (uchylony)",
            True,
            "Form 6: Multiple repealed sections in one article",
        ),
        (
            "1. (uchylony)\n2. (uchylony)",
            True,
            "Form 6: Multiple repealed paragraphs in one unit",
        ),
        (
            "Art. 1061. (uchylony)\n"
            "18) Utracił moc z dniem 14 lutego 2001 r. w zakresie, w którym odnosi"
            " się do spadków otwartych od dnia 14 lutego 2001 r., na podstawie"
            " wyroku Trybunału Konstytucyjnego, o którym mowa w odnośniku 14.",
            True,
            "Repealed article carrying TK footnote definition at bottom",
        ),
        (
            "1. (utracił moc)\n"
            "Z dniem 31 grudnia 2006 r. na podstawie wyroku Trybunału\n"
            "Konstytucyjnego z dnia 17 maja 2006 r. sygn. akt K 33/05"
            " (Dz. U. poz. 602).",
            True,
            "Repealed paragraph carrying TK loss-of-force annotation",
        ),
        (
            "Art. 24. (uchylony)\n"
            "Uznany za niezgodny z Konstytucją Rzeczypospolitej Polskiej z dniem 25"
            " kwietnia 2005 r. w zakresie, w jakim nie przewiduje możliwości...",
            True,
            "Repealed article carrying TK unconstitutionality annotation",
        ),
        (
            "Art. 101. Umocowanie wygasa ze śmiercią mocodawcy",
            False,
            "Substantive Civil Code: power of attorney expiring on death",
        ),
        (
            "Art. 88. Uprawnienie do uchylenia się wygasa",
            False,
            "Substantive Civil Code: rescission right expiring",
        ),
        (
            "Art. 109. Prokura wygasa z mocy prawa",
            False,
            "Substantive Civil Code: commercial power of attorney expiring",
        ),
        (
            "Art. 63. Umowa o pracę wygasa w przypadkach określonych w kodeksie"
            " oraz w przepisach szczególnych.",
            False,
            "Substantive Labor Code: employment contract termination",
        ),
        (
            "Art. 67. Traci moc ustawa z dnia 20 lipca 2001 r. o kredycie"
            " konsumenckim (Dz. U. poz. 1081, z późn. zm.9)).",
            False,
            "Substantive active derogation provision repealing an earlier statute",
        ),
        (
            "Art. 8. § 1. Każdy człowiek od chwili urodzenia ma zdolność prawną.\n"
            "§ 2. (uchylony)",
            False,
            "Inline repeal: § 2 is repealed but § 1 is active substantive law",
        ),
        (
            "Art. 812. § 1. (uchylony)\n"
            "§ 2. Umowa ubezpieczenia może być zawarta na cudzy rachunek.",
            False,
            "Inline repeal: § 1 is repealed but § 2 is active substantive law",
        ),
        (
            "Art. 51. 1. (utracił moc)\n"
            "2. (utracił moc)\n"
            "3. Wprowadzenie do obrotu oryginału albo egzemplarza utworu...",
            False,
            "Inline repeal: para 1-2 repealed but para 3 is active substantive law",
        ),
        (
            "Art. 359. 1. (uchylony)\n"
            "2. Kto, będąc do tego obowiązany na podstawie art. 286, nie dopełnia"
            " obowiązku terminowego przedkładania wykazu podlega karze grzywny.",
            False,
            "Inline repeal: para 1 repealed but para 2 is active penal provision",
        ),
        (
            "Art. 112a. (uchylony)\n"
            "Art. 112a1. (uchylony)\n"
            "Art. 112aa. 1. Kwota wydatków na dany rok organów i jednostek...",
            False,
            "Glued unit: Art. 112a/112a1 repealed but Art. 112aa is active law",
        ),
    ],
)
def test_editorial_repeal_placeholder_shape(
    text: str, expected: bool, note: str
) -> None:
    assert _is_editorial_repeal_placeholder(text) is expected, note


def test_provision_force_never_carries_in_force() -> None:
    observed = date(2026, 8, 30)
    for text in ("Art. 2. (uchylony)", "Art. 101. Umocowanie wygasa"):
        record = provision_force_state(
            text=text,
            snapshot_date=observed,
            locator="https://example.test/unit",
        )
        assert record.scope is ForceScope.PROVISION
        assert record.value is not ForceValue.IN_FORCE


_BUILT_CORPUS = Path(__file__).resolve().parents[2] / "corpus" / "build"
_OLD_PROVISION_REPEAL = re.compile(
    r"\b(?:uchylony|traci\s+moc|wygas[ał]|nie\s+obowi[aą]zuje)\b",
    re.IGNORECASE,
)
_INLINE_REPEAL = re.compile(r"§\s*\d+[^§]*\(uchylon[ay]\)", re.IGNORECASE)


def test_iter_pdf_headings_unclosed_quote_at_end_of_text() -> None:
    text = (
        "1) art. 13 ustawy, która stanowi:\n"
        "„Art. 13. Przepis przejściowy bez domknięcia cudzysłowu.\n"
        "Art. 1. Kodeks niniejszy reguluje stosunki cywilnoprawne."
    )
    with pytest.raises(CorpusBuildError) as exc_info:
        list(act_pdf.iter_pdf_headings(text))
    assert exc_info.value.code == "unbalanced_quotes"
    assert "unclosed quote at end of text" in str(exc_info.value)


def test_iter_pdf_headings_surplus_closing_quote() -> None:
    text = (
        "Art. 1. Przepis z nadmiarowym cudzysłowem zamykającym”.\n"
        "Art. 2. Kolejny przepis."
    )
    with pytest.raises(CorpusBuildError) as exc_info:
        list(act_pdf.iter_pdf_headings(text))
    assert exc_info.value.code == "unbalanced_quotes"
    assert "surplus closing quote mark" in str(exc_info.value)


def test_iter_pdf_headings_quote_open_exceeds_bound() -> None:
    lines = ["„Art. 10. Start cytatu."]
    lines.extend(f"Wiersz cytatu {i}." for i in range(205))
    lines.append("Koniec cytatu.”")
    text = "\n".join(lines)
    with pytest.raises(CorpusBuildError) as exc_info:
        list(act_pdf.iter_pdf_headings(text))
    assert exc_info.value.code == "unbalanced_quotes"
    assert "exceeding bound" in str(exc_info.value)


def test_build_corpus_pdf_unclosed_quote_aborts_build(
    fake_eli: object, tmp_path: Path
) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("FixtureFont", "", resolve_test_font())
    pdf.set_font("FixtureFont", size=12)
    pdf.multi_cell(
        w=0,
        h=6,
        text="„Art. 13. Przepis przejściowy.\nArt. 1. Główny artykuł.",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
        status=200,
        body=bytes(pdf.output()),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_pdf_test_manifest()), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(path, fake.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "unbalanced_quotes"


def test_collect_pdf_units_empty_article_text_aborts_build(
    fake_eli: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("FixtureFont", "", resolve_test_font())
    pdf.set_font("FixtureFont", size=12)
    pdf.multi_cell(
        w=0,
        h=6,
        text="Art. 1. Treść artykułu pierwszego.\nArt. 2. Treść artykułu drugiego.",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    fake = fake_eli
    fake.respond(  # type: ignore[attr-defined]
        url="https://api.sejm.gov.pl/eli/acts/DU/2026/795/text.pdf",
        status=200,
        body=bytes(pdf.output()),
    )

    def _corrupted_headings(_text: str) -> list[tuple[int, str]]:
        return [(0, "1"), (0, "2")]

    monkeypatch.setattr("seeder.eli.iter_pdf_headings", _corrupted_headings)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_pdf_test_manifest()), encoding="utf-8")
    with pytest.raises(CorpusBuildError) as exc_info:
        fetch_manifest_units(path, fake.client)  # type: ignore[attr-defined]
    assert exc_info.value.code == "malformed_structure"
    assert "empty article text for 1 in DU/2026/795" in str(exc_info.value)


@pytest.mark.parametrize(
    ("heading", "identifier"),
    [
        ("Art. 112.", "112"),
        ("Art. 112a.", "112a"),
        # Multi-letter suffixes are ordinary in Polish statute numbering, and the
        # marker used to cap the suffix at one letter: "Art. 112aa." was not seen
        # as a heading, so its article was appended to the previous one. One
        # locator then resolved to two articles, which no amount of retrieval
        # quality can repair and which snapshot_citation_correctness must fail.
        ("Art. 112aa.", "112aa"),
        ("Art. 112ab.", "112ab"),
        # A superscript ordinal the PDF extractor flattened to ASCII.
        ("Art. 112a1.", "112a1"),
        ("Art. 385¹.", "385¹"),
        ("Art. 385(1).", "385(1)"),
    ],
)
def test_article_headings_survive_letter_and_ordinal_suffixes(
    heading: str, identifier: str
) -> None:
    match = act_pdf._ARTICLE_HEADING.match(heading)
    assert match is not None, f"{heading!r} was not recognised as a heading"
    assert match.group(1) == identifier


@pytest.mark.parametrize(
    "line",
    [
        # A reference inside a sentence, not a heading.
        "Art. 5 ustawy stanowi inaczej",
        # Two numbers separated by a space: which article is meant is ambiguous,
        # and guessing would silently mis-split the act.
        "Art. 112 1.",
    ],
)
def test_a_reference_is_not_mistaken_for_a_heading(line: str) -> None:
    assert act_pdf._ARTICLE_HEADING.match(line) is None


def test_consecutive_lettered_articles_split_into_separate_units() -> None:
    """The end-to-end consequence: two articles, two units, two locators."""
    text = (
        "Art. 112a. Pierwszy przepis.\n"
        "Art. 112aa. Drugi przepis.\n"
        "Art. 113. Trzeci przepis.\n"
    )
    headings = list(act_pdf.iter_pdf_headings(text))
    assert [identifier for _, identifier in headings] == ["112a", "112aa", "113"]
