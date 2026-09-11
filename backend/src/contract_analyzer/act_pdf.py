from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

from pdfplumber.utils.text import extract_text as _extract_chars_text

from contract_analyzer.domain import CorpusBuildError

SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
SUPERSCRIPT = str.maketrans("0123456789", SUPERSCRIPT_DIGITS)
SUPERSCRIPT_TO_ASCII = str.maketrans(SUPERSCRIPT_DIGITS, "0123456789")

# Matches article number with optional lowercase suffix and superscript/ordinal digits.
_ARTICLE_MARKER = re.compile(
    rf"Art\.\s*(\d+[a-z]*(?:\(\d+\)|\[\d+\]|\^\d+|[{SUPERSCRIPT_DIGITS}]+|\d+)?)"
    rf"(?:\.|\s*$)"
)
_ARTICLE_HEADING = re.compile(rf"[ \t]*{_ARTICLE_MARKER.pattern}")
_MAX_QUOTE_LINES = 200
_AMBIGUOUS_ARTICLE_NUMBER = re.compile(r"[ \t]*Art\.\s*\d+\s+\d+(?:\.|\s*$)")


def normalize_superscripts(text: str) -> str:
    return re.sub(
        r"<sup>(\d+)</sup>",
        lambda match: match.group(1).translate(SUPERSCRIPT),
        text,
        flags=re.IGNORECASE,
    )


def canonical_article_number(identifier: str) -> str:
    """Reduce any superscript rendering to the canonical ``N(1)`` form.

    The base may carry a lowercase suffix, because Polish drafting inserts article
    ``112a`` and then indexes that in turn. The optional group never matches a
    bare run of digits, so a greedy base keeps ``1121`` whole rather than reading
    it as article 112 index 1.
    """
    match = re.fullmatch(
        rf"(\d+[a-z]*)(?:\((\d+)\)|\[(\d+)\]|\^(\d+)|([{SUPERSCRIPT_DIGITS}]+))?",
        identifier,
    )
    if match is None:
        return identifier
    base = match.group(1)
    superscript = match.group(2) or match.group(3) or match.group(4)
    if superscript is None:
        unicode_digits = match.group(5)
        if unicode_digits is None:
            return base
        superscript = unicode_digits.translate(SUPERSCRIPT_TO_ASCII)
    return f"{base}({superscript})"


def rounded_char_size(char: dict[str, Any]) -> float:
    return round(float(char["size"]), 1)


def page_body_size(chars: list[dict[str, Any]]) -> float:
    sizes = [rounded_char_size(char) for char in chars if char.get("text")]
    if not sizes:
        raise CorpusBuildError(
            "malformed_structure",
            "malformed_structure: PDF page has no sized characters",
        )
    return Counter(sizes).most_common(1)[0][0]


def last_non_blank_char(
    chars: list[dict[str, Any]], before_index: int
) -> dict[str, Any] | None:
    for index in range(before_index - 1, -1, -1):
        if chars[index].get("text", "").strip():
            return chars[index]
    return None


def _ends_an_article_number(chars: list[dict[str, Any]], before_index: int) -> bool:
    """Whether the text just before ``before_index`` is an article number.

    A raised digit is an editorial index on an article number and a footnote
    reference on anything else, and the two are typeset identically, so only what
    carries the digit separates them: optional lowercase suffix, digits, then the
    ``Art.`` that introduces every article number in these documents.

    The introducer is what makes the suffix safe to leave unbounded. Drafting runs
    112a..112z and then 112aa, so a one-letter limit would miss real articles;
    without the introducer, an unbounded suffix would swallow ``z 2024r`` and
    rewrite a footnote on a year into the provision text and its content hash.
    """
    index = before_index - 1

    def back(i: int) -> int:
        while i >= 0 and not chars[i].get("text", "").strip():
            i -= 1
        return i

    saw_digit = False
    while index >= 0:
        text = chars[index].get("text", "")
        if not text.strip() or len(text) != 1:
            break
        if text.isdigit():
            saw_digit = True
        elif "a" <= text <= "z" and not saw_digit:
            pass
        else:
            break
        index -= 1
    if not saw_digit:
        return False

    for expected in (".", "t", "r"):
        index = back(index)
        if index < 0 or chars[index].get("text", "") != expected:
            return False
        index -= 1
    index = back(index)
    return index >= 0 and chars[index].get("text", "") in ("A", "a")


def mark_superscript_digits(chars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not chars:
        return []
    body_size = page_body_size(chars)
    small_threshold = 0.85 * body_size
    base_threshold = 0.9 * body_size
    marked: list[dict[str, Any]] = []
    index = 0
    while index < len(chars):
        char = chars[index]
        text = char.get("text", "")
        preceding = last_non_blank_char(chars, index)
        preceding_text = preceding.get("text", "") if preceding is not None else ""
        if (
            len(text) == 1
            and text.isdigit()
            and rounded_char_size(char) < small_threshold
            and preceding is not None
            and len(preceding_text) == 1
            and (preceding_text.isdigit() or _ends_an_article_number(chars, index))
            and rounded_char_size(preceding) >= base_threshold
            and float(char["y0"]) > float(preceding["y0"])
        ):
            digits = text
            run_end = index + 1
            while run_end < len(chars):
                next_char = chars[run_end]
                next_text = next_char.get("text", "")
                if (
                    len(next_text) == 1
                    and next_text.isdigit()
                    and rounded_char_size(next_char) < small_threshold
                ):
                    digits += next_text
                    run_end += 1
                else:
                    break
            if digits.startswith("0"):
                marked.append(char)
                index += 1
                continue
            merged = char.copy()
            last = chars[run_end - 1]
            merged["text"] = f"[{digits}]"
            merged["x1"] = last["x1"]
            merged["width"] = float(merged["x1"]) - float(merged["x0"])
            marked.append(merged)
            index = run_end
            continue
        marked.append(char)
        index += 1
    return marked


def extract_pdf_pages_text(pdf: Any) -> str:
    return "\n".join(
        _extract_chars_text(mark_superscript_digits(page.chars)) or ""
        for page in pdf.pages
    )


def pdf_article_locator(
    act_url: str, article_number: str, *, printing: int | None = None
) -> str:
    fragment = f"article={quote(article_number, safe='()')}"
    if printing is not None:
        fragment = f"{fragment}&printing={printing}"
    return f"{act_url}/text.pdf#{fragment}"


def pdf_article_unit_id(
    act_id: str, article_number: str, *, printing: int | None = None
) -> str:
    if printing is None:
        return f"{act_id}:article={article_number}"
    return f"{act_id}:article={article_number}:printing={printing}"


def iter_pdf_headings(text: str) -> Iterator[tuple[int, str]]:
    quote_depth = 0
    quote_open_lines = 0
    offset = 0
    for line in text.split("\n"):
        if quote_depth == 0:
            if _AMBIGUOUS_ARTICLE_NUMBER.match(line) is not None:
                raise CorpusBuildError(
                    "ambiguous_article_number",
                    f"ambiguous_article_number: cannot tell superscript from "
                    f"plain number in {line.strip()!r}",
                )
            match = _ARTICLE_HEADING.match(line)
            if match is not None:
                yield offset + match.start(), match.group(1)
        quote_depth += line.count("„") - line.count("”")
        if quote_depth < 0:
            raise CorpusBuildError(
                "unbalanced_quotes",
                f"unbalanced_quotes: surplus closing quote mark in {line.strip()!r}",
            )
        if quote_depth > 0:
            quote_open_lines += 1
            if quote_open_lines > _MAX_QUOTE_LINES:
                raise CorpusBuildError(
                    "unbalanced_quotes",
                    (
                        f"unbalanced_quotes: quote open for {quote_open_lines} lines "
                        f"exceeding bound {_MAX_QUOTE_LINES}"
                    ),
                )
        else:
            quote_open_lines = 0
        offset += len(line) + 1

    if quote_depth > 0:
        raise CorpusBuildError(
            "unbalanced_quotes",
            f"unbalanced_quotes: unclosed quote at end of text (depth {quote_depth})",
        )
