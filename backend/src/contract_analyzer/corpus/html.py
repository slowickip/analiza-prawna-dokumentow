"""Reading ELI's HTML source format: its struct paths, and text out of its markup."""

from __future__ import annotations

import json
import re
from html import unescape
from typing import Any

from contract_analyzer.act_pdf import normalize_superscripts

# ELI's node types that represent structural containers or articles.
STRUCT_SEGMENTS = frozenset(
    {"book", "titl", "bran", "chpt", "schp", "arti", "pass", "para", "pint", "lett"}
)

_HTML_TAG = re.compile(r"<[^>]+>")


def article_identifier_from_title(title: str) -> str:
    """Extract an article identifier (e.g. 'Art. 1') from an ELI node title."""
    normalized = normalize_superscripts(title)
    without_tags = _HTML_TAG.sub("", normalized)
    cleaned = re.sub(r"\s+", " ", unescape(without_tags)).strip()
    match = re.search(r"(Art\.\s*.+?)(?:\.|$)", cleaned, re.IGNORECASE)
    if match is None:
        return cleaned
    return match.group(1).strip()


def html_to_text(html: bytes) -> str:
    """One article's markup as the plain text a unit carries.

    Superscripts are normalised first, so an editorial index survives the tag
    stripping that would otherwise flatten it into the number beside it.
    """
    decoded = normalize_superscripts(html.decode("utf-8"))
    without_tags = _HTML_TAG.sub(" ", decoded)
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()


def node_tree_fingerprint(node: dict[str, Any]) -> str:
    """Identify a struct node by its whole subtree, not by its path.

    Sorted, so two nodes differing only in key order are the same node. Two nodes
    that merely share a path are different provisions, and collapsing those would
    drop one without a word.
    """
    return json.dumps(node, ensure_ascii=False, sort_keys=True)


def is_childless_stub_of(stub: dict[str, Any], full: dict[str, Any]) -> bool:
    """Whether ``stub`` is ``full`` listed again with its subtree left off.

    ELI lists some articles twice, once whole and once as a bare heading: the two
    nodes agree in every field, including the id ELI assigns, and differ only in
    that one carries ``children`` and the other omits them. That is one provision
    written down twice, not two provisions at one address, and the caller keeps
    the copy that has the text.

    The test is deliberately narrow. A repeat that differs anywhere else, or that
    carries a *different* subtree, is not a stub and stays an ambiguity: those are
    the shapes that would silently drop a provision.

    ``full`` is not required to have children. Two copies that are both text-free
    and differ only in writing ``children`` as an empty list against omitting the
    key are the same empty article, and refusing an act over that spelling would
    be a false alarm rather than a caught one.
    """
    if stub.get("children"):
        return False
    without_children = {k: v for k, v in stub.items() if k != "children"}
    return without_children == {k: v for k, v in full.items() if k != "children"}


def path_to_url(act_url: str, segments: tuple[str, ...]) -> str:
    """The ELI URL that serves the article at this struct path."""
    return f"{act_url}/text.html/{'/'.join(segments)}"
