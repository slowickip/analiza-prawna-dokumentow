"""Sparse BM25 weights with diacritic folding.

Qdrant supplies IDF at query time; stored vectors contain term-frequency weights.
"""

import hashlib
import re
import unicodedata
from collections import Counter

# Version tokenisation, term IDs and BM25 constants together to prevent mixed encodings.
SPARSE_POLICY_VERSION = "bm25-idf-v1"
# Chosen by argument, not measurement: conventional BM25 defaults.
BM25_K1 = 1.2
BM25_B = 0.75
# Chosen by argument, not measurement: 256 tokens represents an article-sized unit.
# A fixed average permits streaming seeding before corpus-wide length statistics
# are available.
BM25_AVG_UNIT_TOKENS = 256.0
# NFKD does not decompose Polish ł/Ł, so fold them explicitly.
_SPARSE_LETTER_FOLD = str.maketrans({"ł": "l", "Ł": "l"})
_SPARSE_TOKEN = re.compile(r"[0-9a-z]+")


def sparse_terms(text: str) -> list[str]:
    """Tokenise for the sparse limb: casefold, strip diacritics, split on the rest."""
    lowered = text.casefold().translate(_SPARSE_LETTER_FOLD)
    folded = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    return _SPARSE_TOKEN.findall(stripped)


def sparse_term_id(term: str) -> int:
    """Hash terms into 32-bit indices without building a shared vocabulary.

    Hash collisions merge the affected terms' postings.
    """
    return int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:4], "big")


def sparse_document_vector(text: str) -> dict[int, float]:
    """BM25 term weights for one unit, without the IDF factor Qdrant applies."""
    terms = sparse_terms(text)
    if not terms:
        return {}
    length_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * len(terms) / BM25_AVG_UNIT_TOKENS)
    counts = Counter(sparse_term_id(term) for term in terms)
    return {
        term_id: count * (BM25_K1 + 1.0) / (count + length_norm)
        for term_id, count in counts.items()
    }


def sparse_query_vector(query: str) -> dict[int, float]:
    """Query-side weights: presence only, so scoring stays with the stored weights."""
    return dict.fromkeys((sparse_term_id(term) for term in sparse_terms(query)), 1.0)
