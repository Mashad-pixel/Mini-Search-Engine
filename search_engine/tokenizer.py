"""Tokenization: regex splitting, stop-word removal, and English stemming."""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Final

import snowballstemmer  # type: ignore[import-untyped]

TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[^\W_]+(?:['\u2019][^\W_]+)*"
)

STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can",
        "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
        "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
        "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
        "having", "he", "her", "here", "hers", "herself", "him", "himself",
        "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "me", "more", "most", "my", "myself", "no", "nor",
        "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "she",
        "should", "shouldn't", "so", "some", "such", "than", "that", "the",
        "their", "theirs", "them", "themselves", "then", "there", "these",
        "they", "this", "those", "through", "to", "too", "under", "until",
        "up", "very", "was", "wasn't", "we", "were", "weren't", "what",
        "when", "where", "which", "while", "who", "whom", "why", "with",
        "won't", "would", "wouldn't", "you", "your", "yours", "yourself",
        "yourselves",
    }
)

_STEMMER = snowballstemmer.stemmer("english")


def _normalize(raw: str) -> str | None:
    """Lowercase, unify apostrophes, drop stop words, then stem.

    Args:
        raw: A raw token matched by ``TOKEN_PATTERN``.

    Returns:
        The stemmed token, or ``None`` if it is a stop word.
    """
    word = raw.lower().replace("\u2019", "'")
    if word in STOP_WORDS:
        return None
    return str(_STEMMER.stemWord(word))


def tokenize(text: str) -> list[str]:
    """Convert text into a list of normalized, stemmed tokens.

    Pipeline: regex split -> lowercase -> stop-word removal -> stemming.

    Args:
        text: Arbitrary input text.

    Returns:
        Stemmed tokens in document order (duplicates preserved).
    """
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        token = _normalize(match.group())
        if token is not None:
            tokens.append(token)
    return tokens


def find_term_spans(
    text: str, terms: Collection[str]
) -> list[tuple[int, int]]:
    """Locate original-text words whose stem is in ``terms``.

    Used for snippet generation: stemmed query terms are mapped back to
    character offsets in the unstemmed source text.

    Args:
        text: The original document text.
        terms: Stemmed query terms.

    Returns:
        ``(start, end)`` character offsets, in document order.
    """
    wanted = set(terms)
    spans: list[tuple[int, int]] = []
    for match in TOKEN_PATTERN.finditer(text):
        token = _normalize(match.group())
        if token is not None and token in wanted:
            spans.append(match.span())
    return spans