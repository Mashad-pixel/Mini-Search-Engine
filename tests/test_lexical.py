"""Tests for :class:`SearchEngine` backed by a real Tantivy index.

Every test builds a real on-disk index in ``tmp_path``, so the module is marked
``slow`` and excluded from the fast CI run (``pytest -m "not slow"``).

Assumptions (covered by tests rather than changed in the implementation):
a malformed query such as ``"(unclosed"`` is parsed leniently and never
raises, an empty category filter means "no filter", and ``get_document("")``
returns ``None``.
"""

from __future__ import annotations

import pytest

from search_engine import SearchEngine

pytestmark = pytest.mark.slow


@pytest.fixture
def engine(tmp_path):
    eng = SearchEngine(tmp_path / "idx")
    eng.add_document("d1", "Machine learning models learn from data.", "Technology")
    eng.add_document("d2", "Learning how a machine works is hard.", "History")
    eng.add_document("d3", "Volcanoes erupt molten rock.", "Science")
    eng.commit()
    yield eng
    eng.close()


def ids(hits):
    return {h.doc_id for h in hits}


def test_phrase_search_requires_adjacent_words(engine):
    assert ids(engine.search('"machine learning"')) == {"d1"}


def test_unquoted_terms_match_both(engine):
    assert {"d1", "d2"} <= ids(engine.search("machine learning"))


def test_category_filter_single_and_multiple(engine):
    assert ids(engine.search("machine", category_filter="History")) == {"d2"}
    both = engine.search("machine", category_filter=["History", "Technology"])
    assert ids(both) == {"d1", "d2"}


def test_category_counts(engine):
    counts = engine.categories()
    assert len(counts) == 3
    assert sorted(counts.values()) == [1, 1, 1]


def test_highlight_marks_matches(engine):
    assert "<mark>" in engine.highlight("d1", "machine")


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_returns_nothing(engine, query):
    assert engine.search(query) == []


def test_get_document_returns_text_and_category(engine):
    assert engine.get_document("d1") == {
        "doc_id": "d1",
        "text": "Machine learning models learn from data.",
        "category": "Technology",
    }


def test_get_document_missing_returns_none(engine):
    assert engine.get_document("missing") is None


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_empty_query_with_top_k_returns_nothing(engine):
    assert engine.search("", top_k=10) == []


def test_zero_top_k_returns_nothing(engine):
    assert engine.search("q", top_k=0) == []


def test_negative_top_k_returns_nothing(engine):
    assert engine.search("q", top_k=-1) == []


def test_empty_string_category_filter_behaves_like_no_filter(engine):
    assert ids(engine.search("machine", category_filter="")) == ids(engine.search("machine"))
    assert ids(engine.search("machine", category_filter="")) == {"d1", "d2"}


def test_blank_category_filter_list_behaves_like_no_filter(engine):
    assert ids(engine.search("machine", category_filter=["", " "])) == ids(engine.search("machine"))


def test_get_document_empty_id_returns_none(engine):
    assert engine.get_document("") is None


def test_malformed_query_does_not_raise(engine):
    assert isinstance(engine.search("(unclosed"), list)


def test_unicode_query_does_not_raise(engine):
    assert isinstance(engine.search("naïve"), list)