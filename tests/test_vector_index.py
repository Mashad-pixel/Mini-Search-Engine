"""Tests for the FAISS-backed :class:`VectorIndex`.

The real MiniLM model is never loaded: ``VectorIndex.encode`` is replaced by a
deterministic hashed bag-of-words embedding, so the tests need neither network
access nor a model cache but still exercise FAISS search, ordering, category
filtering and persistence. They are marked ``slow`` because importing
``faiss`` and ``sentence_transformers`` (via ``vector_index``) is heavy.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

from search_engine.vector_index import EMBEDDING_DIM, VectorIndex  # noqa: E402

pytestmark = pytest.mark.slow

DOC_IDS = ["ml", "volcano", "river"]
TEXTS = [
    "Machine learning models learn patterns from data.",
    "Volcanoes erupt molten rock and ash.",
    "A river carries water and sediment to the sea.",
]
CATEGORIES = ["Technology", "Science", "Geography"]


def fake_encode(self, texts, batch_size=64, show_progress=False):
    """Hash each word into one of ``EMBEDDING_DIM`` buckets; L2-normalise."""
    vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            vectors[row, zlib.crc32(word.encode()) % EMBEDDING_DIM] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors / np.where(norms == 0, 1.0, norms), dtype=np.float32)


@pytest.fixture
def index(monkeypatch):
    monkeypatch.setattr(VectorIndex, "encode", fake_encode)
    idx = VectorIndex()
    idx.add_documents(DOC_IDS, TEXTS, categories=CATEGORIES)
    return idx


def test_len_counts_documents(index):
    assert len(index) == 3


def test_search_ranks_best_match_first(index):
    hits = index.search("machine learning", top_k=3)
    assert [doc_id for doc_id, _ in hits][0] == "ml"
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(
        float(fake_encode(None, ["machine learning"])[0] @ fake_encode(None, [TEXTS[0]])[0]),
        abs=1e-5,
    )


def test_search_other_topics_rank_first(index):
    assert index.search("volcanoes erupt", top_k=1)[0][0] == "volcano"
    assert index.search("river sediment sea", top_k=1)[0][0] == "river"


def test_top_k_limits_results(index):
    assert len(index.search("machine learning", top_k=2)) == 2


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_returns_nothing(index, query):
    assert index.search(query) == []


def test_non_positive_top_k_returns_nothing(index):
    assert index.search("machine learning", top_k=0) == []
    assert index.search("machine learning", top_k=-1) == []


def test_category_filter_excludes_other_categories(index):
    hits = index.search("machine learning", top_k=3, categories=["Science", "Geography"])
    assert {doc_id for doc_id, _ in hits} == {"volcano", "river"}


def test_duplicate_ids_are_rejected(index):
    with pytest.raises(ValueError, match="Duplicate"):
        index.add_documents(["ml"], ["anything"])


def test_length_mismatch_is_rejected(index):
    with pytest.raises(ValueError, match="same length"):
        index.add_documents(["x", "y"], ["only one text"])


def test_snippet_is_html_escaped_and_unknown_id_is_empty(monkeypatch):
    monkeypatch.setattr(VectorIndex, "encode", fake_encode)
    idx = VectorIndex()
    idx.add_documents(["x"], ["a <script>alert(1)</script> b"])
    assert "<script>" not in idx.snippet("x")
    assert "&lt;script&gt;" in idx.snippet("x")
    assert idx.snippet("missing") == ""


def test_save_and_load_round_trip(index, tmp_path):
    index.save_to_disk(tmp_path / "vec")
    loaded = VectorIndex.load_from_disk(tmp_path / "vec")
    assert len(loaded) == 3
    expected = index.search("machine learning", top_k=1)
    actual = loaded.search("machine learning", top_k=1)
    assert [doc_id for doc_id, _ in actual] == [doc_id for doc_id, _ in expected] == ["ml"]
    assert actual[0][1] == pytest.approx(expected[0][1])
    assert loaded.search("volcanoes", top_k=1, categories=["Science"])[0][0] == "volcano"


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorIndex.load_from_disk(tmp_path / "nothing")