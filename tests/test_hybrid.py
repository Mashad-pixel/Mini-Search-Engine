from types import SimpleNamespace

import pytest

from search_engine.hybrid_engine import RRF_K, HybridEngine


class FakeLexical:
    def __init__(self, ids):
        self.ids = ids
        self.last_filter = "unset"

    def search(self, query, top_k, category_filter=None):
        self.last_filter = category_filter
        return [SimpleNamespace(doc_id=d, score=1.0, snippet=f"<mark>{d}</mark>")
                for d in self.ids][:top_k]

    def highlight(self, doc_id, query):
        return ""

    def categories(self):
        return {}

    def get_document(self, doc_id):
        return {"doc_id": doc_id, "text": "t", "category": "c"} if doc_id == "a" else None


class FakeVector:
    def __init__(self, ids):
        self.ids = ids
        self.last_categories = "unset"

    def search(self, query, top_k, categories=None):
        self.last_categories = categories
        return [(d, 0.5) for d in self.ids][:top_k]

    def snippet(self, doc_id):
        return f"plain {doc_id}"


def make(lex, vec):
    return HybridEngine(FakeLexical(lex), FakeVector(vec))


def test_rrf_fusion_order_and_scores():
    hits = make(["a", "b"], ["b", "c"]).search("q", top_k=3)
    assert [h.doc_id for h in hits] == ["b", "a", "c"]
    assert hits[0].score == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert hits[2].score == pytest.approx(1 / (RRF_K + 2))


def test_top_k_truncates():
    assert len(make(["a", "b"], ["c", "d"]).search("q", top_k=2)) == 2


def test_snippet_prefers_lexical_then_plain_fallback():
    by_id = {h.doc_id: h.snippet for h in make(["a"], ["a", "z"]).search("q")}
    assert by_id["a"] == "<mark>a</mark>"
    assert by_id["z"] == "plain z"


def test_semantic_hit_uses_lexical_highlight_when_it_has_marks():
    class Marking(FakeLexical):
        def highlight(self, doc_id, query):
            return f"x <mark>{query}</mark>"

    engine = HybridEngine(Marking([]), FakeVector(["z"]))
    assert engine.search("q", mode="semantic")[0].snippet == "x <mark>q</mark>"


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_empty_query_returns_nothing(query, mode):
    assert make(["a"], ["b"]).search(query, mode=mode) == []


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        make([], []).search("q", mode="bogus")


def test_get_document_delegates_to_lexical():
    engine = make(["a"], [])
    assert engine.get_document("a")["category"] == "c"
    assert engine.get_document("zzz") is None


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_empty_category_list_means_no_filter():
    unfiltered = make(["a", "b"], ["b", "c"])
    empty = make(["a", "b"], ["b", "c"])
    assert empty.search("q", mode="hybrid", category_filter=[]) == unfiltered.search(
        "q", mode="hybrid"
    )
    assert empty.lexical.last_filter is None
    assert empty.vector.last_categories is None


def test_non_empty_category_list_is_forwarded_to_both_indexes():
    engine = make(["a"], ["b"])
    engine.search("q", mode="hybrid", category_filter=["Science"])
    assert engine.lexical.last_filter == ["Science"]
    assert engine.vector.last_categories == ["Science"]


@pytest.mark.parametrize("top_k", [0, -1])
@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_non_positive_top_k_returns_nothing(top_k, mode):
    assert make(["a"], ["b"]).search("q", top_k=top_k, mode=mode) == []


def test_rrf_ties_are_broken_deterministically():
    # "a" and "b" swap ranks between the two lists, so their fused scores are equal.
    first = make(["a", "b"], ["b", "a"]).search("q", top_k=2)
    second = make(["a", "b"], ["b", "a"]).search("q", top_k=2)
    assert first[0].score == pytest.approx(first[1].score)
    assert [h.doc_id for h in first] == [h.doc_id for h in second] == ["a", "b"]


def test_rrf_ties_between_disjoint_lists_favour_the_lexical_hit():
    # "a" is only lexical rank 1 and "b" only semantic rank 1: equal scores, lexical first.
    runs = [[h.doc_id for h in make(["a"], ["b"]).search("q")] for _ in range(2)]
    assert runs[0] == runs[1] == ["a", "b"]