"""Hybrid retrieval: BM25 (Tantivy) + dense vectors (FAISS) fused with RRF."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from search_engine.engine import SearchEngine

if TYPE_CHECKING:  # heavy imports (torch, faiss) only when actually loading
    from search_engine.vector_index import VectorIndex

RRF_K = 60
CANDIDATES = 50
MODES = ("lexical", "semantic", "hybrid")


@dataclass(frozen=True)
class HybridHit:
    """A ranked result. ``snippet`` is HTML-escaped; only ``<mark>`` is markup."""

    doc_id: str
    score: float
    snippet: str


class HybridEngine:
    """Combines the lexical ``SearchEngine`` with a semantic ``VectorIndex``."""

    def __init__(self, lexical: SearchEngine, vector: "VectorIndex") -> None:
        self.lexical = lexical
        self.vector = vector

    @classmethod
    def load_from_disk(cls, tantivy_dir: str | Path, vector_dir: str | Path) -> "HybridEngine":
        """Open both indexes.

        Raises:
            FileNotFoundError: If either index has not been built.
        """
        from search_engine.vector_index import VectorIndex

        lexical = SearchEngine.load_from_disk(Path(tantivy_dir))
        try:
            vector = VectorIndex.load_from_disk(vector_dir)
        except FileNotFoundError:
            lexical.close()
            raise
        return cls(lexical, vector)

    def _snippet(self, doc_id: str, query: str, lexical_snippet: str | None = None) -> str:
        """Best available snippet: lexical hit snippet, else Tantivy highlight, else plain.

        Semantic hits often share no literal terms with the query; then the
        highlighter yields no ``<mark>`` and we fall back to the plain preview.
        """
        if lexical_snippet:
            return lexical_snippet
        try:
            highlighted = self.lexical.highlight(doc_id, query)
        except ValueError:  # a query the lexical side cannot handle
            highlighted = ""
        if "<mark>" in highlighted:
            return highlighted
        return self.vector.snippet(doc_id)

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        category_filter: Sequence[str] | None = None,
    ) -> list[HybridHit]:
        """Search using ``mode`` in {"lexical", "semantic", "hybrid"}.

        Returns an empty list for a blank query or ``top_k <= 0``.
        """
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}.")
        if not query.strip() or top_k <= 0:
            return []
        categories = list(category_filter) if category_filter else None

        if mode == "lexical":
            return [
                HybridHit(hit.doc_id, float(hit.score), hit.snippet)
                for hit in self.lexical.search(query, top_k=top_k, category_filter=categories)
            ]

        pool = max(CANDIDATES, top_k)

        if mode == "semantic":
            semantic = self.vector.search(query, top_k=top_k, categories=categories)[:top_k]
            return [
                HybridHit(doc_id, score, self._snippet(doc_id, query))
                for doc_id, score in semantic
            ]

        lexical_hits = self.lexical.search(query, top_k=pool, category_filter=categories)
        semantic_hits = self.vector.search(query, top_k=pool, categories=categories)

        fused: dict[str, float] = {}
        lexical_snippets: dict[str, str] = {}
        for rank, hit in enumerate(lexical_hits, start=1):
            fused[hit.doc_id] = fused.get(hit.doc_id, 0.0) + 1.0 / (RRF_K + rank)
            lexical_snippets[hit.doc_id] = hit.snippet
        for rank, (doc_id, _score) in enumerate(semantic_hits, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            HybridHit(doc_id, score, self._snippet(doc_id, query, lexical_snippets.get(doc_id)))
            for doc_id, score in ranked
        ]

    def get_document(self, doc_id: str) -> dict[str, str] | None:
        """Fetch one document by id from the lexical index.

        Args:
            doc_id: Identifier of the document.

        Returns:
            ``{"doc_id", "text", "category"}``, or ``None`` if missing.
        """
        return self.lexical.get_document(doc_id)

    def categories(self) -> dict[str, int]:
        """Category facet counts from the lexical index."""
        return self.lexical.categories()

    def warmup(self) -> None:
        """Preload the embedding model."""
        self.vector.warmup()

    def close(self) -> None:
        """Release the lexical index."""
        self.lexical.close()