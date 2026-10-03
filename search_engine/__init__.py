"""A small, persistent search engine: Tantivy BM25, FAISS vectors and autocomplete.

``SearchEngine`` is imported eagerly. ``HybridEngine``, ``HybridHit`` and
``SuggestionEngine`` are resolved lazily (PEP 562) so that
``import search_engine`` stays cheap and does not pull in ``torch``/``faiss``
(via ``vector_index``) or ``whoosh`` until one of them is actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from search_engine.engine import SearchEngine, SearchResult

if TYPE_CHECKING:  # for type checkers and IDEs only; never executed at runtime
    from search_engine.hybrid_engine import HybridEngine, HybridHit
    from search_engine.suggester import SuggestionEngine

__all__ = ["SearchEngine", "SearchResult", "HybridEngine", "HybridHit", "SuggestionEngine"]


def __getattr__(name: str) -> object:
    """Resolve the lazily imported public names on first access.

    Args:
        name: The attribute being looked up on the package.

    Returns:
        The requested class.

    Raises:
        AttributeError: If ``name`` is not a lazily exported attribute.
    """
    if name in {"HybridEngine", "HybridHit"}:
        from search_engine import hybrid_engine

        return getattr(hybrid_engine, name)
    if name == "SuggestionEngine":
        from search_engine.suggester import SuggestionEngine

        return SuggestionEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")