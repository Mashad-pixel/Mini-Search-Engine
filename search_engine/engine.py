"""High-level search engine wrapping the Tantivy index."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

from search_engine.index import TantivyIndex

DEFAULT_INDEX_DIR: Final[Path] = Path("data") / "tantivy_index"
DEFAULT_CATEGORY: Final[str] = "Uncategorized"
SNIPPET_WIDTH: Final[int] = 150


class SearchResult(NamedTuple):
    """A single ranked search hit.

    Attributes:
        doc_id: The document identifier.
        score: BM25 relevance score.
        snippet: HTML-safe excerpt with matches wrapped in ``<mark>`` tags.
    """

    doc_id: str
    score: float
    snippet: str


class SearchEngine:
    """Indexes documents and answers ranked, filterable, highlighted queries."""

    def __init__(
        self, directory: str | Path = DEFAULT_INDEX_DIR, *, create: bool = True
    ) -> None:
        """Open (or create) the engine's Tantivy index.

        Args:
            directory: Directory holding the index files.
            create: Create the index if it does not exist yet.

        Raises:
            FileNotFoundError: If the index is missing and ``create`` is false.
        """
        self.index: TantivyIndex = TantivyIndex(directory, create=create)

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def add_document(
        self, doc_id: str, text: str, category: str = DEFAULT_CATEGORY
    ) -> None:
        """Queue a document for indexing (call :meth:`commit` to publish).

        Args:
            doc_id: Unique document identifier (re-adding replaces it).
            text: Full document text.
            category: Facet category for filtering.
        """
        self.index.add_document(doc_id, text, category)

    def commit(self) -> None:
        """Make all queued documents durable and searchable."""
        self.index.commit()

    def close(self) -> None:
        """Commit pending work and release the index writer."""
        self.index.close()

    # ------------------------------------------------------------------ #
    # Searching
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int = 10,
        category_filter: str | Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """Search for documents matching a query.

        Quoted segments are exact phrases that every returned document must
        contain; unquoted terms are ranked with BM25.

        Args:
            query: The user's query string.
            top_k: Maximum number of results.
            category_filter: One category or several (OR), or ``None``.

        Returns:
            Ranked results with scores and ``<mark>``-highlighted snippets.
        """
        hits = self.index.search(
            query,
            top_k=top_k,
            category_filter=category_filter,
            fragment_size=SNIPPET_WIDTH,
        )
        return [
            SearchResult(
                hit.doc_id,
                hit.score,
                hit.snippet or self._snippet(hit.doc_id, query),
            )
            for hit in hits
        ]

    def _snippet(self, doc_id: str, query: str, width: int = SNIPPET_WIDTH) -> str:
        """Build a highlighted excerpt using Tantivy's built-in highlighter.

        Args:
            doc_id: Identifier of the document to excerpt.
            query: The user's query string.
            width: Maximum snippet length in characters.

        Returns:
            HTML with ``<mark>`` tags, or an empty string if unavailable.
        """
        return self.index.highlight(doc_id, query, fragment_size=width) or ""

    def highlight(self, doc_id: str, query: str, width: int = SNIPPET_WIDTH) -> str:
        """``<mark>``-highlighted excerpt of one document for ``query`` ('' if none)."""
        return self._snippet(doc_id, query, width)

    def get_document(self, doc_id: str) -> dict[str, str] | None:
        """Fetch one document by id.

        Args:
            doc_id: Identifier of the document.

        Returns:
            ``{"doc_id", "text", "category"}``, or ``None`` if missing.
        """
        return self.index.get_document(doc_id)

    def categories(self) -> dict[str, int]:
        """Count documents per category.

        Returns:
            ``{category: document_count}``, largest first.
        """
        return self.index.category_counts()

    @property
    def num_docs(self) -> int:
        """Number of searchable (committed) documents."""
        return self.index.num_docs

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_to_disk(self, directory_path: str | Path) -> None:
        """Flush the index to disk, optionally copying it to a new directory.

        Args:
            directory_path: Destination directory.
        """
        self.index.save_to_disk(directory_path)

    @classmethod
    def load_from_disk(cls, directory_path: str | Path) -> SearchEngine:
        """Open an index previously built in ``directory_path``.

        Args:
            directory_path: Index directory.

        Returns:
            A ready-to-query ``SearchEngine``.

        Raises:
            FileNotFoundError: If no index exists there.
        """
        return cls(directory_path, create=False)