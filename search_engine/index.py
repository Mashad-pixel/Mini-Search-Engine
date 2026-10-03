"""Tantivy-backed search index with native facets and highlighting."""

from __future__ import annotations

import html
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

import tantivy

TEXT_ANALYZER: Final[str] = "en_stem"
"""Tantivy's built-in English analyzer: tokenize, lowercase, Snowball stem."""

_WRITER_HEAP_BYTES: Final[int] = 100_000_000
_AGG_MAX_CATEGORIES: Final[int] = 1000
_UNKNOWN_CATEGORY: Final[str] = "Uncategorized"

_CURLY_QUOTES: Final[dict[int, str]] = str.maketrans(
    {"\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"'}
)
# A double-quoted phrase that is not already prefixed (+, -, field:, escape).
_BARE_PHRASE: Final[re.Pattern[str]] = re.compile(r'(?<![\w:+\-\\])"[^"]+"')


class SearchHit(NamedTuple):
    """A single ranked hit returned by :class:`TantivyIndex`.

    Attributes:
        doc_id: The document identifier.
        score: Tantivy's BM25 relevance score.
        snippet: HTML-safe excerpt with matches wrapped in ``<mark>`` tags.
    """

    doc_id: str
    score: float
    snippet: str


def _facet(category: str) -> tantivy.Facet:
    """Build a one-level facet (``/<category>``) with safe escaping.

    Args:
        category: A non-empty category name.

    Returns:
        The corresponding Tantivy facet.
    """
    escaped = category.replace("\\", "\\\\").replace("/", "\\/")
    return tantivy.Facet.from_string("/" + escaped)


def _clean_categories(
    category_filter: str | Sequence[str] | None,
) -> list[str]:
    """Normalize a category filter into a list of non-empty names.

    Args:
        category_filter: One name, several names, or ``None``.

    Returns:
        De-duplicated, stripped names in input order.
    """
    if category_filter is None:
        return []
    names = [category_filter] if isinstance(category_filter, str) else category_filter
    return list(dict.fromkeys(n.strip() for n in names if n and n.strip()))


class TantivyIndex:
    """A disk-backed Tantivy index with BM25 ranking, facets and highlighting.

    Schema:
        * ``doc_id``: raw (untokenized) string, stored and indexed.
        * ``text``: full text, stored and indexed with the English stemmer
          and positions (so phrase queries work).
        * ``category``: native Tantivy facet (``/<category>``), indexed.

    Writes are buffered until :meth:`commit`; searches only see committed
    documents.
    """

    def __init__(self, directory: str | Path, *, create: bool = True) -> None:
        """Open an existing index, or create one if allowed.

        Args:
            directory: Index directory.
            create: Create the index when it does not exist yet.

        Raises:
            FileNotFoundError: If the index is missing and ``create`` is false.
            ValueError: If an index exists but has a different schema.
        """
        self._directory: Path = Path(directory)
        self._schema: tantivy.Schema = self._build_schema()
        self._writer: tantivy.IndexWriter | None = None

        exists = self._directory.is_dir() and tantivy.Index.exists(
            str(self._directory)
        )
        if not exists:
            if not create:
                raise FileNotFoundError(f"No Tantivy index in {self._directory}")
            self._directory.mkdir(parents=True, exist_ok=True)

        self._index: tantivy.Index = tantivy.Index(
            self._schema, path=str(self._directory), reuse=True
        )

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_schema() -> tantivy.Schema:
        """Create the index schema.

        Returns:
            The finished schema.
        """
        builder = tantivy.SchemaBuilder()
        builder.add_text_field("doc_id", stored=True, tokenizer_name="raw")
        builder.add_text_field("text", stored=True, tokenizer_name=TEXT_ANALYZER)
        builder.add_facet_field("category")
        return builder.build()

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def _get_writer(self) -> tantivy.IndexWriter:
        """Return the shared writer, creating it on first use.

        Returns:
            The active index writer.
        """
        if self._writer is None:
            self._writer = self._index.writer(
                heap_size=_WRITER_HEAP_BYTES, num_threads=1
            )
        return self._writer

    def add_document(
        self,
        doc_id: str,
        text: str,
        category: str,
        *,
        replace: bool = True,
    ) -> None:
        """Queue a document for indexing (visible after :meth:`commit`).

        Args:
            doc_id: Unique document identifier.
            text: Full document text.
            category: Category name; becomes the facet ``/<category>``.
            replace: Delete any existing document with the same ``doc_id``.

        Raises:
            ValueError: If ``doc_id`` or ``category`` is empty, or the
                category contains a NUL character.
        """
        category = category.strip()
        if not doc_id:
            raise ValueError("doc_id must not be empty.")
        if not category or "\x00" in category:
            raise ValueError("category must be non-empty and contain no NUL.")

        writer = self._get_writer()
        if replace:
            writer.delete_documents_by_term("doc_id", doc_id)

        document = tantivy.Document()
        document.add_text("doc_id", doc_id)
        document.add_text("text", text)
        document.add_facet("category", _facet(category))
        writer.add_document(document)

    def commit(self) -> None:
        """Make all queued writes durable and visible to searches."""
        if self._writer is not None:
            self._writer.commit()
        self._index.reload()

    def close(self) -> None:
        """Commit, wait for background merges, and release the writer lock."""
        if self._writer is not None:
            self._writer.commit()
            self._writer.wait_merging_threads()
            self._writer = None
        self._index.reload()

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    @property
    def directory(self) -> Path:
        """The directory holding the index files."""
        return self._directory

    @property
    def num_docs(self) -> int:
        """Number of committed documents."""
        return int(self._index.searcher().num_docs)

    @property
    def num_segments(self) -> int:
        """Number of committed segments."""
        return int(self._index.searcher().num_segments)

    def category_counts(self) -> dict[str, int]:
        """Count committed documents per category with a native aggregation.

        Returns:
            ``{category: document_count}`` ordered by count (descending),
            then name.
        """
        searcher = self._index.searcher()
        if searcher.num_docs == 0:
            return {}
        aggregation = {
            "categories": {
                "terms": {"field": "category", "size": _AGG_MAX_CATEGORIES}
            }
        }
        result = searcher.aggregate(tantivy.Query.all_query(), aggregation)
        counts = {
            str(bucket["key"]): int(bucket["doc_count"])
            for bucket in result["categories"]["buckets"]
        }
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    # ------------------------------------------------------------------ #
    # Searching
    # ------------------------------------------------------------------ #
    def _parse(self, query: str) -> tantivy.Query | None:
        """Parse user input with Tantivy's query parser (never raises).

        Curly quotes are normalized, and every bare ``"quoted phrase"`` is
        made mandatory (``+"..."``), so ``python "machine learning"`` ranks
        on both terms but only returns documents containing the phrase.
        Malformed syntax is handled leniently.

        Args:
            query: Raw user query.

        Returns:
            A parsed query, or ``None`` if the input is blank.
        """
        normalized = query.translate(_CURLY_QUOTES).strip()
        if not normalized:
            return None
        normalized = _BARE_PHRASE.sub(lambda m: "+" + m.group(0), normalized)
        parsed, _errors = self._index.parse_query_lenient(normalized, ["text"])
        return parsed

    def _with_category_filter(
        self, text_query: tantivy.Query, categories: Sequence[str]
    ) -> tantivy.Query:
        """Restrict a query to documents in any of the given categories.

        The filter is wrapped in a zero-score clause so it never changes
        BM25 scores.

        Args:
            text_query: The parsed text query.
            categories: Cleaned category names (OR semantics).

        Returns:
            The combined query (``text_query`` itself if no categories).
        """
        if not categories:
            return text_query
        facet_query = tantivy.Query.boolean_query(
            [
                (
                    tantivy.Occur.Should,
                    tantivy.Query.term_query(self._schema, "category", _facet(name)),
                )
                for name in categories
            ]
        )
        return tantivy.Query.boolean_query(
            [
                (tantivy.Occur.Must, text_query),
                (tantivy.Occur.Must, tantivy.Query.const_score_query(facet_query, 0.0)),
            ]
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        category_filter: str | Sequence[str] | None = None,
        fragment_size: int = 150,
    ) -> list[SearchHit]:
        """Run a BM25 search with optional category filtering.

        Args:
            query: Free-text query; quoted segments are exact phrases.
            top_k: Maximum number of hits.
            category_filter: One category or several (OR); ``None`` for all.
            fragment_size: Maximum snippet length in characters.

        Returns:
            Hits sorted by descending score, each with a highlighted snippet.
        """
        if top_k <= 0:
            return []
        text_query = self._parse(query)
        if text_query is None:
            return []

        full_query = self._with_category_filter(
            text_query, _clean_categories(category_filter)
        )
        searcher = self._index.searcher()
        result = searcher.search(full_query, limit=top_k, count=False)
        if not result.hits:
            return []

        generator = self._snippet_generator(searcher, text_query, fragment_size)
        hits: list[SearchHit] = []
        for score, address in result.hits:
            doc = searcher.doc(address)
            hits.append(
                SearchHit(
                    doc_id=str(doc.get_first("doc_id")),
                    score=float(score),
                    snippet=self._render_snippet(generator, doc, fragment_size),
                )
            )
        return hits

    # ------------------------------------------------------------------ #
    # Highlighting
    # ------------------------------------------------------------------ #
    def _snippet_generator(
        self,
        searcher: tantivy.Searcher,
        text_query: tantivy.Query,
        fragment_size: int,
    ) -> tantivy.SnippetGenerator:
        """Create Tantivy's built-in highlighter for the ``text`` field.

        Args:
            searcher: The searcher the query ran against.
            text_query: The parsed text query (without filters).
            fragment_size: Maximum fragment length in characters.

        Returns:
            A configured snippet generator.
        """
        generator = tantivy.SnippetGenerator.create(
            searcher, text_query, self._schema, "text"
        )
        generator.set_max_num_chars(fragment_size)
        return generator

    @staticmethod
    def _render_snippet(
        generator: tantivy.SnippetGenerator,
        doc: tantivy.Document,
        fragment_size: int,
    ) -> str:
        """Turn Tantivy's highlight ranges into escaped HTML with ``<mark>``.

        Tantivy reports highlight ranges as UTF-8 byte offsets, so slicing is
        done on bytes. Adjacent matches separated only by whitespace are
        merged into a single ``<mark>``. All text is HTML-escaped.

        Args:
            generator: Snippet generator bound to the query.
            doc: The stored document.
            fragment_size: Fallback excerpt length if nothing matched.

        Returns:
            An HTML string safe to insert with ``innerHTML``.
        """
        text = str(doc.get_first("text") or "")
        snippet = generator.snippet_from_doc(doc)
        fragment = snippet.fragment()
        if not fragment:
            plain = " ".join(text.split())
            suffix = " \u2026" if len(plain) > fragment_size else ""
            return html.escape(plain[:fragment_size]) + suffix

        raw = fragment.encode("utf-8")
        merged: list[tuple[int, int]] = []
        for span in sorted(snippet.highlighted(), key=lambda r: (r.start, r.end)):
            if merged and (
                span.start <= merged[-1][1]
                or not raw[merged[-1][1] : span.start].strip()
            ):
                merged[-1] = (merged[-1][0], max(merged[-1][1], span.end))
            else:
                merged.append((span.start, span.end))

        parts: list[str] = []
        cursor = 0
        for start, end in merged:
            parts.append(html.escape(raw[cursor:start].decode("utf-8")))
            parts.append(f"<mark>{html.escape(raw[start:end].decode('utf-8'))}</mark>")
            cursor = end
        parts.append(html.escape(raw[cursor:].decode("utf-8")))

        offset = text.find(fragment)
        prefix = "\u2026 " if offset > 0 else ""
        suffix = (
            " \u2026"
            if offset >= 0 and offset + len(fragment) < len(text)
            else ""
        )
        return prefix + "".join(parts) + suffix

    def highlight(
        self, doc_id: str, query: str, fragment_size: int = 150
    ) -> str | None:
        """Highlight one stored document for a query.

        Args:
            doc_id: Identifier of the document to excerpt.
            query: Free-text query (same syntax as :meth:`search`).
            fragment_size: Maximum snippet length in characters.

        Returns:
            HTML with ``<mark>`` around matching terms, or ``None`` if the
            document does not exist (or the query is blank).
        """
        text_query = self._parse(query)
        if text_query is None or fragment_size <= 0:
            return None

        searcher = self._index.searcher()
        lookup = tantivy.Query.term_query(self._schema, "doc_id", doc_id)
        result = searcher.search(lookup, limit=1, count=False)
        if not result.hits:
            return None

        doc = searcher.doc(result.hits[0][1])
        generator = self._snippet_generator(searcher, text_query, fragment_size)
        return self._render_snippet(generator, doc, fragment_size)

    # ------------------------------------------------------------------ #
    # Document lookup
    # ------------------------------------------------------------------ #
    def get_document(self, doc_id: str) -> dict[str, str] | None:
        """Fetch one committed document by its identifier.

        Facet values are indexed but not stored, so the category is recovered
        by checking which category facet the document matches.

        Args:
            doc_id: Identifier of the document.

        Returns:
            ``{"doc_id", "text", "category"}``, or ``None`` if no document has
            this id. ``category`` is ``"Uncategorized"`` if none matches.
        """
        if not doc_id:
            return None

        searcher = self._index.searcher()
        lookup = tantivy.Query.term_query(self._schema, "doc_id", doc_id)
        result = searcher.search(lookup, limit=1, count=False)
        if not result.hits:
            return None

        doc = searcher.doc(result.hits[0][1])
        return {
            "doc_id": str(doc.get_first("doc_id")),
            "text": str(doc.get_first("text") or ""),
            "category": self._category_of(searcher, lookup),
        }

    def _category_of(self, searcher: tantivy.Searcher, lookup: tantivy.Query) -> str:
        """Find which category facet the document matched by ``lookup`` has.

        Args:
            searcher: The searcher used for the lookup.
            lookup: A query matching exactly the one document.

        Returns:
            The category name, or ``"Uncategorized"`` if none matches.
        """
        for name in self.category_counts():
            combined = tantivy.Query.boolean_query(
                [
                    (tantivy.Occur.Must, lookup),
                    (
                        tantivy.Occur.Must,
                        tantivy.Query.term_query(self._schema, "category", _facet(name)),
                    ),
                ]
            )
            if searcher.search(combined, limit=1, count=False).hits:
                return name
        return _UNKNOWN_CATEGORY

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_to_disk(self, directory_path: str | Path) -> None:
        """Flush the index to disk, optionally copying it elsewhere.

        Tantivy persists segments natively in its index directory, so saving
        to the index's own directory is just a commit. Saving to another
        directory commits, waits for merges, and copies the index files.

        Args:
            directory_path: Destination directory.

        Raises:
            FileExistsError: If the destination is a different, non-empty
                directory.
        """
        self.close()
        target = Path(directory_path)
        if target.resolve() == self._directory.resolve():
            return
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"Destination is not empty: {target}")
        shutil.copytree(self._directory, target, dirs_exist_ok=True)

    @classmethod
    def load_from_disk(cls, directory_path: str | Path) -> TantivyIndex:
        """Open an index previously built in ``directory_path``.

        Args:
            directory_path: Index directory.

        Returns:
            A ready-to-search index.

        Raises:
            FileNotFoundError: If no index exists there.
            ValueError: If the index schema does not match.
        """
        return cls(directory_path, create=False)