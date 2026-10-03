"""Prefix autocomplete backed by a Whoosh edge-n-gram index."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from whoosh import index
from whoosh.fields import ID, NGRAMWORDS, TEXT, Schema
from whoosh.filedb.filestore import FileStorage, RamStorage, copy_storage
from whoosh.query import And, Term

TITLE_CHARS = 80
MIN_PREFIX_CHARS = 2
MIN_NGRAM = 2
MAX_NGRAM = 15
CANDIDATE_FACTOR = 10
MIN_CANDIDATES = 50

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")


def make_schema() -> Schema:
    """Build the suggestion schema.

    ``ngrams`` stores edge n-grams (word prefixes of 2-15 characters) of the
    title, so a typed prefix is a plain term lookup.
    """
    return Schema(
        doc_id=ID(stored=True, unique=True),
        title=TEXT(stored=True),
        ngrams=NGRAMWORDS(minsize=MIN_NGRAM, maxsize=MAX_NGRAM, stored=False, at="start"),
    )


def make_title(text: str) -> str:
    """Return the first ~80 characters of ``text`` as a one-line title.

    Whitespace is collapsed; a cut falls on a word boundary and gets an
    ellipsis.
    """
    flat = _WHITESPACE.sub(" ", text).strip()
    if len(flat) <= TITLE_CHARS:
        return flat
    cut = flat[:TITLE_CHARS]
    if flat[TITLE_CHARS] != " ":
        cut = cut.rsplit(" ", 1)[0] or cut
    return cut.rstrip() + "\u2026"


class SuggestionEngine:
    """Type-ahead suggestions over document titles."""

    def __init__(self) -> None:
        self._ix: index.Index | None = None

    # ------------------------------------------------------------------ #
    # Building / persistence
    # ------------------------------------------------------------------ #
    def build_from_documents(self, doc_ids: Sequence[str], texts: Sequence[str]) -> int:
        """Index the first ~80 characters of each document as its title.

        Documents with empty text are skipped; a repeated ``doc_id`` keeps its
        last text.

        Args:
            doc_ids: Unique document identifiers.
            texts: Document texts, aligned with ``doc_ids``.

        Returns:
            The number of documents indexed.

        Raises:
            ValueError: If the two sequences differ in length.
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length.")

        titles: dict[str, str] = {}
        for doc_id, text in zip(doc_ids, texts):
            title = make_title(text)
            if title:
                titles[str(doc_id)] = title
            else:
                titles.pop(str(doc_id), None)

        if self._ix is not None:
            self._ix.close()
        self._ix = RamStorage().create_index(make_schema())
        writer = self._ix.writer(limitmb=128)
        for doc_id, title in titles.items():
            writer.add_document(doc_id=doc_id, title=title, ngrams=title)
        writer.commit()
        return len(titles)

    def save_to_disk(self, directory: str | Path) -> None:
        """Write the index to ``directory`` as a native Whoosh index.

        An existing Whoosh index in the directory is replaced.

        Raises:
            RuntimeError: If nothing has been built or loaded yet.
            FileExistsError: If the directory holds something other than a
                Whoosh index.
        """
        if self._ix is None:
            raise RuntimeError("Nothing to save; call build_from_documents() first.")

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        storage = FileStorage(str(target))
        if index.exists_in(str(target)):
            for name in storage.list():
                storage.delete_file(name)
        elif any(target.iterdir()):
            raise FileExistsError(f"{target} is not empty and holds no suggestion index.")
        copy_storage(self._ix.storage, storage)

    @classmethod
    def load_from_disk(cls, directory: str | Path) -> "SuggestionEngine":
        """Open an index written by :meth:`save_to_disk`.

        Raises:
            FileNotFoundError: If there is no index in ``directory``.
        """
        path = Path(directory)
        if not path.is_dir() or not index.exists_in(str(path)):
            raise FileNotFoundError(f"No suggestion index in {path}")
        engine = cls()
        engine._ix = index.open_dir(str(path))
        return engine

    @property
    def num_docs(self) -> int:
        """Number of indexed documents."""
        return int(self._ix.doc_count()) if self._ix is not None else 0

    def close(self) -> None:
        """Release the underlying index."""
        if self._ix is not None:
            self._ix.close()
            self._ix = None

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #
    def suggest(self, prefix: str, limit: int = 8) -> list[dict[str, str]]:
        """Return up to ``limit`` suggestions for a typed prefix.

        Every typed word must be the start of some word in the title.
        Titles beginning with the typed text come first; identical titles
        are listed once.

        Args:
            prefix: What the user has typed so far.
            limit: Maximum number of suggestions.

        Returns:
            ``[{"doc_id": ..., "title": ...}, ...]``; empty when ``prefix``
            has fewer than 2 characters.

        Raises:
            RuntimeError: If no index has been built or loaded.
        """
        if self._ix is None:
            raise RuntimeError("Suggestion index is not loaded.")

        typed = " ".join(prefix.split())
        if len(typed) < MIN_PREFIX_CHARS or limit <= 0:
            return []

        words = [
            word[:MAX_NGRAM]
            for word in _NON_WORD.sub(" ", typed.lower()).split()
            if len(word) >= MIN_NGRAM
        ]
        if not words:
            return []

        query = And([Term("ngrams", word) for word in words])
        candidates = max(limit * CANDIDATE_FACTOR, MIN_CANDIDATES)

        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        with self._ix.searcher() as searcher:
            for hit in searcher.search(query, limit=candidates):
                title = hit["title"]
                key = title.casefold()
                if key in seen:
                    continue
                seen.add(key)
                unique.append({"doc_id": hit["doc_id"], "title": title})

        lowered = typed.casefold()
        unique.sort(key=lambda item: not item["title"].casefold().startswith(lowered))
        return unique[:limit]
