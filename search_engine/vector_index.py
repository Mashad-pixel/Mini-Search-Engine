"""Dense retrieval: MiniLM sentence embeddings in a FAISS inner-product index.

Vectors are L2-normalised, so inner product equals cosine similarity.
"""

from __future__ import annotations

import html
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
INDEX_FILE = "index.faiss"
META_FILE = "meta.json"
SNIPPET_CHARS = 240

_WHITESPACE = re.compile(r"\s+")


def _make_snippet(text: str) -> str:
    """Return a short, HTML-escaped plain-text preview of ``text``."""
    flat = _WHITESPACE.sub(" ", text).strip()
    if len(flat) > SNIPPET_CHARS:
        cut = flat[:SNIPPET_CHARS].rsplit(" ", 1)[0] or flat[:SNIPPET_CHARS]
        flat = cut + "\u2026"
    return html.escape(flat, quote=False)


def _normalise_categories(value: object) -> list[str]:
    """Coerce a category value (None, str or iterable) to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        return [str(item) for item in value if item is not None and str(item)]  # type: ignore[union-attr]
    except TypeError:
        return [str(value)]


class VectorIndex:
    """FAISS-backed semantic index over document texts."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.doc_ids: list[str] = []
        self.categories: list[list[str]] = []
        self.snippets: list[str] = []
        self._position: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.doc_ids)

    @property
    def model(self) -> SentenceTransformer:
        """The embedding model, loaded lazily on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def warmup(self) -> None:
        """Load the model and run one encode so the first query is fast."""
        self.encode(["warmup"])

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into an ``(n, 384)`` float32 matrix of unit vectors."""
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EMBEDDING_DIM}-dimensional embeddings, got shape {vectors.shape}."
            )
        return vectors

    def add_documents(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        categories: Sequence[object] | None = None,
        show_progress: bool = False,
    ) -> None:
        """Embed ``texts`` and add them to the index under ``doc_ids``.

        Args:
            doc_ids: Unique document identifiers.
            texts: Document texts, aligned with ``doc_ids``.
            categories: Optional per-document category (str or list of str),
                used for category filtering at query time.
            show_progress: Show an encoding progress bar.

        Raises:
            ValueError: On length mismatch or duplicate ids.
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length.")
        if categories is not None and len(categories) != len(doc_ids):
            raise ValueError("categories must have the same length as doc_ids.")
        if not doc_ids:
            return

        seen: set[str] = set()
        for doc_id in doc_ids:
            if doc_id in self._position or doc_id in seen:
                raise ValueError(f"Duplicate doc_id: {doc_id!r}")
            seen.add(doc_id)

        vectors = self.encode(texts, show_progress=show_progress)
        self.index.add(vectors)

        for offset, (doc_id, text) in enumerate(zip(doc_ids, texts)):
            self._position[doc_id] = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            self.snippets.append(_make_snippet(text))
            self.categories.append(
                _normalise_categories(categories[offset]) if categories is not None else []
            )

    def snippet(self, doc_id: str) -> str:
        """Return the stored HTML-escaped preview for ``doc_id`` (or '')."""
        position = self._position.get(doc_id)
        return self.snippets[position] if position is not None else ""

    def search(
        self,
        query: str,
        top_k: int = 10,
        categories: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(doc_id, cosine_score)`` pairs, best first.

        Args:
            query: Natural-language query.
            top_k: Maximum number of results.
            categories: If given, only documents in any of these categories.
        """
        total = self.index.ntotal
        if total == 0 or top_k <= 0 or not query.strip():
            return []

        allowed = set(categories) if categories else None
        query_vector = self.encode([query])
        fetch = min(total, top_k) if allowed is None else min(total, max(top_k * 10, 100))

        while True:
            scores, positions = self.index.search(query_vector, fetch)
            results: list[tuple[str, float]] = []
            for score, position in zip(scores[0], positions[0]):
                if position < 0:
                    continue
                if allowed is not None and not allowed.intersection(self.categories[position]):
                    continue
                results.append((self.doc_ids[position], float(score)))
                if len(results) == top_k:
                    break
            if len(results) == top_k or fetch >= total:
                return results
            fetch = min(total, fetch * 4)

    def save_to_disk(self, path: str | Path) -> None:
        """Write the FAISS index and doc-id mapping into directory ``path``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        meta = {
            "model": self.model_name,
            "dim": EMBEDDING_DIM,
            "doc_ids": self.doc_ids,
            "categories": self.categories,
            "snippets": self.snippets,
        }
        with open(directory / META_FILE, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False)

    @classmethod
    def load_from_disk(cls, path: str | Path) -> "VectorIndex":
        """Load an index previously written by :meth:`save_to_disk`.

        Raises:
            FileNotFoundError: If the index files are missing.
            ValueError: If the files are inconsistent.
        """
        directory = Path(path)
        index_path = directory / INDEX_FILE
        meta_path = directory / META_FILE
        if not index_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"No vector index in {directory}")

        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)

        instance = cls(model_name=meta.get("model", MODEL_NAME))
        instance.index = faiss.read_index(str(index_path))
        instance.doc_ids = list(meta["doc_ids"])
        instance.categories = [list(c) for c in meta["categories"]]
        instance.snippets = list(meta["snippets"])
        instance._position = {doc_id: i for i, doc_id in enumerate(instance.doc_ids)}

        if not (
            instance.index.ntotal
            == len(instance.doc_ids)
            == len(instance.categories)
            == len(instance.snippets)
        ):
            raise ValueError(f"Vector index in {directory} is inconsistent; re-run ingest_vectors.py.")
        return instance