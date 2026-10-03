"""Build the FAISS vector index from ``data/documents.jsonl``.

Run AFTER ``ingest_tantivy.py``. Categories come from ``data/categories.json``
(written by ``ingest_tantivy.py``) so semantic and keyword search agree.

Usage:  python ingest_vectors.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

from ingest_tantivy import CATEGORIES, CATEGORIES_PATH, DEFAULT_SEED, DOCUMENTS_PATH, read_jsonl
from search_engine.vector_index import VectorIndex

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "data" / "vector_index"


def load_categories(doc_ids: Sequence[str], seed: int) -> list[str]:
    """Return one category per document, matching the Tantivy index.

    Uses ``categories.json`` when present; otherwise replays the seeded random
    assignment performed by ``ingest_tantivy.py``.
    """
    if CATEGORIES_PATH.is_file():
        mapping = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
        missing = [d for d in doc_ids if d not in mapping]
        if missing:
            raise SystemExit(
                f"{len(missing)} documents are missing from {CATEGORIES_PATH}; "
                "re-run `python ingest_tantivy.py` first."
            )
        return [mapping[d] for d in doc_ids]

    print(f"  {CATEGORIES_PATH.name} not found; replaying seed {seed} (same as ingest_tantivy.py).")
    rng = random.Random(seed)
    return [rng.choice(CATEGORIES) for _ in read_jsonl(DOCUMENTS_PATH)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Only used if categories.json is missing.")
    args = parser.parse_args(argv)

    if not DOCUMENTS_PATH.is_file():
        raise SystemExit(f"{DOCUMENTS_PATH} not found. Run `python ingest_real_data.py` first.")

    print(f"Reading documents from {DOCUMENTS_PATH} ...")
    records = list(read_jsonl(DOCUMENTS_PATH))
    if not records:
        raise SystemExit("No documents to index.")
    categories = load_categories([r["doc_id"] for r in records], args.seed)

    # Tantivy replaces a re-added doc_id, so the last occurrence wins; do the same here.
    latest: dict[str, tuple[str, str]] = {}
    for record, category in zip(records, categories):
        latest[record["doc_id"]] = (record["text"], category)
    doc_ids = list(latest)
    texts = [latest[d][0] for d in doc_ids]
    cats = [latest[d][1] for d in doc_ids]
    print(f"  Loaded {len(doc_ids)} unique documents.")

    print("Loading sentence-transformers model (all-MiniLM-L6-v2) and encoding ...")
    started = time.perf_counter()
    index = VectorIndex()
    index.add_documents(doc_ids, texts, categories=cats, show_progress=True)
    elapsed = time.perf_counter() - started

    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR, ignore_errors=True)
    index.save_to_disk(VECTOR_DIR)
    print(f"\nDone! Vector index saved to {VECTOR_DIR}")
    print(f"  Documents indexed : {len(index)}")
    print(f"  Vector dimension  : 384")
    print(f"  Time              : {elapsed:.1f}s")

if __name__ == "__main__":
    main()