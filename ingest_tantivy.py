"""Build the Tantivy index from ``data/documents.jsonl`` with random categories.

Also writes ``data/categories.json`` (doc_id -> category) so that the vector
index built by ``ingest_vectors.py`` uses exactly the same categories.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

from search_engine import SearchEngine

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
INDEX_DIR = DATA_DIR / "tantivy_index"
CATEGORIES_PATH = DATA_DIR / "categories.json"

CATEGORIES: tuple[str, ...] = ("Science", "Technology", "History", "Geography", "Arts")
DEFAULT_SEED = 42


def read_jsonl(path: Path) -> Iterator[dict[str, str]]:
    """Read and validate documents from a JSONL file.

    Args:
        path: Source file with one ``{"doc_id": ..., "text": ...}`` per line.

    Yields:
        Mappings with string ``doc_id`` and ``text`` keys.

    Raises:
        ValueError: If a line is invalid JSON or lacks the string fields.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("doc_id"), str)
                or not isinstance(record.get("text"), str)
            ):
                raise ValueError(
                    f"{path}:{line_number}: expected string 'doc_id' and 'text'"
                )
            yield {"doc_id": record["doc_id"], "text": record["text"]}


def directory_size_bytes(directory: Path) -> int:
    """Sum the sizes of all files under a directory.

    Args:
        directory: Directory to measure.

    Returns:
        Total size in bytes.
    """
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Seed for category assignment."
    )
    return parser.parse_args(argv)


def main() -> None:
    """Rebuild the Tantivy index from scratch and print indexing statistics."""
    args = parse_args()
    if not DOCUMENTS_PATH.is_file():
        raise SystemExit(
            f"{DOCUMENTS_PATH} not found. Generate it first with Day 3's "
            "ingest_real_data.py."
        )

    rng = random.Random(args.seed)
    shutil.rmtree(INDEX_DIR, ignore_errors=True)

    started = time.perf_counter()
    engine = SearchEngine(INDEX_DIR)
    submitted = 0
    assignments: dict[str, str] = {}
    for record in read_jsonl(DOCUMENTS_PATH):
        category = rng.choice(CATEGORIES)
        engine.add_document(record["doc_id"], record["text"], category)
        assignments[record["doc_id"]] = category
        submitted += 1
    engine.close()  # commit, wait for merges, release the writer lock
    elapsed = time.perf_counter() - started
    CATEGORIES_PATH.write_text(json.dumps(assignments, ensure_ascii=False), encoding="utf-8")

    size_mb = directory_size_bytes(INDEX_DIR) / (1024 * 1024)
    print(f"Documents read:     {submitted}")
    print(f"Documents indexed:  {engine.num_docs}")
    print(f"Segments:           {engine.index.num_segments}")
    print(f"Time:               {elapsed:.2f}s ({submitted / max(elapsed, 1e-9):.0f} docs/s)")
    print(f"Index size:         {size_mb:.2f} MB in {INDEX_DIR}")
    print("Documents per category:")
    for category, count in engine.categories().items():
        print(f"  {category:<12} {count}")


if __name__ == "__main__":
    main()