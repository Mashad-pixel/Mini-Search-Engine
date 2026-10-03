"""Build the Whoosh suggestion index from ``data/documents.jsonl``.

Usage:  python ingest_suggestions.py
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from search_engine.suggester import SuggestionEngine

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_PATH = BASE_DIR / "data" / "documents.jsonl"
SUGGEST_DIR = BASE_DIR / "data" / "suggest_index"


def read_documents(path: Path) -> tuple[list[str], list[str]]:
    """Read ``doc_id`` and ``text`` from a JSONL file.

    Raises:
        ValueError: If a line is invalid JSON or lacks string fields.
    """
    doc_ids: list[str] = []
    texts: list[str] = []
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
                raise ValueError(f"{path}:{line_number}: expected string 'doc_id' and 'text'")
            doc_ids.append(record["doc_id"])
            texts.append(record["text"])
    return doc_ids, texts


def main() -> None:
    """Build the suggestion index and print statistics."""
    if not DOCUMENTS_PATH.is_file():
        raise SystemExit(f"{DOCUMENTS_PATH} not found. Run `python ingest_real_data.py` first.")

    doc_ids, texts = read_documents(DOCUMENTS_PATH)
    print(f"Documents read:     {len(doc_ids)}")

    started = time.perf_counter()
    engine = SuggestionEngine()
    indexed = engine.build_from_documents(doc_ids, texts)

    shutil.rmtree(SUGGEST_DIR, ignore_errors=True)
    engine.save_to_disk(SUGGEST_DIR)
    elapsed = time.perf_counter() - started

    size_kb = sum(p.stat().st_size for p in SUGGEST_DIR.rglob("*") if p.is_file()) / 1024
    print(f"Documents indexed:  {indexed}")
    print(f"Time:               {elapsed:.2f}s")
    print(f"Index size:         {size_kb:.1f} KB in {SUGGEST_DIR}")


if __name__ == "__main__":
    main()
