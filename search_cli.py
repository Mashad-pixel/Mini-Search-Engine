"""Command-line search over the Tantivy + FAISS indexes.

    python search_cli.py "machine learning" --mode hybrid --top-k 5 --category Science
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from preflight import ensure_indexes
from search_engine.hybrid_engine import MODES, HybridEngine

BASE_DIR = Path(__file__).resolve().parent
BOLD, RESET = "\x1b[1;33m", "\x1b[0m"


def render(snippet: str) -> str:
    """Turn the API's escaped ``<mark>`` snippet into ANSI-highlighted text."""
    return html.unescape(snippet.replace("<mark>", BOLD).replace("</mark>", RESET))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--category", action="append", help="Repeat for several.")
    parser.add_argument("--auto-ingest", action="store_true", help="Build missing indexes.")
    args = parser.parse_args()

    try:
        ensure_indexes(auto=args.auto_ingest)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    engine = HybridEngine.load_from_disk(
        BASE_DIR / "data" / "tantivy_index", BASE_DIR / "data" / "vector_index"
    )
    try:
        hits = engine.search(args.query, top_k=args.top_k, mode=args.mode,
                             category_filter=args.category)
        if not hits:
            print("No results.")
        for rank, hit in enumerate(hits, start=1):
            print(f"{rank:>2}. {hit.doc_id}  (score {hit.score:.3f})\n    {render(hit.snippet)}\n")
    finally:
        engine.close()


if __name__ == "__main__":
    main()