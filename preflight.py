"""Check that the search indexes exist; optionally build them.

    python preflight.py          # report status
    python preflight.py --build  # build whatever is missing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"

# (label, directory, marker glob, ingest script) in the order they must be built.
# Tantivy first: it writes categories.json, which the vector ingest reuses.
INDEXES: tuple[tuple[str, Path, str, str], ...] = (
    ("Tantivy keyword index", DATA_DIR / "tantivy_index", "meta.json", "ingest_tantivy.py"),
    ("FAISS vector index", DATA_DIR / "vector_index", "index.faiss", "ingest_vectors.py"),
    ("Whoosh suggestion index", DATA_DIR / "suggest_index", "*.toc", "ingest_suggestions.py"),
)


def missing_indexes() -> list[tuple[str, str]]:
    """Return ``(label, script)`` for every index that is not built yet."""
    return [
        (label, script)
        for label, directory, marker, script in INDEXES
        if not (directory.is_dir() and any(directory.glob(marker)))
    ]


def ensure_indexes(auto: bool = False) -> None:
    """Make sure all indexes exist.

    Args:
        auto: Run the ingest scripts for missing indexes instead of failing.

    Raises:
        RuntimeError: With step-by-step instructions if something is missing
            and cannot (or may not) be built.
        subprocess.CalledProcessError: If an ingest script fails.
    """
    missing = missing_indexes()
    if not missing:
        return

    if not DOCUMENTS_PATH.is_file():
        if not auto:
            raise RuntimeError(
                f"{DOCUMENTS_PATH} not found.\n"
                "Run `python ingest_real_data.py` first, then the ingest scripts "
                "(or `python preflight.py --build`)."
            )
        print("[preflight] Generating corpus (ingest_real_data.py) ...", flush=True)
        subprocess.run([sys.executable, str(BASE_DIR / "ingest_real_data.py")],
                       cwd=BASE_DIR, check=True)

    if not auto:
        steps = "\n".join(f"  python {script}    # {label}" for label, script in missing)
        raise RuntimeError(
            "Search indexes are not built yet. Run, in this order:\n"
            f"{steps}\n"
            "Or build them automatically: `python preflight.py --build`, or set "
            "MINI_SEARCH_AUTO_INGEST=1 before starting the API."
        )

    for label, script in missing:
        print(f"[preflight] Building {label} ({script}) ...", flush=True)
        subprocess.run([sys.executable, str(BASE_DIR / script)], cwd=BASE_DIR, check=True)

    still_missing = missing_indexes()
    if still_missing:
        names = ", ".join(label for label, _ in still_missing)
        raise RuntimeError(f"Ingestion finished but these are still missing: {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build missing indexes.")
    args = parser.parse_args()
    try:
        ensure_indexes(auto=args.build)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print("All indexes present.")


if __name__ == "__main__":
    main()