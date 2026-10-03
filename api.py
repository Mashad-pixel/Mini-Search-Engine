"""FastAPI backend: hybrid search, autocomplete suggestions and the static web UI."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from preflight import ensure_indexes
from search_engine.hybrid_engine import HybridEngine
from search_engine.suggester import SuggestionEngine

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "data" / "tantivy_index"
VECTOR_DIR = BASE_DIR / "data" / "vector_index"
SUGGEST_DIR = BASE_DIR / "data" / "suggest_index"
WEB_DIR = BASE_DIR / "web"


class SearchHit(BaseModel):
    """A single search result returned by the API.

    ``snippet`` is HTML-escaped text in which matching terms may be wrapped in
    ``<mark>`` tags; ``<mark>`` is the only markup it ever contains.
    """

    doc_id: str
    score: float
    snippet: str


class CategoryCount(BaseModel):
    """A facet value and the number of documents it contains."""

    category: str
    count: int


class Suggestion(BaseModel):
    """An autocomplete suggestion (plain text, not HTML)."""

    doc_id: str
    title: str


class DocumentOut(BaseModel):
    """A full document as stored in the index (plain text, not HTML)."""

    doc_id: str
    text: str
    category: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the search, vector and suggestion indexes once at startup.

    Raises:
        RuntimeError: If any index has not been built yet.
    """
    # Friendly preflight; set MINI_SEARCH_AUTO_INGEST=1 to build missing indexes.
    auto = os.getenv("MINI_SEARCH_AUTO_INGEST", "").lower() in {"1", "true", "yes"}
    ensure_indexes(auto=auto)

    try:
        app.state.engine = HybridEngine.load_from_disk(INDEX_DIR, VECTOR_DIR)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Missing index ({exc}). Run `python ingest_tantivy.py` and "
            f"`python ingest_vectors.py` first."
        ) from exc

    try:
        app.state.suggester = SuggestionEngine.load_from_disk(SUGGEST_DIR)
    except FileNotFoundError as exc:
        app.state.engine.close()
        raise RuntimeError(
            f"Missing suggestion index ({exc}). Run `python ingest_suggestions.py` first."
        ) from exc

    # Skip warmup in dev for faster reloads: set SKIP_WARMUP=1
    if os.getenv("SKIP_WARMUP") != "1":
        app.state.engine.warmup()

    yield
    app.state.suggester.close()
    app.state.engine.close()


app = FastAPI(title="Mini Search API", lifespan=lifespan)


@app.get("/search", response_model=list[SearchHit])
def search(
    request: Request,
    q: str = Query(..., min_length=1, description='Query; use quotes for phrases.'),
    top_k: int = Query(10, ge=1, le=100, description="Maximum results."),
    mode: Literal["lexical", "semantic", "hybrid"] = Query(
        "hybrid",
        description="lexical = BM25 only, semantic = vectors only, hybrid = RRF fusion.",
    ),
    category: list[str] | None = Query(
        None,
        description="Restrict to a category; repeat the parameter to allow several.",
    ),
) -> list[SearchHit]:
    """Run a (optionally category-filtered) search against the indexes.

    Args:
        request: The incoming request (gives access to the engine).
        q: Query string, e.g. ``python "machine learning"``.
        top_k: Maximum number of results to return.
        mode: Retrieval mode: ``lexical``, ``semantic`` or ``hybrid``.
        category: Zero or more category names (OR semantics).

    Returns:
        Ranked hits with ``doc_id``, ``score`` and ``snippet``.
    """
    engine: HybridEngine = request.app.state.engine
    return [
        SearchHit(doc_id=hit.doc_id, score=hit.score, snippet=hit.snippet)
        for hit in engine.search(q, top_k=top_k, mode=mode, category_filter=category)
    ]


@app.get("/suggest", response_model=list[Suggestion])
def suggest(
    request: Request,
    q: str = Query("", max_length=200, description="Prefix typed so far."),
    limit: int = Query(8, ge=1, le=20, description="Maximum suggestions."),
) -> list[Suggestion]:
    """Return autocomplete suggestions for a typed prefix.

    Args:
        request: The incoming request (gives access to the suggester).
        q: The prefix; fewer than 2 characters yields an empty list.
        limit: Maximum number of suggestions.

    Returns:
        ``[{"doc_id": ..., "title": ...}, ...]`` as plain text.
    """
    suggester: SuggestionEngine = request.app.state.suggester
    return [Suggestion(**item) for item in suggester.suggest(q, limit=limit)]


@app.get("/categories", response_model=list[CategoryCount])
def categories(request: Request) -> list[CategoryCount]:
    """List every category with its document count.

    Returns:
        Categories ordered by document count (descending), then name.
    """
    engine: HybridEngine = request.app.state.engine
    return [
        CategoryCount(category=name, count=count)
        for name, count in engine.categories().items()
    ]


@app.get("/document/{doc_id:path}", response_model=DocumentOut)
def document(request: Request, doc_id: str) -> DocumentOut:
    """Return one full document.

    Args:
        request: The incoming request (gives access to the engine).
        doc_id: Identifier of the document (URL-encoded in the path).

    Returns:
        The document's ``doc_id``, ``text`` and ``category``.

    Raises:
        HTTPException: 404 if no document has this id.
    """
    engine: HybridEngine = request.app.state.engine
    found = engine.get_document(doc_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id!r} not found.")
    return DocumentOut(**found)


# Must be registered last so it does not shadow /search, /suggest, /categories
# and /document/{doc_id}.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["."],
        
    )