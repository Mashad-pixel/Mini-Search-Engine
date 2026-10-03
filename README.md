this is a mini search engine it has limited data stored init.

# Mini Search

Mini Search is a hybrid search engine built from small, readable parts: Tantivy BM25
(keywords, phrases, facets, highlighting) and FAISS vectors (all-MiniLM-L6-v2) fused with
Reciprocal Rank Fusion, plus Whoosh autocomplete, a FastAPI backend and a small static web UI.
Matching terms are highlighted with `<mark>` in keyword, semantic and hybrid results (semantic
hits with no literal overlap show a plain preview). It ships with a synthetic corpus generator,
a command-line client, a pytest suite and a Precision/Recall/MRR evaluation script.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![tests](https://github.com/your-username/your-repo/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/your-repo/actions/workflows/tests.yml)

<!--
Screenshots (add the files, then remove this comment wrapper):

![Search page](docs/screenshot-search.png)
![Document viewer](docs/screenshot-document.png)
-->

## Folder layout

```
api.py                 FastAPI app (/search, /suggest, /categories, /document/{id}, serves web/)
preflight.py           Checks / builds the indexes
search_cli.py          Command-line search
ingest_real_data.py    Generates data/documents.jsonl (synthetic corpus)
ingest_tantivy.py      Builds data/tantivy_index + data/categories.json
ingest_vectors.py      Builds data/vector_index   (run after ingest_tantivy.py)
ingest_suggestions.py  Builds data/suggest_index
search_engine/         Library code (index, engine, hybrid_engine, vector_index, suggester, tokenizer)
web/                   index.html (search), document.html (viewer), app.js, style.css
tests/                 pytest suite (fast tests + tests marked "slow")
eval/                  make_queries.py, evaluate.py, queries.jsonl (Precision@k / Recall@k / MRR)
docs/                  Screenshots used by this README
data/                  Generated files (git-ignored; .gitkeep is tracked)
.github/workflows/     CI (tests + lint)
pyproject.toml         Project metadata, ruff, mypy and pytest configuration
pyrightconfig.json     Pylance / Pyright settings
requirements.txt       Runtime dependencies
requirements-dev.txt   Development dependencies (pytest, pytest-cov, ruff, mypy)
LICENSE                MIT
```

## Setup (Python 3.12 recommended)

Prebuilt wheels for `torch`, `faiss-cpu` and `tantivy` are most reliable on 3.11/3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt
```

macOS/Linux: `python3.12 -m venv .venv && source .venv/bin/activate`, then the same pip commands.

In VS Code, run **Python: Select Interpreter** and choose `.venv`; `pyrightconfig.json`
already points Pylance at it.

## Build the indexes (in this order)

```powershell
python ingest_real_data.py
python ingest_tantivy.py
python ingest_vectors.py
python ingest_suggestions.py
```

Shortcut: `python preflight.py --build` builds whatever is missing, including the corpus.

## Run the API

```powershell
python api.py                                   # http://127.0.0.1:8000
$env:MINI_SEARCH_AUTO_INGEST=1; python api.py   # build missing indexes on startup
```

If an index is missing and auto-ingest is off, startup stops with the exact commands to run.
The search page is served at `/` and the document viewer at `/document.html?id=<doc_id>`.

## CLI

```powershell
python search_cli.py "machine learning" --mode hybrid --top-k 5 --category Science
```

Modes: `lexical`, `semantic`, `hybrid`. Add `--auto-ingest` to build missing indexes first.

## Tests and evaluation

```powershell
pytest -q -m "not slow"            # fast tests: no models, no real indexes
pytest -q                          # everything, including tests that build real indexes
python eval/make_queries.py        # regenerate starter queries from the synthetic corpus
python eval/evaluate.py --k 10     # Precision@k, Recall@k and MRR per mode (--verbose: per query)
```

`eval/make_queries.py` labels documents by literal phrase match, which favours keyword
search; hand-label queries in `eval/queries.jsonl` (`{"query": ..., "relevant": [doc_ids]}`)
for a fair comparison.

## License

Released under the [MIT License](LICENSE).
