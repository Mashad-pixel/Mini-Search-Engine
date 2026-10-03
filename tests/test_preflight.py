"""Tests for :mod:`preflight` against a throwaway ``data/`` directory.

``BASE_DIR``, ``DATA_DIR``, ``DOCUMENTS_PATH`` and ``INDEXES`` are patched onto
``tmp_path`` so the real ``data/`` folder is never read or written, and
``subprocess.run`` is replaced so no ingest script is ever executed.

Assumption: the marker for the Whoosh index is any ``*.toc`` file, so the
tests create ``_MAIN_1.toc`` (the name Whoosh uses for its first revision).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import preflight

SCRIPTS = ["ingest_tantivy.py", "ingest_vectors.py", "ingest_suggestions.py"]
MARKER_FILES = {
    "ingest_tantivy.py": "meta.json",
    "ingest_vectors.py": "index.faiss",
    "ingest_suggestions.py": "_MAIN_1.toc",
}


def build_index(script: str) -> None:
    """Create the marker file that makes ``script``'s index count as built."""
    for _label, directory, _marker, name in preflight.INDEXES:
        if name == script:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / MARKER_FILES[script]).write_text("x", encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point every ``preflight`` path at an empty temporary ``data/`` folder."""
    root = tmp_path / "project"
    data = root / "data"
    data.mkdir(parents=True)
    indexes = tuple(
        (label, data / directory.name, marker, script)
        for label, directory, marker, script in preflight.INDEXES
    )
    monkeypatch.setattr(preflight, "BASE_DIR", root)
    monkeypatch.setattr(preflight, "DATA_DIR", data)
    monkeypatch.setattr(preflight, "DOCUMENTS_PATH", data / "documents.jsonl")
    monkeypatch.setattr(preflight, "INDEXES", indexes)
    return data


def test_paths_are_sandboxed(data_dir, tmp_path):
    assert tmp_path in preflight.DOCUMENTS_PATH.parents
    assert all(tmp_path in directory.parents for _l, directory, _m, _s in preflight.INDEXES)


def test_all_indexes_missing_when_data_is_empty(data_dir):
    missing = preflight.missing_indexes()
    assert [script for _label, script in missing] == SCRIPTS
    assert [label for label, _script in missing] == [
        "Tantivy keyword index",
        "FAISS vector index",
        "Whoosh suggestion index",
    ]


def test_empty_index_directories_still_count_as_missing(data_dir):
    for _label, directory, _marker, _script in preflight.INDEXES:
        directory.mkdir()
    assert len(preflight.missing_indexes()) == 3


def test_no_indexes_missing_when_all_markers_exist(data_dir):
    for script in SCRIPTS:
        build_index(script)
    assert preflight.missing_indexes() == []


def test_only_unbuilt_indexes_are_reported(data_dir):
    build_index("ingest_tantivy.py")
    assert [script for _label, script in preflight.missing_indexes()] == SCRIPTS[1:]


def test_ensure_indexes_is_a_noop_when_everything_exists(data_dir, monkeypatch):
    for script in SCRIPTS:
        build_index(script)

    def fail(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called")

    monkeypatch.setattr(subprocess, "run", fail)
    assert preflight.ensure_indexes(auto=False) is None
    assert preflight.ensure_indexes(auto=True) is None


def test_ensure_indexes_without_corpus_points_at_ingest_real_data(data_dir):
    with pytest.raises(RuntimeError, match=r"ingest_real_data\.py"):
        preflight.ensure_indexes(auto=False)


def test_ensure_indexes_lists_missing_scripts_in_build_order(data_dir):
    preflight.DOCUMENTS_PATH.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        preflight.ensure_indexes(auto=False)
    message = str(excinfo.value)
    positions = [message.index(f"python {script}") for script in SCRIPTS]
    assert positions == sorted(positions)


def test_auto_build_runs_scripts_in_order(data_dir, monkeypatch):
    calls: list[str] = []

    def fake_run(command, cwd=None, check=False, **kwargs):
        script = Path(command[1]).name
        calls.append(script)
        if script == "ingest_real_data.py":
            preflight.DOCUMENTS_PATH.write_text("{}\n", encoding="utf-8")
        else:
            build_index(script)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    preflight.ensure_indexes(auto=True)
    assert calls == ["ingest_real_data.py", *SCRIPTS]
    assert preflight.missing_indexes() == []


def test_auto_build_reports_indexes_that_stay_missing(data_dir, monkeypatch):
    preflight.DOCUMENTS_PATH.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0)
    )
    with pytest.raises(RuntimeError, match="still missing"):
        preflight.ensure_indexes(auto=True)