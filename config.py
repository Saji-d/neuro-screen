"""
config.py
=========
Central application configuration for Neuro-Screen (paths + brand).

Holds filesystem paths, brand identity (as published in the thesis report),
risk thresholds, and the evaluation settings. Every other module imports constants from here so that values that
must stay consistent (thresholds, column names, model paths) live in one
place only.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

# ------------------------------------------------------------------
# Filesystem layout
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Trained artifacts (scripts/train_models.py): one folder per task plus a
# shared evaluation file.
TASK_DIRS = {
    "sleep_risk": MODELS_DIR / "sleep_risk",
    "cognitive": MODELS_DIR / "cognitive",
}
EVAL_METRICS = MODELS_DIR / "eval_metrics.json"

# Candidate locations for the raw survey CSV (the thesis dataset).
DATASET_CANDIDATES = [
    DATA_DIR / "Thesis Dataset.csv",
    DATA_DIR / "Detection of cognitive impairment in insomniac university students dataset.csv",
    DATA_DIR / "dataset.csv",
]

# ------------------------------------------------------------------
# Brand identity
# ------------------------------------------------------------------
BRAND = {
    "name": "Neuro-Screen",
    "subtitle": ("Hybrid Ensemble Framework for Early Detection of Cognitive "
                 "Impairment in Insomniac University Students"),
    "author": "Sajidur Rahman Sajid",
    "institution": "American International University-Bangladesh (AIUB)",
}

# ------------------------------------------------------------------
# Small JSON helpers
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_json_cached(path_str: str, mtime: float):
    """Cached JSON read, keyed on path + mtime so a re-trained/edited file
    (different mtime) is picked up without needing an app restart."""
    try:
        with open(path_str, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def load_json(path: Path):
    """Load a JSON file as a dict (returns {} on any failure).

    Cached by (path, mtime) — these files are static config/reported-results/
    trained-artifact metadata, never per-user data, so caching is safe and
    avoids re-parsing on every Streamlit rerun.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    return _load_json_cached(str(path), mtime)


def save_json(path: Path, payload) -> None:
    """Serialize a dict/list to JSON (ensuring the parent dir exists)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
