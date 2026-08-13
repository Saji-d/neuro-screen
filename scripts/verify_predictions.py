#!/usr/bin/env python
"""
scripts/verify_predictions.py
=============================
Parity check: dashboard predictions == direct notebook reproduction.

For a sample of held-out rows it re-runs the EXACT Thesis3.ipynb path
(preprocessing -> split -> CatBoost predict_proba -> ANN forward on the
one-hot + scaled frame -> arithmetic-mean blend) and compares every value
against modules.predictor.predict() — the single path the dashboard uses for
both the Quick check-in and the conversational assistant.

Also re-prints the reproduced held-out metrics next to the paper's reported
numbers.

Usage::

    python scripts/verify_predictions.py
    python scripts/verify_predictions.py --samples 20

Exits non-zero if any probability differs beyond 1e-9.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from modules import predictor  # noqa: E402
from modules.core.preprocessing import (  # noqa: E402
    NUMERIC_FEATURES,
    derive_target,
    detect_categorical_features,
    fix_encoding,
    select_features,
)

SEED = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5
ATOL = 1e-9


def find_dataset(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(explicit)
    for candidate in config.DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Survey CSV not found in data/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dashboard/notebook prediction parity.")
    parser.add_argument("dataset", nargs="?", default=None)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    if not predictor.models_available():
        raise SystemExit("Trained artifacts are missing — run scripts/train_models.py first.")

    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import train_test_split

    # ---- 1. Exact notebook preprocessing + split ----------------------
    df = pd.read_csv(find_dataset(args.dataset))
    df = fix_encoding(df)
    df = derive_target(df)
    X, y = select_features(df)
    cat_features = detect_categorical_features(X)
    feature_cols = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    print(f"Preprocessing reproduced: {len(feature_cols)} features, "
          f"{len(cat_features)} categorical, test set {len(X_test)} rows.")

    # ---- 2. Load the same artifacts the dashboard uses ----------------
    bundle = predictor.load_predictor()
    meta = bundle.meta
    assert meta["feature_cols"] == feature_cols, "Feature order mismatch!"
    assert meta["cat_features"] == cat_features, "Cat feature mismatch!"

    # ---- 3. Per-row parity: dashboard vs direct notebook path ---------
    import torch
    from catboost import Pool
    from modules.core.preprocessing import build_ann_frame, to_tensor_float

    print("\n=== Parity check (dashboard predict vs direct notebook path) ===")
    worst = {"hybrid": 0.0, "cat": 0.0, "ann": 0.0}
    mismatched_labels = 0
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(X_test), size=min(args.samples, len(X_test)), replace=False)

    for i in sample_idx:
        row = X_test.iloc[i]
        raw_inputs = {c: row[c] for c in feature_cols}

        # Dashboard path (single pipeline used by the whole app).
        result = predictor.predict(raw_inputs)

        # Direct notebook path on the same row.
        row_frame = pd.DataFrame([{c: row[c] for c in feature_cols}],
                                 columns=feature_cols).fillna("Missing")
        pool = Pool(row_frame, cat_features=cat_features)
        cat_ref = float(bundle.catboost.predict_proba(pool)[0][1])

        ann_frame = build_ann_frame(row_frame, cat_features, bundle.scaler,
                                    bundle.ann_columns)
        with torch.no_grad():
            ann_ref = float(bundle.ann(to_tensor_float(ann_frame)).item())
        hybrid_ref = (cat_ref + ann_ref) / 2.0

        worst["cat"] = max(worst["cat"], abs(result.cat_proba - cat_ref))
        worst["ann"] = max(worst["ann"], abs(result.ann_proba - ann_ref))
        worst["hybrid"] = max(worst["hybrid"], abs(result.proba_impaired - hybrid_ref))
        if result.prediction != int(hybrid_ref >= THRESHOLD):
            mismatched_labels += 1

    print(f"Rows checked: {len(sample_idx)}")
    print(f"Max |dCatBoost| : {worst['cat']:.3e}")
    print(f"Max |dANN|      : {worst['ann']:.3e}")
    print(f"Max |dHybrid|   : {worst['hybrid']:.3e}")
    print(f"Label mismatches: {mismatched_labels}")

    ok = all(v <= ATOL for v in worst.values()) and mismatched_labels == 0
    print("\nRESULT:", "PARITY OK" if ok else "PARITY FAILED")

    # ---- 4. Reproduced held-out metrics vs paper ----------------------
    from catboost import Pool
    cat_probs = bundle.catboost.predict_proba(Pool(X_test, cat_features=cat_features))[:, 1]
    X_test_ann = build_ann_frame(X_test, cat_features, bundle.scaler, bundle.ann_columns)
    with torch.no_grad():
        ann_probs = bundle.ann(to_tensor_float(X_test_ann)).numpy().flatten()
    hybrid = (cat_probs + ann_probs) / 2.0

    print("\n=== Reproduced held-out metrics (notebook path) ===")
    for name, probs in (("CatBoost", cat_probs), ("ANN", ann_probs), ("Hybrid", hybrid)):
        preds = (probs >= THRESHOLD).astype(int)
        print(f"  {name:9s} acc={accuracy_score(y_test, preds):.4f} "
              f"prec={precision_score(y_test, preds):.4f} "
              f"rec={recall_score(y_test, preds):.4f} "
              f"f1={f1_score(y_test, preds):.4f} "
              f"auc={roc_auc_score(y_test, probs):.4f}")

    paper = config.load_json(config.REPORTED_RESULTS).get("metrics_table", {}).get("rows", {})
    if paper:
        print("\n  Paper (ICCA 2026, Table 2):")
        for name in ("CatBoost", "ANN", "Hybrid"):
            print(f"  {name:9s} acc/prec/rec/f1/auc = {paper.get(name, [])}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
