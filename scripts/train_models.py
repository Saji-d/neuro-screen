#!/usr/bin/env python
"""
scripts/train_models.py
=======================
Reproduces the Thesis3.ipynb training pipeline end-to-end and exports the
artifacts the Neuro-Screen dashboard needs for real inference.

Pipeline (identical to the notebook, final paper configuration):

  1. Load the survey CSV, fix string encoding artifacts.
  2. Aggregate the 7 cognitive symptoms into ``cognitive_score``; binarize
     at the 72nd percentile -> ``cognitive_impairment`` target.
  3. Derive the ``insomniac`` flag; build the 21-feature frame with missing
     values filled by the string ``"Missing"``.
  4. 80/20 stratified split (random_state=42).
  5. Train CatBoost (800 iterations, lr 0.002, depth 5, early stopping 50).
  6. Train the 128-64-1 ANN on the one-hot encoded + age-scaled frame
     (Adam lr 0.008, BCE, 40 epochs, batch 16, dropout 0.6/0.4).
  7. Hybrid = arithmetic mean of the two models' probabilities, threshold 0.5.
  8. Evaluate all three models and export the artifacts.

Usage::

    python scripts/train_models.py
    python scripts/train_models.py path/to/dataset.csv

Outputs (see config.ARTIFACTS)::

    models/catboost_model.cbm
    models/ann_model.pt
    models/scaler.pkl
    models/label_encoder.pkl
    models/feature_columns.pkl
    models/preprocessing_pipeline.pkl
    models/metadata.json
    models/eval_metrics.json
    models/roc_data.json
    data/feature_schema.generated.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from modules.core.preprocessing import (  # noqa: E402
    INSOMNIAC_SLEEP_HOURS,
    INSOMNIAC_SLEEP_QUALITY,
    NUMERIC_FEATURES,
    derive_target,
    detect_categorical_features,
    fix_encoding,
    select_features,
)
from modules.core.model_defs import CognitiveImpairmentANN  # noqa: E402

SEED = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5
EPOCHS = 40
BATCH_SIZE = 16
LR = 0.008

CATBOOST_CONFIG = dict(
    iterations=800,
    learning_rate=0.002,
    depth=5,
    random_seed=SEED,
    early_stopping_rounds=50,
    eval_metric="Logloss",
    verbose=100,
)


# ------------------------------------------------------------------
# Dataset discovery
# ------------------------------------------------------------------
def find_dataset(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"Dataset not found at: {explicit}")
    for candidate in config.DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the survey dataset. Place it in the data/ folder as "
        "`Detection of cognitive impairment in insomniac university students "
        "dataset.csv` (or pass an explicit path)."
    )


# ------------------------------------------------------------------
# Schema validation / generation against the real CSV columns
# ------------------------------------------------------------------
def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def validate_schema(csv_columns: list[str]) -> None:
    """Cross-check the shipped feature schema against the actual CSV columns
    and write data/feature_schema.generated.json with resolved mappings."""
    schema = config.load_json(config.FEATURE_SCHEMA)
    features = schema.get("features", [])
    if not features:
        print("[schema] No shipped feature schema found; skipping.")
        return

    columns_set = set(csv_columns)
    unresolved = []
    generated = {"version": 2, "source": "generated", "features": []}

    for feat in features:
        fc = feat.get("model_column")
        entry = dict(feat)
        if fc and fc in columns_set:
            entry["model_column"] = fc
        else:
            best, best_score = None, 0.0
            for col in csv_columns:
                score = _similarity(fc or feat.get("label", ""), col)
                if score > best_score:
                    best, best_score = col, score
            if best is not None and best_score >= 0.35:
                entry["model_column"] = best
                print(f"[schema] '{feat.get('key')}': '{fc}' -> '{best}' (auto)")
            else:
                entry["model_column"] = None
                entry["_unresolved"] = True
                unresolved.append(
                    (feat.get("key"), fc or feat.get("label"), best, round(best_score, 2))
                )
        generated["features"].append(entry)

    config.save_json(config.FEATURE_SCHEMA_GENERATED, generated)
    print(f"[schema] Generated data/feature_schema.generated.json "
          f"({len(features)} features).")
    if unresolved:
        print("\n[!] The following schema features could not be matched to a "
              "CSV column automatically. Edit their 'model_column' in "
              "data/feature_schema.json and re-run, or add them to the "
              "generated file by hand.")
        for key, expected, closest, score in unresolved:
            print(f"    - {key}: expected '{expected}' | closest: "
                  f"'{closest}' (sim {score})")
        print("\n    Available CSV columns were:")
        for col in csv_columns:
            print(f"      - {col}")


# ------------------------------------------------------------------
# Evaluation helpers (mirror the notebook's printed output)
# ------------------------------------------------------------------
def evaluate(name: str, y_true: np.ndarray, probs: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    preds = (probs >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds)), 4),
        "recall": round(float(recall_score(y_true, preds)), 4),
        "f1": round(float(f1_score(y_true, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print(f"\n[{name}] results:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  CM       : TN={metrics['confusion_matrix']['tn']} "
          f"FP={metrics['confusion_matrix']['fp']} "
          f"FN={metrics['confusion_matrix']['fn']} "
          f"TP={metrics['confusion_matrix']['tp']}")
    return metrics


def _downsample_curve(fpr, tpr, max_points: int = 200):
    step = max(1, len(fpr) // max_points)
    return fpr[::step].tolist(), tpr[::step].tolist()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Neuro-Screen hybrid model.")
    parser.add_argument("dataset", nargs="?", default=None,
                        help="Path to the survey CSV (optional; auto-detected in data/).")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="ANN epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="ANN batch size.")
    parser.add_argument("--lr", type=float, default=LR, help="ANN learning rate.")
    args = parser.parse_args()

    # -- heavy imports (fail with a friendly message) --------------------
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from catboost import CatBoostClassifier
        from sklearn.metrics import (accuracy_score, confusion_matrix,
                                     f1_score, precision_score, recall_score,
                                     roc_auc_score, roc_curve)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for training. Install catboost + torch first:\n"
            "    pip install catboost torch\n"
            f"(underlying error: {exc})"
        )

    print("=== Cognitive Impairment Detection - CatBoost + ANN + Hybrid ===\n")
    dataset_path = find_dataset(args.dataset)

    # ---- 1. Load + clean -----------------------------------------------
    print("=== Step 1: Preprocessing ===")
    df = pd.read_csv(dataset_path)
    df = fix_encoding(df)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ---- 2. Target engineering -----------------------------------------
    df = derive_target(df)
    insomniac_rate = float(df["insomniac"].mean())
    impaired_rate = float(df["cognitive_impairment"].mean())
    print(f"Insomniac students: {int(df['insomniac'].sum())} ({insomniac_rate*100:.1f}%)")
    print(f"Cognitive impairment positive rate: {impaired_rate*100:.1f}%\n")

    # ---- 3. Feature selection ------------------------------------------
    X, y = select_features(df)
    cat_features = detect_categorical_features(X)
    feature_cols = list(X.columns)
    print(f"Using {len(feature_cols)} features for modeling")

    validate_schema(feature_cols)

    # ---- 4. Split -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # ---- 5. CatBoost ----------------------------------------------------
    print("=== Training CatBoost ===")
    cat_model = CatBoostClassifier(**CATBOOST_CONFIG)
    cat_model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )
    cat_probs = cat_model.predict_proba(X_test)[:, 1]
    cat_metrics = evaluate("CatBoost", y_test, cat_probs)

    # ---- 6. ANN ---------------------------------------------------------
    print("\n=== Training ANN ===")
    X_train_ann = pd.get_dummies(X_train, columns=cat_features)
    X_test_ann = pd.get_dummies(X_test, columns=cat_features)
    X_train_ann, X_test_ann = X_train_ann.align(
        X_test_ann, join="left", axis=1, fill_value=0
    )

    num_cols = [c for c in NUMERIC_FEATURES if c in X_train_ann.columns]
    scaler = StandardScaler()
    X_train_ann[num_cols] = scaler.fit_transform(X_train_ann[num_cols].astype(np.float64))
    X_test_ann[num_cols] = scaler.transform(X_test_ann[num_cols].astype(np.float64))

    train_ds = TensorDataset(
        torch.tensor(X_train_ann.values.astype(np.float32)),
        torch.tensor(y_train.values.astype(np.float32)).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    ann_model = CognitiveImpairmentANN(X_train_ann.shape[1])
    optimizer = optim.Adam(ann_model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # The notebook does not seed torch, so its ANN is non-deterministic run to
    # run. We add a seed purely for reproducible exports; the architecture,
    # loss, optimizer and training loop are unchanged.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ann_model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(ann_model(bx), by)
            loss.backward()
            optimizer.step()
            running += loss.item() * bx.size(0)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"ANN Epoch {epoch:02d}/{args.epochs} -> Loss: {running/len(train_ds):.4f}")

    ann_model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_ann.values.astype(np.float32))
        ann_probs = ann_model(test_tensor).numpy().flatten()
    ann_metrics = evaluate("ANN", y_test, ann_probs)

    # ---- 7. Hybrid ------------------------------------------------------
    print("\n=== Hybrid Model (CatBoost + ANN) ===")
    hybrid_probs = (cat_probs + ann_probs) / 2.0
    hybrid_metrics = evaluate("Hybrid", y_test, hybrid_probs)

    # ---- 8. Export artifacts -------------------------------------------
    print("\n=== Exporting artifacts ===")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cat_model.save_model(str(config.ARTIFACTS["catboost"]))
    torch.save(
        {"state_dict": ann_model.state_dict(), "input_dim": X_train_ann.shape[1]},
        str(config.ARTIFACTS["ann"]),
    )
    with open(config.ARTIFACTS["scaler"], "wb") as fh:
        pickle.dump(scaler, fh)
    with open(config.ARTIFACTS["feature_columns"], "wb") as fh:
        pickle.dump(list(X_train_ann.columns), fh)

    # Category vocabulary learned by the model (per-feature -> options). This
    # is the real "label encoder" for this pipeline (the notebook uses
    # pd.get_dummies + CatBoost's native categoricals, not sklearn's
    # LabelEncoder), and lets the UI/chat match answers to the exact strings
    # the model knows.
    category_vocab = {col: sorted(X[col].dropna().unique().tolist()) for col in cat_features}
    with open(config.ARTIFACTS["label_encoder"], "wb") as fh:
        pickle.dump(category_vocab, fh)

    preprocessing_pipeline = {
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_cols": num_cols,
        "ann_columns": list(X_train_ann.columns),
        "threshold": THRESHOLD,
        "vocab": category_vocab,
    }
    with open(config.ARTIFACTS["preprocessing_pipeline"], "wb") as fh:
        pickle.dump(preprocessing_pipeline, fh)

    metrics_all = {
        "CatBoost": cat_metrics,
        "ANN": ann_metrics,
        "Hybrid": hybrid_metrics,
    }

    meta = {
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_cols": num_cols,
        "threshold": THRESHOLD,
        "labels": {"0": "Healthy", "1": "Impaired"},
        "ann_input_dim": X_train_ann.shape[1],
        "config": {
            "catboost": CATBOOST_CONFIG,
            "ann": {"epochs": args.epochs, "batch_size": args.batch_size,
                    "lr": args.lr, "dropout": [0.6, 0.4]},
            "split": {"test_size": TEST_SIZE, "random_state": SEED},
        },
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "features": len(feature_cols),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "insomniac_rate": round(insomniac_rate, 4),
            "impaired_rate": round(impaired_rate, 4),
        },
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    config.save_json(config.ARTIFACTS["meta"], meta)
    config.save_json(config.ARTIFACTS["eval"], metrics_all)

    roc = {}
    for name, probs in (("CatBoost", cat_probs), ("ANN", ann_probs), ("Hybrid", hybrid_probs)):
        fpr, tpr, _ = roc_curve(y_test, probs)
        fpr_s, tpr_s = _downsample_curve(fpr, tpr)
        roc[name] = {
            "fpr": fpr_s,
            "tpr": tpr_s,
            "auc": float(roc_auc_score(y_test, probs)),
        }
    config.save_json(config.ARTIFACTS["roc"], roc)

    # Invalidate any in-memory cache in the running app.
    from modules import predictor
    predictor.reset_cache()

    print("\nSaved:")
    for name in ("catboost", "ann", "scaler", "label_encoder", "feature_columns",
                 "preprocessing_pipeline", "meta", "eval", "roc"):
        print(f"  {config.ARTIFACTS[name]}")
    print("\nDone. Restart or refresh the dashboard to start using the real model.")

    # Quick comparison to the paper's reported numbers.
    reported = config.load_json(config.REPORTED_RESULTS).get("metrics_table", {}).get("rows", {})
    if reported:
        print("\n[comparison] Reproduced vs. reported (paper Table 2):")
        for model in ("CatBoost", "ANN", "Hybrid"):
            rep = reported.get(model, [])
            live = [metrics_all[model][m] for m in
                    ("accuracy", "precision", "recall", "f1", "roc_auc")]
            print(f"  {model:9s} live={live}  paper={rep}")


if __name__ == "__main__":
    main()
