#!/usr/bin/env python
"""
scripts/train_models.py
=======================
Train the Neuro-Screen four-model stacks and write every artifact the app uses.

    python scripts/train_models.py            # final fit + 5-fold CV metrics
    python scripts/train_models.py --skip-cv  # final fit only (fast)

Follows ``thesis-updated-code.ipynb``. Two tasks are trained on the 12
self-reported predictors:

* ``sleep_risk`` - all respondents, label = meets the sleep-risk rule
* ``cognitive``  - respondents meeting the rule, label = symptom score >= the
                   72nd percentile of the training split

Outputs (``models/``): ``<task>/`` artifacts and ``eval_metrics.json`` with
5-fold cross-validated metrics (mean/SD per model), out-of-fold ROC points,
CatBoost feature importance and the cohort score distribution.

Deviation from the notebook, stated plainly: the Top-6 / CatBoost-ANN stacks,
the tuned threshold and the logistic/ridge comparators are not re-run here;
inner folds reuse the outer-training label cutoff (identical in practice
because the discrete score gives the same 72nd percentile, 10, in every fold).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config
from modules.ml import pipeline
from modules.ml.pipeline import BASE
from modules.ml.survey import (CATEGORICAL, FEATURES, SCORE_MAX, SCORE_QUANTILE,
                               load_survey, option_vocabulary)

SEED = 42
THREADS = 4
TASKS = {"sleep_risk": "Sleep risk", "cognitive": "Cognitive symptoms"}


def _labels(task: str, sleep, score, train_idx):
    if task == "sleep_risk":
        return sleep.to_numpy(), None
    cutoff = float(score.iloc[train_idx].quantile(SCORE_QUANTILE))
    return (score >= cutoff).astype(int).to_numpy(), cutoff


def _metrics(y, p, threshold=0.5) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    hard = p >= threshold
    neg = (y == 0)
    return dict(
        accuracy=accuracy_score(y, hard), balanced_accuracy=balanced_accuracy_score(y, hard),
        precision=precision_score(y, hard, zero_division=0), recall=recall_score(y, hard),
        specificity=float((~hard[neg]).mean()) if neg.any() else float("nan"),
        f1=f1_score(y, hard, zero_division=0), auc=roc_auc_score(y, p),
    )


def cross_validate(task, X, score, sleep, ids):
    from sklearn.model_selection import KFold, StratifiedKFold

    stratify = task == "sleep_risk"
    splitter = (StratifiedKFold(5, shuffle=True, random_state=SEED).split(ids, sleep.iloc[ids])
                if stratify else KFold(5, shuffle=True, random_state=SEED).split(ids))
    rows, oof_y, oof_p = [], [], []
    for fold, (i, j) in enumerate(splitter, 1):
        train, val = ids[i], ids[j]
        labels, _ = _labels(task, sleep, score, train)
        comp, head = pipeline.fit_stack(X.iloc[train], labels[train], SEED + fold, stratify, THREADS)
        p = pipeline.stack_scores(comp, head, X.iloc[val])
        truth = labels[val]
        for model in ["Majority", *BASE, "Stack"]:
            rows.append(dict(model=model, fold=fold, **_metrics(truth, p[model].to_numpy())))
        oof_y.append(truth)
        oof_p.append(p["Stack"].to_numpy())
        print(f"  {TASKS[task]}: fold {fold}/5 done", flush=True)
    df = pd.DataFrame(rows)
    summary = {}
    for model, g in df.groupby("model"):
        summary[model] = {m: {"mean": float(g[m].mean()), "sd": float(g[m].std(ddof=1))}
                          for m in ["accuracy", "balanced_accuracy", "precision", "recall",
                                    "specificity", "f1", "auc"]}
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(np.concatenate(oof_y), np.concatenate(oof_p))
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int))
    return summary, {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    csv = next((p for p in config.DATASET_CANDIDATES if p.exists()), None)
    if csv is None:
        sys.exit("Survey CSV not found in data/.")
    X, score, sleep = load_survey(csv)
    print(f"Loaded {len(X)} respondents; {int(sleep.sum())} meet the sleep-risk rule.")

    groups = {"sleep_risk": np.arange(len(X)), "cognitive": np.flatnonzero(sleep.to_numpy())}
    vocab = option_vocabulary(X)
    age = X[FEATURES[0]].dropna()
    evaluation = {"trained_at": datetime.now().isoformat(timespec="seconds"), "tasks": {}}
    versions = {}
    for mod in ("catboost", "xgboost", "lightgbm", "sklearn", "torch", "pandas", "numpy"):
        versions[mod] = __import__(mod).__version__

    for task, ids in groups.items():
        print(f"\n== {TASKS[task]}: final fit on {len(ids)} respondents")
        labels, cutoff = _labels(task, sleep, score, ids)
        comp, head = pipeline.fit_stack(X.iloc[ids], labels[ids], SEED, task == "sleep_risk", THREADS)
        importance = pd.Series(comp.cat.get_feature_importance(), index=X.columns)
        meta = {
            "task": task, "label": TASKS[task], "n_train": int(len(ids)),
            "positive_rate": float(labels[ids].mean()), "score_cutoff": cutoff,
            "features": FEATURES, "categorical": CATEGORICAL, "threshold": 0.5,
            "options": vocab, "age_range": [int(age.min()), int(age.max())],
            "age_median": float(age.median()),
            "feature_importance": importance.sort_values(ascending=False).round(3).to_dict(),
            "trained_at": evaluation["trained_at"], "versions": versions,
            "python": platform.python_version(),
        }
        pipeline.save_task(config.MODELS_DIR / task, comp, head, meta)
        evaluation["tasks"][task] = {"label": TASKS[task], "n": int(len(ids)),
                                     "positive_rate": meta["positive_rate"], "cutoff": cutoff}

    dist = score.value_counts().sort_index()
    eligible = score[sleep.eq(1)].value_counts().sort_index()
    evaluation["score_distribution"] = {
        "scores": list(range(SCORE_MAX + 1)),
        "all": [int(dist.get(s, 0)) for s in range(SCORE_MAX + 1)],
        "sleep_risk_only": [int(eligible.get(s, 0)) for s in range(SCORE_MAX + 1)],
    }
    evaluation["dataset"] = {"respondents": int(len(X)), "sleep_risk": int(sleep.sum()),
                             "predictors": len(FEATURES)}

    if not args.skip_cv:
        for task, ids in groups.items():
            print(f"\n== {TASKS[task]}: 5-fold cross-validation")
            summary, roc = cross_validate(task, X, score, sleep, ids)
            evaluation["tasks"][task].update({"cv": summary, "roc": roc})

    (config.MODELS_DIR / "eval_metrics.json").write_text(json.dumps(evaluation, indent=2))
    print("\nDone. Artifacts written to", config.MODELS_DIR)


if __name__ == "__main__":
    main()
