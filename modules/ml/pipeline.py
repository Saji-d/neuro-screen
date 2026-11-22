"""
modules/ml/pipeline.py
======================
The four-model stack from ``thesis-updated-code.ipynb`` (sections 4-5).

Components (all trained on the same 12 self-reported predictors):

* CatBoost  - native categorical handling, 800 iters, depth 5, lr .002
* XGBoost   - 180 trees, depth 3, lr .045 on one-hot + scaled age
* LightGBM  - 180 trees, 15 leaves, lr .045 on the same encoding
* ANN       - 128-64-1 MLP, dropout .6/.4, Adam lr .008, 40 epochs

A logistic-regression head is learned on out-of-fold component scores
(3-fold) and the components are then refitted on the full training set.
At inference ``head(components(x))`` gives the stack probability.

Heavy imports (torch, catboost, ...) are deferred so the Streamlit shell
can start and show a status banner when they are missing.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from modules.ml.survey import AGE, CATEGORICAL

BASE = ["CatBoost", "XGBoost", "LightGBM", "ANN"]


# ------------------------------------------------------------------ training
def _make_preprocessor(raw: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical = [c for c in raw if c != AGE]
    return ColumnTransformer([
        ("age", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         [AGE] if AGE in raw else []),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def _ann_class():
    import torch.nn as nn

    class SymptomANN(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(width, 128), nn.ReLU(), nn.Dropout(.6),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.4),
                nn.Linear(64, 1), nn.Sigmoid())

        def forward(self, x):
            return self.layers(x).flatten()

    return SymptomANN


@dataclass
class Components:
    prep: object
    cat: object
    xgb: object
    lgb: object
    net: object
    prior: float


def fit_components(raw: pd.DataFrame, y: np.ndarray, seed: int, threads: int = 4) -> Components:
    import torch
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from xgboost import XGBClassifier

    categorical = [c for c in raw if c != AGE]
    cat = CatBoostClassifier(
        iterations=800, depth=5, learning_rate=.002, eval_metric="AUC", verbose=False,
        thread_count=threads, allow_writing_files=False, random_seed=seed,
    ).fit(raw, y, cat_features=categorical)

    prep = _make_preprocessor(raw)
    a = prep.fit_transform(raw).astype("float32")
    xgb = XGBClassifier(n_estimators=180, max_depth=3, learning_rate=.045,
                        eval_metric="logloss", n_jobs=threads, random_state=seed).fit(a, y)
    lgb = LGBMClassifier(n_estimators=180, num_leaves=15, learning_rate=.045,
                         n_jobs=threads, random_state=seed, verbosity=-1).fit(pd.DataFrame(a), y)

    torch.manual_seed(seed)
    net = _ann_class()(a.shape[1])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(a), torch.tensor(y, dtype=torch.float32)),
        batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(net.parameters(), lr=.008)
    loss_fn = nn.BCELoss()
    net.train()
    for _ in range(40):
        for batch, target in loader:
            opt.zero_grad()
            loss_fn(net(batch), target).backward()
            opt.step()
    net.eval()
    return Components(prep, cat, xgb, lgb, net, float(np.mean(y)))


def component_scores(c: Components, raw: pd.DataFrame) -> pd.DataFrame:
    import torch

    a = c.prep.transform(raw).astype("float32")
    with torch.no_grad():
        ann = c.net(torch.from_numpy(a)).numpy()
    return pd.DataFrame({
        "Majority": c.prior,
        "CatBoost": c.cat.predict_proba(raw)[:, 1],
        "XGBoost": c.xgb.predict_proba(a)[:, 1],
        "LightGBM": c.lgb.predict_proba(pd.DataFrame(a))[:, 1],
        "ANN": ann,
    })


def fit_stack(X: pd.DataFrame, y: np.ndarray, seed: int, stratify: bool, threads: int = 4):
    """Fit the four-model stack on (X, y). Returns (components, head)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold

    oof = pd.DataFrame(np.nan, index=range(len(X)), columns=BASE)
    splitter = (StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y) if stratify
                else KFold(3, shuffle=True, random_state=seed).split(X))
    for fold, (i, j) in enumerate(splitter, 1):
        comp = fit_components(X.iloc[i], y[i], seed + fold, threads)
        oof.iloc[j] = component_scores(comp, X.iloc[j])[BASE].to_numpy()
    head = LogisticRegression().fit(oof, y)
    return fit_components(X, y, seed, threads), head


def stack_scores(comp: Components, head, raw: pd.DataFrame) -> pd.DataFrame:
    p = component_scores(comp, raw)
    p["Stack"] = head.predict_proba(p[BASE])[:, 1]
    return p


# --------------------------------------------------------------- persistence
def save_task(directory: Path, comp: Components, head, meta: dict) -> None:
    import json
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    comp.cat.save_model(str(directory / "catboost.cbm"))
    comp.xgb.save_model(str(directory / "xgboost.json"))
    comp.lgb.booster_.save_model(str(directory / "lightgbm.txt"))
    torch.save({"state_dict": comp.net.state_dict(),
                "width": comp.net.layers[0].in_features}, directory / "ann.pt")
    with open(directory / "sklearn.pkl", "wb") as fh:
        pickle.dump({"prep": comp.prep, "head": head, "prior": comp.prior}, fh)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


class LoadedTask:
    """Inference-only view of one trained task."""

    def __init__(self, directory: Path):
        import json
        import lightgbm as lgb
        import torch
        import xgboost as xgb
        from catboost import CatBoostClassifier

        self.meta = json.loads((directory / "meta.json").read_text())
        self.cat = CatBoostClassifier().load_model(str(directory / "catboost.cbm"))
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(str(directory / "xgboost.json"))
        self.lgb = lgb.Booster(model_file=str(directory / "lightgbm.txt"))
        ckpt = torch.load(directory / "ann.pt", map_location="cpu", weights_only=True)
        self.net = _ann_class()(ckpt["width"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        with open(directory / "sklearn.pkl", "rb") as fh:
            blob = pickle.load(fh)
        self.prep, self.head, self.prior = blob["prep"], blob["head"], blob["prior"]

    def score(self, raw: pd.DataFrame) -> dict[str, float]:
        import torch

        a = self.prep.transform(raw).astype("float32")
        with torch.no_grad():
            ann = float(self.net(torch.from_numpy(a)).numpy()[0])
        parts = {
            "CatBoost": float(self.cat.predict_proba(raw)[0, 1]),
            "XGBoost": float(self.xgb.predict_proba(a)[0, 1]),
            "LightGBM": float(self.lgb.predict(a)[0]),
            "ANN": ann,
        }
        stack = float(self.head.predict_proba(pd.DataFrame([parts])[BASE])[0, 1])
        return {**parts, "Stack": stack}

    def shap(self, raw: pd.DataFrame) -> dict[str, float]:
        """Per-feature CatBoost SHAP contributions (log-odds) for one row."""
        from catboost import Pool

        pool = Pool(raw, cat_features=[c for c in raw if c != AGE])
        values = self.cat.get_feature_importance(pool, type="ShapValues")[0]
        return {col: float(values[i]) for i, col in enumerate(raw.columns)}
