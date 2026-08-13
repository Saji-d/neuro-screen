"""
modules/core/preprocessing.py
=============================
Preprocessing that EXACTLY mirrors Thesis3.ipynb.

The notebook pipeline is the source of truth:

1. Regex/fixing pass over object columns (encoding artifact ``â€“`` -> ``-``).
2. Seven self-reported cognitive symptoms are scored and summed into a
   cumulative ``cognitive_score``.
3. The binary target ``cognitive_impairment`` is ``cognitive_score >=
   quantile(0.72)``; an ``insomniac`` flag is derived from sleep hours and
   sleep quality.
4. Features = every column except the derived symptom scores, the target,
   the insomniac flag and ``Timestamp`` (== 21 features on the 22-column
   survey).
5. Missing values are filled with the string ``"Missing"`` (this is a real
   category the model learned, never an imputed number).
6. CatBoost consumes the categorical columns natively; the ANN consumes a
   one-hot encoded frame whose numeric column (age) is standard-scaled.

These helpers are shared by scripts/train_models.py (training) and
modules/predictor.py (inference) so the two paths can never drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The seven symptom questions aggregated into the target. Values are the
# per-answer scores used in Thesis3.ipynb.
SYMPTOM_MAPS: dict[str, dict[str, int]] = {
    'How would you rate your "Mental Stamina" (the ability to think hard for long periods)?': {
        '1 (Total Exhaustion)': 3, '2 (Fading Fast)': 2, '3 (Full Energy)': 0},
    'How often do you find yourself "spacing out" or losing your train of thought during conversations?': {
        'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Always': 3},
    'Do you experience "Audio Lag" (hearing someone speak but taking a few seconds to understand the meaning)?': {
        'No': 0, 'Maybe': 1, 'Yes': 2},
    'How often do you forget basic information in your daily life?': {
        'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Always': 3},
    'How much do you rely on "external reminders" (alarms, notes, friends) to remember basic daily tasks?': {
        'Not at all': 0, 'Sometimes': 1, 'Completely dependent': 2},
    'How frequently do you experience "Brain Fog" (confusion or lack of clarity)?': {
        'Never': 0, 'Sometimes': 1, 'Always': 2},
    'Do you struggle with making quick decisions or solving logical problems?': {
        'Never': 0, 'Sometimes': 1, 'Always': 2},
}

TARGET_QUANTILE = 0.72
MISSING_TOKEN = "Missing"

INSOMNIAC_SLEEP_HOURS = {"Less than 5 hours", "5-6 hours"}
INSOMNIAC_SLEEP_QUALITY = {"1 (Very Poor)", "2 (Poor)", "3 (Average)"}

NUMERIC_FEATURES = ["What is your age?"]

TARGET_COLUMN = "cognitive_impairment"
INSOMNIAC_COLUMN = "insomniac"
COGNITIVE_SCORE_COLUMN = "cognitive_score"
TIMESTAMP_COLUMN = "Timestamp"

EXCLUDED_COLUMNS = {
    TIMESTAMP_COLUMN,
    COGNITIVE_SCORE_COLUMN,
    TARGET_COLUMN,
    INSOMNIAC_COLUMN,
}


def _to_object(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise string columns to pandas 'object' dtype.

    The notebook ran on pandas 2.x, where ``pd.read_csv`` materialises text
    columns as ``object``. pandas 3.x reads them as the new ``str`` dtype,
    which silently breaks ``dtype == "object"`` checks (and therefore the
    categorical-feature detection). Recasting to ``object`` reproduces the
    notebook's behaviour exactly.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            if df[col].dtype != "object":
                df[col] = df[col].astype("object")
    return df


def fix_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's string-cleaning pass (encoding fix + strip).

    The notebook replaces the cp1252-mojibake of a UTF-8 en-dash with a plain
    hyphen. The delivered CSV carries the same corruption in two byte variants
    (third byte decoded as U+2013 en-dash or U+201C left double quote), so both
    are collapsed to ``-``. Raw unicode dashes are normalised too. The result is
    the exact category vocabulary the notebook model learned (e.g. ``5-6 hours``).
    """
    df = _to_object(df)
    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = df[col].astype(str)
            for token in ("\u2013", "\u2014", "â€“", "â€\u201c"):
                cleaned = cleaned.str.replace(token, "-", regex=False)
            df[col] = cleaned.str.strip().astype("object")
    return df


def derive_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add symptom scores, cognitive score, binary target and insomniac flag."""
    df = df.copy()
    symptom_cols = [col for col in SYMPTOM_MAPS if col in df.columns]
    for col in symptom_cols:
        df[col + "_score"] = df[col].map(SYMPTOM_MAPS[col])

    cognitive_cols = [col + "_score" for col in symptom_cols]
    df[COGNITIVE_SCORE_COLUMN] = df[cognitive_cols].sum(axis=1)

    threshold = df[COGNITIVE_SCORE_COLUMN].quantile(TARGET_QUANTILE)
    df[TARGET_COLUMN] = (df[COGNITIVE_SCORE_COLUMN] >= threshold).astype(int)

    df[INSOMNIAC_COLUMN] = (
        df["On average, how many hours of actual sleep do you get per night?"]
        .isin(INSOMNIAC_SLEEP_HOURS)
        | df["How would you rate your overall sleep quality?"]
        .isin(INSOMNIAC_SLEEP_QUALITY)
    ).astype(int)
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) exactly as the notebook builds them.

    X contains every column except the derived symptom scores, the cognitive
    score, the target, the insomniac flag and the timestamp. Missing values
    become the string ``"Missing"``.
    """
    df = df.copy()
    exclude = set()
    for col in SYMPTOM_MAPS:
        if col + "_score" in df.columns:
            exclude.add(col + "_score")
    exclude |= EXCLUDED_COLUMNS

    feature_cols = [col for col in df.columns if col not in exclude]
    X = df[feature_cols].fillna(MISSING_TOKEN)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def detect_categorical_features(X: pd.DataFrame) -> list[str]:
    """Categorical feature list exactly as the notebook derives it.

    Text columns are categorical. The check accepts both the classic
    ``object`` dtype and pandas 3.x's ``str`` dtype so the detection is robust
    either way.
    """
    return [col for col in X.columns
            if pd.api.types.is_object_dtype(X[col])
            or pd.api.types.is_string_dtype(X[col])]


def build_ann_frame(
    X: pd.DataFrame,
    categorical_features: list[str],
    scaler,
    ann_feature_columns: list[str],
) -> pd.DataFrame:
    """One-hot encode + scale a frame for ANN inference.

    The one-hot columns produced by ``pd.get_dummies`` on the *training* set
    are replayed: the row is one-hot encoded with the same categorical
    columns, re-indexed to the persisted training column order (filling any
    absent category with 0), and the numeric column(s) are standardised with
    the persisted scaler.
    """
    frame = pd.get_dummies(X, columns=categorical_features)

    # Align to the exact training column layout (join='left', fill_value=0).
    frame = frame.reindex(columns=ann_feature_columns, fill_value=0)

    numeric_cols = [c for c in NUMERIC_FEATURES if c in frame.columns]
    if numeric_cols and scaler is not None:
        frame[numeric_cols] = scaler.transform(frame[numeric_cols].astype(np.float64))

    return frame.astype(np.float32)


def to_tensor_float(frame: pd.DataFrame):
    """Convert the ANN frame to the same float32 tensor the notebook builds."""
    import torch
    return torch.tensor(frame.values.astype(np.float32), dtype=torch.float32)
