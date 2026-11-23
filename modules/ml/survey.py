"""
modules/ml/survey.py
====================
Survey loading, scoring and target definition for Neuro-Screen.

Mirrors ``thesis-updated-code.ipynb`` (sections 1-3):

* The seven cognitive-symptom items are scored (0-17) and kept OUT of the
  predictors, as are the two sleep-rule items, so the label cannot leak into
  the model.
* ``sleep`` = 1 when the respondent sleeps <6 h or rates sleep quality <= 3.
* Two tasks: "Sleep risk" (all respondents) and "Cognitive symptoms"
  (respondents meeting the sleep-risk rule; label = score >= the 72nd
  percentile of the training split's scores).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AGE = "What is your age?"
DURATION = "On average, how many hours of actual sleep do you get per night?"
QUALITY = "How would you rate your overall sleep quality?"

SYMPTOM_MAPS: dict[str, dict[str, int]] = {
    'How would you rate your "Mental Stamina" (the ability to think hard for long periods)?': {
        "1 (Total Exhaustion)": 3, "2 (Fading Fast)": 2, "3 (Full Energy)": 0},
    'How often do you find yourself "spacing out" or losing your train of thought during conversations?': {
        "Never": 0, "Rarely": 1, "Sometimes": 2, "Always": 3},
    'Do you experience "Audio Lag" (hearing someone speak but taking a few seconds to understand the meaning)?': {
        "No": 0, "Maybe": 1, "Yes": 2},
    "How often do you forget basic information in your daily life?": {
        "Never": 0, "Rarely": 1, "Sometimes": 2, "Always": 3},
    'How much do you rely on "external reminders" (alarms, notes, friends) to remember basic daily tasks?': {
        "Not at all": 0, "Sometimes": 1, "Completely dependent": 2},
    'How frequently do you experience "Brain Fog" (confusion or lack of clarity)?': {
        "Never": 0, "Sometimes": 1, "Always": 2},
    "Do you struggle with making quick decisions or solving logical problems?": {
        "Never": 0, "Sometimes": 1, "Always": 2},
}

# The twelve predictors, in notebook order.
FEATURES = [
    AGE,
    "What is your gender?",
    "What is your University name?",
    "Your current year of study?",
    "How many cups of caffeine (tea/coffee/energy drinks) do you have daily?",
    "How often do you use electronic devices (phone/laptop) in bed before sleep?",
    "How often do you feel cognitive load?",
    "How often do you feel stressed?",
    "How often do you wake up in the middle of the night?",
    "How many academic deadlines have you missed or delayed in the last month?",
    "To what extent do you feel sleep deprivation is lowering your GPA?",
    "How often does physical or mental fatigue prevent you from completing your daily goals?",
]
CATEGORICAL = [c for c in FEATURES if c != AGE]

SCORE_QUANTILE = 0.72
SCORE_MAX = 17


def _clean_text(series: pd.Series) -> pd.Series:
    """Strip whitespace and collapse the CSV's mojibake en-dash to '-'."""
    s = series.astype(str).str.strip()
    for token in ("â€“", "â€”", "–", "—"):
        s = s.str.replace(token, "-", regex=False)
    return s.str.strip()


def load_survey(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, score, sleep) exactly as the notebook builds them."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = _clean_text(df[col])

    items = pd.DataFrame({c: df[c].map(m) for c, m in SYMPTOM_MAPS.items()})
    if items.isna().any().any():
        raise ValueError("Missing or unknown symptom response in survey.")
    score = items.sum(axis=1).rename("cognitive_score")
    assert score.between(0, SCORE_MAX).all()

    sleep = (
        df[DURATION].isin(["Less than 5 hours", "5-6 hours"])
        | df[QUALITY].isin(["1 (Very Poor)", "2 (Poor)", "3 (Average)"])
    ).astype(int)

    X = df[FEATURES].copy()
    X[AGE] = pd.to_numeric(X[AGE], errors="coerce").replace(2, np.nan)
    X[CATEGORICAL] = X[CATEGORICAL].fillna("Missing")
    return X, score, sleep


def option_vocabulary(X: pd.DataFrame) -> dict[str, list[str]]:
    """Observed options per categorical predictor (most frequent first)."""
    return {c: X[c].value_counts().index.tolist() for c in CATEGORICAL}
