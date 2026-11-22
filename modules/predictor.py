"""
modules/predictor.py
====================
The single inference path for the app.

Answers (12 predictors) -> two trained stacks -> a ``Prediction``. If the
trained artifacts are missing, :func:`models_ready` is False and the UI says
so; nothing is ever fabricated.

Interpretation note. The notebook's honest cross-validation shows the stacks
barely beat the majority-class baseline (AUC near 0.5). The raw probability
therefore hovers near the cohort base rate, so the result is reported
*relative to that base rate* rather than as a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

import config
from modules.ml.survey import AGE, FEATURES

# A result within this many percentage points of the cohort base rate is
# described as "in line with" the cohort.
BAND_PP = 3.0

SHORT = {
    "What is your age?": "Age",
    "What is your gender?": "Gender",
    "What is your University name?": "University",
    "Your current year of study?": "Year of study",
    "How many cups of caffeine (tea/coffee/energy drinks) do you have daily?": "Daily caffeine",
    "How often do you use electronic devices (phone/laptop) in bed before sleep?": "Screens in bed",
    "How often do you feel cognitive load?": "Cognitive load",
    "How often do you feel stressed?": "Stress",
    "How often do you wake up in the middle of the night?": "Night wakings",
    "How many academic deadlines have you missed or delayed in the last month?": "Missed deadlines",
    "To what extent do you feel sleep deprivation is lowering your GPA?": "GPA impact",
    "How often does physical or mental fatigue prevent you from completing your daily goals?": "Fatigue",
}


@dataclass
class TaskResult:
    key: str
    label: str
    probability: float
    base_rate: float
    parts: dict[str, float]
    drivers: list[tuple[str, str, float]] = field(default_factory=list)  # name, value, shap

    @property
    def delta_pp(self) -> float:
        return (self.probability - self.base_rate) * 100

    @property
    def band(self) -> str:
        if self.delta_pp >= BAND_PP:
            return "above"
        if self.delta_pp <= -BAND_PP:
            return "below"
        return "inline"


@dataclass
class Prediction:
    sleep: TaskResult
    cognitive: TaskResult


def models_ready() -> bool:
    return all((d / "meta.json").exists() for d in config.TASK_DIRS.values())


@st.cache_resource(show_spinner=False)
def _load(task: str):
    from modules.ml.pipeline import LoadedTask

    return LoadedTask(config.TASK_DIRS[task])


def task_meta(task: str) -> dict:
    return config.load_json(config.TASK_DIRS[task] / "meta.json")


def evaluation() -> dict:
    return config.load_json(config.EVAL_METRICS)


def _frame(answers: dict[str, object]) -> pd.DataFrame:
    row = {c: answers[c] for c in FEATURES}
    row[AGE] = float(row[AGE])
    return pd.DataFrame([row], columns=FEATURES)


def predict(answers: dict[str, object]) -> Prediction:
    frame = _frame(answers)
    out = {}
    for task in ("sleep_risk", "cognitive"):
        model = _load(task)
        parts = model.score(frame)
        shap = model.shap(frame)
        drivers = sorted(
            ((SHORT[c], str(answers[c]), v) for c, v in shap.items()),
            key=lambda t: abs(t[2]), reverse=True)
        out[task] = TaskResult(task, model.meta["label"], parts["Stack"],
                               model.meta["positive_rate"], parts, drivers)
    return Prediction(sleep=out["sleep_risk"], cognitive=out["cognitive"])
