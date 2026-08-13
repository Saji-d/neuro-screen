"""
modules/core/risk.py
====================
User-facing risk model on top of the trained hybrid classifier.

The trained model is a *binary* classifier (Healthy / Impaired) with a
decision threshold of 0.5 on the blended probability (as in Thesis3.ipynb).
This module converts a probability into the dashboard's result object:
prediction label, severity tier, model confidence, per-instance feature
contributions, and research-grounded recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import config

PREDICTION_LABELS = {0: "Healthy", 1: "Impaired"}


@dataclass
class FeatureDriver:
    """A single feature's contribution to the prediction (from SHAP values)."""

    feature: str          # readable feature name
    value: str            # the answer the student gave
    contribution: float   # signed SHAP contribution to P(impaired)

    @property
    def magnitude(self) -> float:
        return abs(self.contribution)


@dataclass
class RiskResult:
    """Complete result object produced by the single inference pipeline."""

    proba_impaired: float                 # hybrid probability of 'Impaired'
    cat_proba: Optional[float] = None
    ann_proba: Optional[float] = None
    prediction: int = 0
    severity: str = "Low Risk"
    drivers: list[FeatureDriver] = field(default_factory=list)
    threshold: float = 0.5
    model_name: str = "CatBoost + ANN hybrid (P = (P_catboost + P_ann) / 2)"
    model_available: bool = True
    trained_at: str = ""

    # -- derived properties -------------------------------------------------
    @property
    def prediction_label(self) -> str:
        return PREDICTION_LABELS.get(self.prediction, "Unknown")

    @property
    def risk_score(self) -> int:
        """0-100 risk meter value (probability of impairment)."""
        return int(round(self.proba_impaired * 100))

    @property
    def confidence(self) -> int:
        """Model confidence = max(P_imp, 1 - P_imp) in percent."""
        return int(round(max(self.proba_impaired, 1.0 - self.proba_impaired) * 100))

    @property
    def healthy_proba(self) -> float:
        return 1.0 - self.proba_impaired


def severity_for(proba: float) -> str:
    """Map a hybrid probability to a UI severity tier."""
    if proba < config.RISK["low_max"]:
        return "Low Risk"
    if proba < config.RISK["moderate_max"]:
        return "Moderate Risk"
    return "High Risk"


# ------------------------------------------------------------------
# Recommendations — written for students, grounded in the thesis findings
# (mental/physical fatigue, stress frequency, sleep quality and sleep hours
# were the strongest predictors of cognitive impairment).
# ------------------------------------------------------------------
_LOW_RECS = [
    "Your answers place you in the healthy range. Keep protecting your sleep — "
    "consistent bedtime and wake-up times are the single best habit.",
    "Take a short re-check in a week so we can catch any early changes while "
    "they are still small.",
]

_MODERATE_RECS = [
    "Sleep quality mattered slightly more than sleep hours in our study — try "
    "a screen-free wind-down 30 minutes before bed and see how it feels.",
    "Fatigue and stress were the strongest predictors we found. Break long "
    "study sessions into 20-25 minute blocks with short breaks.",
    "Re-run this assessment in a few days. Small lifestyle shifts often move "
    "the score quickly.",
]

_HIGH_RECS = [
    "Please talk to the AIUB Psychological Support Center this week — that is "
    "exactly what it is there for, and it is free for students.",
    "Try to recover sleep over the next two days even with deadlines — short "
    "term recovery is more effective than you expect.",
    "Let your academic advisor know that tiredness is making coursework "
    "harder; they can help adjust workload.",
    "Reduce caffeine after midday and avoid screens for 30 minutes before bed.",
]


def recommendations_for(result: RiskResult) -> list[str]:
    """Return tiered recommendations keyed to the severity tier."""
    if result.severity == "Low Risk":
        return list(_LOW_RECS)
    if result.severity == "High Risk":
        return list(_HIGH_RECS)
    return list(_MODERATE_RECS)


# ------------------------------------------------------------------
# Driver naming — maps the exact model column names (CSV question text) to
# readable labels.
# ------------------------------------------------------------------
DRIVER_LABELS: dict[str, str] = {
    "How often does physical or mental fatigue prevent you from completing your daily goals? ": "Mental / Physical Fatigue",
    "How often do you feel stressed?": "Stress Frequency",
    "To what extent do you feel sleep deprivation is lowering your GPA?": "Sleep Impact on GPA",
    "How would you rate your overall sleep quality?": "Overall Sleep Quality",
    "On average, how many hours of actual sleep do you get per night?": "Average Sleep Hours",
    "Your current year of study?": "Academic Year",
    "How often do you use electronic devices (phone/laptop) in bed before sleep?": "Bedtime Device Use",
    "How much do you rely on \"external reminders\" (alarms, notes, friends) to remember basic daily tasks?": "Reminder Reliance",
    "How many cups of caffeine (tea/coffee/energy drinks) do you have daily?": "Daily Caffeine Intake",
    "How would you rate your \"Mental Stamina\" (the ability to think hard for long periods)?": "Mental Stamina",
    "How often do you wake up in the middle of the night?": "Nightly Awakenings",
    "How many academic deadlines have you missed or delayed in the last month?": "Missed Deadlines",
    "What is your gender?": "Gender",
    "What is your age?": "Age",
    "What is your University name?": "University",
    "How often do you feel cognitive load?": "Cognitive Load",
    "How often do you find yourself \"spacing out\" or losing your train of thought during conversations?": "Spacing Out",
    "Do you experience \"Audio Lag\" (hearing someone speak but taking a few seconds to understand the meaning)?": "Audio Lag",
    "How often do you forget basic information in your daily life?": "Forgetfulness",
    "How frequently do you experience \"Brain Fog\" (confusion or lack of clarity)?": "Brain Fog",
    "Do you struggle with making quick decisions or solving logical problems?": "Decision-Making Difficulty",
}


def readable_driver(column_name: str) -> str:
    """Return a short human-readable label for a model column name."""
    return DRIVER_LABELS.get(column_name, column_name)
