# Neuro-Screen

A hybrid CatBoost + ANN screening tool that detects cognitive-impairment risk factors linked to insomnia in university students.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-FFCC00)](https://catboost.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey)](#disclaimer)

> Undergraduate thesis project — Group **G52**, Department of Computer Science, American International University-Bangladesh (AIUB).

## Live Demo

Neuro-Screen is a Python/Streamlit application, which cannot run on a static/edge host like Vercel — it needs a live Python process. It is designed to run locally or on a Streamlit-compatible host (e.g. Streamlit Community Cloud). See [Running Locally](#running-locally) below. This README will be updated with a live URL once the app is deployed to a compatible platform.

## Overview

Cognitive impairment linked to poor sleep is an increasingly common but under-screened problem among university students. **Neuro-Screen** is a research prototype that turns a short, self-reported questionnaire about sleep, lifestyle, and everyday cognitive symptoms into a risk assessment — using a trained hybrid machine-learning model rather than a rules-of-thumb score.

The dashboard walks a user through a 21-question check-in (as a form or as a conversational chat), runs the trained model, and explains *why* it produced that result using feature-contribution analysis — while keeping the model, the preprocessing pipeline, and the reported thesis metrics completely intact.

## Research Objective

Sleep disruption and insomnia are widespread among university students, and their link to reduced cognitive performance — memory lapses, poor concentration, slowed decision-making — is well documented but rarely screened for in an academic setting. Neuro-Screen's objective is to build and validate an automated, reproducible framework that flags students at risk of insomnia-related cognitive impairment early, using inexpensive self-report data instead of clinical instrumentation, so that at-risk students can be pointed toward academic and psychological support before the problem compounds.

## Key Features

- **21-question assessment** — a quick check-in mapped 1:1 to the trained model's feature columns (sleep, lifestyle, stress, and cognitive-symptom questions)
- **Hybrid CatBoost + ANN inference** — every prediction runs through the real trained pipeline; there is no mock or randomized scoring path
- **Risk score & prediction** — a blended probability, severity tier, and Healthy/Impaired classification
- **Explainability** — feature-contribution breakdown showing which inputs drove the individual prediction, alongside the official paper-reported evaluation
- **Recommendations** — practical, sleep/lifestyle-focused suggestions generated from the result's contributing factors
- **Results visualization** — risk meter, per-model (CatBoost / ANN / Hybrid) probability breakdown, confusion matrix and ROC curve
- **Conversational assistant** — a chat interface that collects the same 21 answers in natural conversation and calls the identical inference pipeline, with a built-in crisis-language safety redirect

## Methodology

```
Survey data → Cleaning / feature engineering → Stratified 80/20 split
    → CatBoost (native categorical handling)   ─┐
    → ANN (128 → 64 → 1, one-hot + scaled input) ─┼→ Hybrid probability → Prediction
                                                   ┘        ↓
                                          Feature-contribution explanation → Recommendations
```

- **CatBoost** — gradient-boosted trees trained directly on the native categorical survey responses (no one-hot encoding), chosen for its strength with categorical, non-linear feature interactions.
- **Artificial Neural Network (ANN)** — a 3-layer MLP (128 → 64 → 1, ReLU/Sigmoid, dropout 0.6/0.4) trained on a one-hot encoded, standard-scaled version of the same features, for complementary feature extraction.
- **Hybrid Ensemble** — the two models' probabilities are blended by simple arithmetic mean (`P_hybrid = (P_CatBoost + P_ANN) / 2`), thresholded at 0.5 for the final Healthy/Impaired classification.
- **Explainability** — the Results and Explainability pages surface each feature's contribution to the specific prediction just made, kept strictly separate from the aggregate, paper-reported feature importances.
- **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, and the confusion matrix, computed on a held-out 20% stratified test split.

The target label (`cognitive_impairment`) is derived from seven self-reported cognitive-symptom questions (mental stamina, spacing out, audio lag, forgetfulness, reminder reliance, brain fog, decision-making), summed into a cumulative score and binarized at its 72nd percentile.

## Model Performance

Official numbers from the Neuro-Screen conference paper (ICCA 2026) and thesis report, evaluated on the held-out test split (n = 447):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| CatBoost | 94.12% | 93.80% | 95.00% | 94.39% | 0.9780 |
| ANN | 91.25% | 89.50% | 92.00% | 90.73% | 0.9450 |
| **Hybrid (Neuro-Screen)** | **95.20%** | **94.40%** | **96.10%** | **95.24%** | **0.9820** |

**Confusion matrix (Hybrid, n = 447):** TN 212 · FP 12 · FN 9 · TP 214

The app's **Explainability** page also reproduces evaluation metrics live from the currently trained artifacts (`models/eval_metrics.json`) and labels them separately as **Live (trained)**, so a re-trained model's numbers are never confused with the officially published **Reported (paper)** results.

## Dataset

- **2,237** anonymized survey responses from university students aged 20–35 across multiple universities in Bangladesh
- **22** raw columns → **21** model input features (age, gender, university, academic year, and 17 lifestyle/sleep/cognitive-symptom questions)
- **80/20 stratified split** (random seed 42) → 1,790 training / 447 test rows
- ~75.7% of respondents self-report as insomniac; ~37.1% are labeled cognitively impaired under the derived target
- Missing values are treated as a genuine `"Missing"` category (not imputed), since CatBoost handles categorical missingness natively

## Tech Stack

- **Python 3.11**
- **Streamlit** — application framework and UI
- **CatBoost** — gradient boosting classifier
- **PyTorch** — the ANN model
- **scikit-learn** — preprocessing pipeline, scaler, label encoding, evaluation metrics
- **Pandas / NumPy** — data handling and feature engineering
- **Plotly** — risk meter, confusion matrix, ROC curve, and feature-importance charts

## Project Structure

```
app.py                          → page shell, theme, native multipage router
config.py                       → paths, brand identity, risk thresholds, JSON helpers
data/
  Thesis Dataset.csv            → raw survey dataset
  feature_schema.json           → UI → model-column mapping (21 features)
  reported_results.json         → official paper-reported numbers
models/                         → trained artifacts (CatBoost, ANN, scaler, encoders, metrics)
modules/
  core/preprocessing.py         → notebook-exact feature engineering
  core/risk.py                  → severity tiers, contributing factors, recommendations
  core/model_defs.py            → ANN architecture definition
  predictor.py                  → the single inference path (CatBoost + ANN hybrid)
  chat/assistant.py             → conversational check-in (same inference pipeline)
  charts/charts.py              → Plotly figures (confusion matrix, ROC, importances)
  ui/pages/                     → Home, Assessment, Results, Explainability, Assistant
scripts/
  train_models.py               → reproduces the thesis notebook pipeline, exports artifacts
  verify_predictions.py         → checks dashboard vs. notebook prediction parity
assets/                         → logo, stylesheet
```

## Running Locally

```bash
git clone https://github.com/Saji-d/neuro-screen.git
cd neuro-screen

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

The trained model artifacts already ship in `models/`, so predictions work out of the box. To retrain from scratch (after installing `catboost` and `torch`, which are optional extras for training only):

```bash
python scripts/train_models.py
python scripts/verify_predictions.py   # sanity-checks dashboard vs. notebook parity
```

## Thesis Team

**Thesis Group:** G52

| Author | Student ID |
|---|---|
| Sajidur Rahman Sajid | 22-49076-3 |
| Khadija Akter | 22-48295-3 |
| Md Iqramul Kabir | 22-46165-1 |
| Shahadat Hossain Gazi | 22-48095-2 |

## Academic Context

American International University-Bangladesh (AIUB)
CSC 4298 — Thesis/Project · Spring 2025–2026

## Disclaimer

Neuro-Screen is an academic research prototype for screening cognitive-impairment **risk factors** associated with insomnia. It is **not a medical diagnostic tool** and does not replace professional healthcare, clinical assessment, or psychological support. If you or someone you know is in distress, please contact a qualified healthcare provider or your institution's psychological support services.

## Credits

Developed by **Sajidur Rahman Sajid** as part of the Neuro-Screen thesis, alongside thesis group G52 (Khadija Akter, Md Iqramul Kabir, Shahadat Hossain Gazi) at AIUB.
