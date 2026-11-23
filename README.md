<div align="center">
  <img src="assets/banner.svg" alt="Neuro-Screen banner" width="100%">
  <p><code>Four-model stack (CatBoost, XGBoost, LightGBM, ANN) that screens sleep risk and cognitive-symptom risk in university students from 12 self-reported answers.</code></p>
  <a href="https://neuro-screen.streamlit.app/">Live demo</a> · <a href="https://github.com/Saji-d/neuro-screen">Source</a>
</div>

> [!WARNING]
> **Research prototype, not a medical device.** Labels come from self-reported survey items, not a clinical test.

<div align="center"><img src="docs/preview.png" alt="Neuro-Screen preview" width="90%"></div>

## Results (5-fold cross-validation)

With the symptom and sleep-rule questions **removed from the predictors**, the stack does not beat the majority-class baseline:

| Task | Majority accuracy | Stack accuracy | Stack ROC-AUC |
|---|---|---|---|
| Cognitive symptoms (n=1,692) | 0.621 | 0.621 | 0.48 |
| Sleep risk (n=2,236) | 0.757 | 0.757 | 0.49 |

Symptom and sleep-rule items are excluded from the predictors by design, so the label cannot leak into the model. Full per-model tables are in the app's **Reliability** page and `models/eval_metrics.json`.

## Method

`notebooks/thesis-updated-code.ipynb` is the reference analysis. `scripts/train_models.py` reproduces its four-model stack: CatBoost, XGBoost and LightGBM plus a 128-64-1 PyTorch ANN, combined by a logistic head learned on 3-fold out-of-fold scores. The cognitive label is a symptom score >= the 72nd percentile of the training split, among respondents meeting the sleep-risk rule (<6 h sleep or quality <= 3/5).

## Run

```bash
pip install -r requirements.txt
python scripts/train_models.py      # writes models/ (add --skip-cv for a fast fit)
streamlit run app.py
```

## Structure

```
app.py                  Streamlit shell (top navigation)
config.py               paths + brand
modules/ml/             survey scoring + the stack (train and load)
modules/predictor.py    the single inference path
modules/ui/             theme, components and pages (home, assess, results, reliability)
scripts/train_models.py training + cross-validation
models/                 trained artifacts + eval_metrics.json
notebooks/              reference analysis
```

<img src="assets/divider.svg" alt="" width="100%">

## ◈ Author

**Sajidur Rahman Sajid**: Full-Stack Developer and AI engineer from Dhaka.<br>
BSc in CSE at American International University-Bangladesh (AIUB).

The survey dataset and the reference analysis (`notebooks/thesis-updated-code.ipynb`) come from a group project at AIUB. This app and repository are my own work.

<p>
  <a href="https://www.sajidur.me"><img src="https://img.shields.io/badge/Portfolio-0B0E1A?style=flat-square&logo=googlechrome&logoColor=9B8CFA" alt="Portfolio"></a>
  <a href="https://www.linkedin.com/in/sajidur-rahman-sajid/"><img src="https://img.shields.io/badge/LinkedIn-0B0E1A?style=flat-square&logo=linkedin&logoColor=8B7CF6" alt="LinkedIn"></a>
  <a href="https://github.com/Saji-d"><img src="https://img.shields.io/badge/GitHub-0B0E1A?style=flat-square&logo=github&logoColor=EEF0FA" alt="GitHub"></a>
  <a href="mailto:sajidsajidurrahman99@gmail.com"><img src="https://img.shields.io/badge/Email-0B0E1A?style=flat-square&logo=gmail&logoColor=9B8CFA" alt="Email"></a>
</p>
