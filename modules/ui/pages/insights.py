"""Reliability page: cross-validated metrics, ROC, distributions, importance."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from modules import predictor
from modules.predictor import SHORT
from modules.ui import theme
from modules.ui.theme import esc, md

ORDER = ["Majority", "CatBoost", "XGBoost", "LightGBM", "ANN", "Stack"]
NAMES = {"Majority": "Majority baseline", "Stack": "Four-model stack"}


def _style(fig: go.Figure, h: int = 340) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#aab3d1", family="Inter"),
                      legend=dict(orientation="h", y=1.12), transition=dict(duration=600))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    return fig


def _table(cv: dict) -> str:
    rows = []
    for m in ORDER:
        if m not in cv:
            continue
        d = cv[m]
        cls = ' class="hl"' if m == "Stack" else ""
        cell = lambda k: f'{d[k]["mean"]:.3f} <span style="color:#6b7597">±{d[k]["sd"]:.3f}</span>'
        rows.append(f'<tr{cls}><td>{esc(NAMES.get(m, m))}</td><td>{cell("accuracy")}</td>'
                    f'<td>{cell("balanced_accuracy")}</td><td>{cell("auc")}</td></tr>')
    return ('<table class="tbl"><tr><th>Model</th><th>Accuracy</th><th>Balanced acc.</th><th>ROC-AUC</th></tr>'
            + "".join(rows) + "</table>")


def render() -> None:
    ev = predictor.evaluation()
    if not ev:
        md(theme.callout("Evaluation file missing. Run <code>python scripts/train_models.py</code>.", "rose"))
        return
    tasks = ev["tasks"]
    md(theme.kicker("Reliability") + '<div class="step-title rise">How far can you trust this?</div>'
       '<p class="lead rise d1">Five-fold cross-validation on the survey, reported as mean ± sample SD. '
       "Everything below is produced by <code>scripts/train_models.py</code>.</p>")

    cog_cv = tasks.get("cognitive", {}).get("cv", {})
    if cog_cv:
        st_acc = cog_cv["Stack"]["accuracy"]["mean"]
        mj_acc = cog_cv["Majority"]["accuracy"]["mean"]
        st_auc = cog_cv["Stack"]["auc"]["mean"]
        md('<div class="grid g3">'
           + theme.stat(f"{st_auc:.2f}", "cognitive-screen ROC-AUC (0.50 = chance)", delay=1)
           + theme.stat(f"{st_acc * 100:.1f}%", "stack accuracy", delay=2)
           + theme.stat(f"{mj_acc * 100:.1f}%", "accuracy of always guessing the majority", delay=3)
           + "</div>")
        st.write("")
        md(theme.callout(
            "<b>What this means.</b> The stack does not beat simply predicting the most common class, and "
            "its AUC is close to 0.5. With the symptom questions removed from the inputs, these twelve "
            "lifestyle answers do not separate students with higher and lower symptom scores in this "
            "dataset.", "warn", "🔬"))

    for key, title in (("cognitive", "Cognitive-symptom screen (primary)"), ("sleep_risk", "Sleep-risk screen (secondary)")):
        t = tasks.get(key)
        if not t or "cv" not in t:
            continue
        theme.section(title, f"{t['n']:,} respondents · positive rate {t['positive_rate'] * 100:.1f}%"
                      + (f" · symptom cutoff ≥ {t['cutoff']:.0f}/17" if t.get("cutoff") else ""))
        md(f'<div class="card rise">{_table(t["cv"])}</div>')

    theme.section("ROC curves", "Out-of-fold stack predictions. The dashed line is a coin flip.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color="#6b7597", dash="dash")))
    for key, color in (("cognitive", "#8b7cf6"), ("sleep_risk", "#22d3ee")):
        roc = tasks.get(key, {}).get("roc")
        if roc:
            auc = tasks[key]["cv"]["Stack"]["auc"]["mean"]
            fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", fill="tozeroy" if key == "cognitive" else None,
                                     fillcolor="rgba(139,124,246,.10)",
                                     name=f'{tasks[key]["label"]} (AUC {auc:.2f})', line=dict(color=color, width=2.5)))
    fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate")
    st.plotly_chart(_style(fig, 380), use_container_width=True, config={"displayModeBar": False})

    sd = ev.get("score_distribution")
    if sd:
        theme.section("Cognitive-symptom score", "Sum of seven self-reported items (0–17). "
                      "These items are the outcome and are not used as inputs.")
        fig = go.Figure()
        fig.add_bar(x=sd["scores"], y=sd["all"], name="All respondents", marker_color="rgba(139,124,246,.75)")
        fig.add_bar(x=sd["scores"], y=sd["sleep_risk_only"], name="Meet sleep-risk rule", marker_color="rgba(34,211,238,.75)")
        fig.update_layout(barmode="group", bargap=.25)
        st.plotly_chart(_style(fig), use_container_width=True, config={"displayModeBar": False})

    imp = predictor.task_meta("cognitive").get("feature_importance")
    if imp:
        theme.section("What the CatBoost model leaned on", "Training-set feature importance. "
                      "Importance is not evidence of a real effect when AUC is near chance.")
        items = sorted(((SHORT.get(k, k), v) for k, v in imp.items()), key=lambda t: t[1])
        fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                               marker=dict(color=[v for _, v in items], colorscale=[[0, "#22d3ee"], [1, "#8b7cf6"]])))
        st.plotly_chart(_style(fig, 400), use_container_width=True, config={"displayModeBar": False})

    theme.section("Method in brief")
    md('<div class="card rise"><p style="line-height:1.8">'
       "<b>Data.</b> Cross-sectional survey of Bangladeshi university students; provenance, ethics approval "
       "and consent details are still unverified in the notebook.<br>"
       "<b>Predictors.</b> Twelve self-reported items. The seven symptom items and two sleep-rule items "
       "define the labels and are excluded.<br>"
       "<b>Stack.</b> CatBoost, XGBoost, LightGBM and a PyTorch ANN; a logistic head learned on 3-fold "
       "out-of-fold scores, components refitted on the full training split.<br>"
       "<b>Evaluation.</b> Five-fold CV with the symptom cutoff (72nd percentile) recomputed on each "
       "training split. Decision threshold 0.5.</p></div>")
    theme.footer()
