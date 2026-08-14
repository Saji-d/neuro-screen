"""
modules/ui/pages/explainability.py
==================================
Explainability page — the model's performance story.

Two honest data sources are kept clearly separate:

* **Live (trained model)** — numbers reproduced from the artifacts generated
  by ``scripts/train_models.py`` (eval_metrics.json, roc_data.json). Only
  shown when those artifacts exist.
* **Reported (paper)** — the official evaluation numbers published in the
  ICCA 2026 paper (data/reported_results.json). Always shown.

The page also renders per-instance SHAP contributions for the most recent
prediction, if one exists in the session.
"""

from __future__ import annotations

import streamlit as st

import config
from modules.charts import charts
from modules.ui import components
from modules.ui.theme import esc, glass_card

_METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc"]
_METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]


def _live_available() -> bool:
    return config.ARTIFACTS["eval"].exists() and config.ARTIFACTS["roc"].exists()


def _live_section() -> None:
    components.section_title("Live · trained model", "Reproduced on the held-out test set")
    metrics_all = config.load_json(config.ARTIFACTS["eval"])

    rows = {
        model: [m.get(k, 0) for k in _METRIC_ORDER]
        for model, m in metrics_all.items() if isinstance(m, dict)
    }

    if rows.get("Hybrid"):
        h = metrics_all["Hybrid"]
        cols = st.columns(5, gap="medium")
        for col, (name, key) in zip(cols, zip(_METRIC_NAMES, _METRIC_ORDER)):
            with col:
                components.stat_card(
                    icon="🎯", label=name, value=f"{h.get(key, 0)*100:.1f}%",
                    sub="Hybrid",
                )

    st.markdown("")
    if rows:
        st.plotly_chart(charts.metrics_grouped(rows, _METRIC_NAMES),
                        width="stretch")
        st.markdown("")
        cm = metrics_all.get("Hybrid", {}).get("confusion_matrix", {})
        if cm:
            st.plotly_chart(charts.confusion_matrix(
                cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0), cm.get("tp", 0)),
                width="stretch")

    roc = config.load_json(config.ARTIFACTS["roc"])
    hybrid_roc = roc.get("Hybrid", {}) or roc.get("hybrid", {})
    if hybrid_roc.get("fpr") and hybrid_roc.get("tpr"):
        st.markdown("")
        st.plotly_chart(charts.roc_curve(
            hybrid_roc["fpr"], hybrid_roc["tpr"], float(hybrid_roc.get("auc", 0))),
            width="stretch")


def _reported_section() -> None:
    components.section_title("Reported · paper", "Official ICCA 2026 evaluation")
    paper = config.load_json(config.REPORTED_RESULTS)
    mt = paper.get("metrics_table", {})
    rows = mt.get("rows", {})
    names = mt.get("metric_names", _METRIC_NAMES)

    if rows:
        st.plotly_chart(charts.metrics_grouped(rows, names), width="stretch")
        st.markdown("")

    cm = paper.get("confusion_matrix", {})
    if cm:
        st.plotly_chart(charts.confusion_matrix(
            cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0), cm.get("tp", 0)),
            width="stretch")
        st.markdown("")

    imp = paper.get("feature_importance", [])
    if imp:
        st.plotly_chart(charts.feature_importance(imp), width="stretch")

    st.markdown(
        glass_card(
            f"<div class='ns-kicker' style='margin-bottom:6px;'>Target definition</div>"
            f"<p style='color:#94a3b8;font-size:0.9rem;'>{esc(paper.get('target', {}).get('method', ''))}</p>",
            padding="1.3rem 1.5rem",
        ),
        unsafe_allow_html=True,
    )


def _shap_section() -> None:
    result = st.session_state.get("last_result")
    if not result or not result.drivers:
        return
    components.section_title("This prediction", "Why your result looks like this")
    st.plotly_chart(
        charts.shap_contrib(
            [d.feature for d in result.drivers],
            [d.value for d in result.drivers],
            [d.contribution for d in result.drivers],
        ),
        width="stretch",
    )


def render() -> None:
    components.render_model_banner()

    if _live_available():
        _live_section()
        st.markdown("")
    else:
        st.markdown(
            glass_card(
                "📦 Live evaluation will appear here once the model is trained "
                "via <code>scripts/train_models.py</code>. The reported paper "
                "numbers below are shown in the meantime.",
                padding="1.1rem 1.3rem",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")
    _shap_section()
    st.markdown("")
    _reported_section()

    components.render_disclaimer()
