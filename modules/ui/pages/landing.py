"""
modules/ui/pages/landing.py
===========================
Landing page — hero, paper-reported headline metrics, abstract, methodology
overview, and authorship. No prediction happens here.
"""

from __future__ import annotations

import streamlit as st

import config
from modules import predictor
from modules.ui import components
from modules.ui.theme import esc, glass_card

# Paper-reported hybrid numbers (Table 2 / Abstract of the ICCA 2026 paper).
_HYBRID_ROW = [0.9520, 0.9440, 0.9610, 0.9524, 0.9820]
_METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]


def _cta_block() -> None:
    c1, c2 = st.columns([1, 1])
    with c1:
        components.nav_button("Start a quick check-in", "assessment",
                              "cta_assessment", variant="primary")
    with c2:
        components.nav_button("See how the model performs", "explainability",
                              "cta_explainability", variant="ghost")


def render() -> None:
    components.render_model_banner()

    st.markdown(
        f"""
        <div class="ns-hero ns-animate">
          <span class="ns-eyebrow">🎓&nbsp; AIUB · CSC 4298 · Thesis/Project</span>
          <h1 class="ns-hero-title">Neuro-Screen<span class="ns-accent">.</span></h1>
          <p class="ns-hero-sub">{esc(config.BRAND["subtitle"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _cta_block()

    # ---- headline metrics (paper-reported hybrid model) ----
    st.markdown('<div class="ns-section-head ns-animate"><span class="ns-kicker">'
                'Paper-reported performance · Hybrid ensemble</span></div>',
                unsafe_allow_html=True)
    cols = st.columns(5, gap="medium")
    for col, (name, value) in zip(cols, zip(_METRIC_NAMES, _HYBRID_ROW)):
        with col:
            components.stat_card(
                icon=_METRIC_ICON[name],
                label=name,
                value=f"{value*100:.1f}%",
                sub=None,
            )

    st.markdown("")

    # ---- abstract + dataset ----
    # Short, plain-language summary for the homepage card — the full paper
    # abstract (data/reported_results.json) stays intact as the authoritative
    # source and is unaffected by this; it's just too long for people to
    # actually read here.
    _SUMMARY = (
        "Neuro-Screen is a hybrid CatBoost + ANN ensemble that screens "
        "university students for cognitive-impairment risk linked to "
        "insomnia, using self-reported sleep and lifestyle answers instead "
        "of clinical testing — so at-risk students can get support early."
    )
    paper = config.load_json(config.REPORTED_RESULTS)
    ds = paper.get("dataset", {})

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(
            glass_card(
                f'<div class="ns-kicker" style="margin-bottom:8px;">About the thesis</div>'
                f'<div style="color:#cbd5e1;line-height:1.65;font-size:0.95rem;">{esc(_SUMMARY)}</div>',
            ),
            unsafe_allow_html=True,
        )
    with right:
        # Prefer the live trained-metadata when models are available; fall back to paper.
        meta = None
        if predictor.models_available():
            try:
                meta = config.load_json(config.ARTIFACTS["meta"])
            except Exception:
                meta = None
        if meta:
            md = meta.get("dataset", {})
            insom_rate = md.get("insomniac_rate", 0) * 100
            pos_rate = md.get("impaired_rate", 0) * 100
            trained_date = str(meta.get("trained_at", "")).split("T")[0]
            st.markdown(
                glass_card(
                    f'<div class="ns-kicker" style="margin-bottom:12px;">The dataset</div>'
                    f'<div class="ns-stat-value" style="font-size:2rem;">{esc(str(md.get("rows", "")))}</div>'
                    f'<div class="ns-stat-label">Self-reported surveys collected</div>'
                    f'<div style="margin-top:10px;color:#94a3b8;font-size:0.85rem;">'
                    f'{esc(str(md.get("train", "")))} train · {esc(str(md.get("test", "")))} test '
                    f'(stratified 80/20, seed 42)</div>'
                    f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:4px;">'
                    f'{esc(str(md.get("features", "")))} features (20 categorical · 1 numeric) · '
                    f'{esc(f"trained {trained_date}")}</div>'
                    f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:4px;">'
                    f'Insomniac students: {esc(f"{insom_rate:.1f}%")} · '
                    f'Positive class: {esc(f"{pos_rate:.1f}%")}</div>',
                ),
                unsafe_allow_html=True,
            )
        else:
            insom_rate = ds.get("insomniac_rate", 0) * 100
            pos_rate = ds.get("impairment_positive_rate", 0) * 100
            st.markdown(
                glass_card(
                    f'<div class="ns-kicker" style="margin-bottom:12px;">The dataset</div>'
                    f'<div class="ns-stat-value" style="font-size:2rem;">{esc(str(ds.get("responses", "")))}</div>'
                    f'<div class="ns-stat-label">Self-reported surveys collected (paper)</div>'
                    f'<div style="margin-top:10px;color:#94a3b8;font-size:0.85rem;">'
                    f'{esc(str(ds.get("train", "")))} train · {esc(str(ds.get("test", "")))} test '
                    f'(stratified 80/20, seed 42)</div>'
                    f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:4px;">'
                    f'{esc(str(ds.get("age_range", "")))} years · {esc(str(ds.get("population", "")))}</div>'
                    f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:4px;">'
                    f'Insomniac students: {esc(f"{insom_rate:.1f}%")} · '
                    f'Positive class: {esc(f"{pos_rate:.1f}%")}</div>',
                ),
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ---- how it works ----
    components.section_title("Methodology", "How Neuro-Screen works")
    steps = [
        ("1", "🧭", "Quick check-in",
         "Answer 21 short questions about sleep, lifestyle and everyday cognitive "
         "symptoms — no accounts, no invasive tests."),
        ("2", "🤖", "Hybrid inference",
         "A CatBoost gradient-boosting model and a 3-layer neural network each "
         "score your responses; their probabilities are blended into one "
         "confidence estimate (P_hybrid = (P_catboost + P_ann) / 2)."),
        ("3", "🔬", "Explainable result",
         "You get a risk level, the contributing factors, and practical "
         "recommendations — not just a number."),
    ]
    s1, s2, s3 = st.columns(3, gap="medium")
    for col, (num, icon, title, body) in zip((s1, s2, s3), steps):
        with col:
            st.markdown(
                glass_card(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                    f'<span class="ns-num-badge">{num}</span><span style="font-size:1.4rem;">{icon}</span>'
                    f'<div class="ns-feature-label">{title}</div></div>'
                    f'<div style="color:#94a3b8;font-size:0.88rem;line-height:1.55;">{body}</div>',
                    padding="1.3rem 1.3rem",
                ),
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ---- authors / thesis group ----
    components.section_title("Team", "The people behind Neuro-Screen")
    a1, a2 = st.columns(2, gap="large")
    with a1:
        author_cards = "".join(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
            f'<span style="color:#e6edf7;font-size:0.92rem;">{esc(a["name"])}</span>'
            f'<span style="color:#64748b;font-size:0.8rem;">{esc(a["id"])}</span></div>'
            for a in config.BRAND["authors"]
        )
        st.markdown(
            glass_card(
                f'<div class="ns-kicker" style="margin-bottom:6px;">Authors</div>{author_cards}',
                padding="1.3rem 1.4rem",
            ),
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            glass_card(
                f'<div class="ns-kicker" style="margin-bottom:8px;">👥 Thesis Group</div>'
                f'<div class="ns-stat-value" style="font-size:2rem;">'
                f'{esc(config.BRAND["thesis_group"])}</div>'
                f'<div class="ns-stat-label">Thesis group</div>'
                f'<div style="color:#64748b;font-size:0.78rem;margin-top:14px;">'
                f'{esc(config.BRAND["course"])} · {esc(config.BRAND["semester"])}</div>',
                padding="1.3rem 1.4rem",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")
    components.render_disclaimer()

    if not predictor.models_available():
        components.nav_button("View reported results", "explainability",
                              "cta_reported", variant="ghost")


_METRIC_ICON = {
    "Accuracy": "🎯",
    "Precision": "⚡",
    "Recall": "📡",
    "F1-Score": "🧪",
    "ROC-AUC": "📈",
}
