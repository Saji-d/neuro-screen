"""Landing page."""

from __future__ import annotations

import streamlit as st

from modules import predictor
from modules.ui import nav, theme
from modules.ui.theme import card, esc, md, stat


def render() -> None:
    ev = predictor.evaluation()
    n = ev.get("dataset", {}).get("respondents", 2236)

    md(f"""
    <div class="hero">
      <div>
        <div class="badge rise"><i></i>Research prototype · AIUB thesis G52</div>
        <h1 class="rise d1">Screen for <span class="grad-text">cognitive strain</span> from sleep and student life.</h1>
        <p class="lead rise d2">Twelve quick questions go through a four-model stack (CatBoost, XGBoost,
        LightGBM and a neural network) trained on {n:,} university students. You get a result, the
        reasons behind it and an honest read on how far to trust it.</p>
      </div>
      <div class="rise d3">{theme.brain_svg()}</div>
    </div>
    """)

    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("Start assessment", type="primary", use_container_width=True):
            nav.goto("assess")
    with c2:
        if st.button("How reliable is it?", use_container_width=True):
            nav.goto("insights")

    md(f'<div class="grid g4" style="margin-top:2.2rem">'
       + stat(f"{n}", "survey respondents", delay=1)
       + stat("12", "self-reported predictors", delay=2)
       + stat("4", "models in the stack", delay=3)
       + stat("5", "cross-validation folds", delay=4)
       + "</div>")

    theme.section("How it works", "No account, no data stored. Your answers are scored in this session only.")
    md('<div class="grid g3">'
       + card("📝", "1 · Answer 12 questions",
              "Age, habits, stress, workload. Nothing about your memory or sleep hours is asked, so the "
              "model cannot simply read the answer back.", 1)
       + card("🧠", "2 · Four models vote",
              "CatBoost, XGBoost, LightGBM and a 128-64-1 neural net each score you. A logistic head "
              "learns how to combine them.", 2)
       + card("🔎", "3 · See the why",
              "SHAP values show which answers pushed the score up or down, next to the cohort average "
              "for comparison.", 3)
       + "</div>")

    theme.section("Two questions, one stack")
    md('<div class="grid g2">'
       + card("😴", "Sleep-risk screen",
              "Does your profile resemble students who sleep under 6 hours or rate their sleep 3/5 or "
              "worse? Trained on all respondents (secondary task).", 1)
       + card("💭", "Cognitive-symptom screen",
              "Among students meeting that sleep-risk rule, does your profile resemble those with elevated "
              "cognitive symptoms (brain fog, forgetfulness, spacing out)? The primary task.", 2)
       + "</div>")

    theme.section("Be clear-eyed about it")
    md(theme.callout(
        "<b>Honest note.</b> An earlier version of this project reported ~95% accuracy because the "
        "symptom questions themselves were fed in as inputs. With that leak removed, cross-validated "
        "results sit close to the majority-class baseline. The app shows those numbers openly on the "
        "<b>Reliability</b> page. Treat output as a conversation starter, not a verdict.", "info", "🔬"))
    st.write("")
    md(theme.disclaimer())
    theme.footer()
