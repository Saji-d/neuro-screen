"""Landing page."""

from __future__ import annotations

import streamlit as st

from modules import predictor
from modules.ui import nav, theme
from modules.ui.questions import Q
from modules.ui.theme import esc, expand, md, stat

FAQ = [
    ("Is my data stored?",
     "No. Your answers live only in your browser session while the app is open. Nothing is written to a "
     "database or sent anywhere else."),
    ("Why are results always close to the average?",
     "Because the twelve questions carry very little information about the symptom score. When the symptom "
     "questions themselves were used as inputs, accuracy looked like 95%, but that was the label leaking "
     "into the model. Without the leak, cross-validated accuracy matches simply guessing the most common "
     "answer. The Reliability page shows the exact numbers."),
    ("What is the sleep-risk rule?",
     "In the survey, a respondent meets the rule if they sleep under 6 hours a night or rate their sleep "
     "quality 3 out of 5 or worse. About three quarters of respondents meet it."),
    ("Can this diagnose cognitive impairment?",
     "No. The label is built from seven self-reported symptom questions, not a clinical test, and the model "
     "has not been clinically validated. Use it to start a conversation, not to draw conclusions."),
    ("Who built it?",
     "A thesis project by Group G52 at the American International University-Bangladesh (AIUB), "
     "Department of Computer Science."),
]


def _diagram() -> str:
    def box(x, y, w, h, t, sub="", glow=False):
        cls = "glow" if glow else "box"
        return (f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="14"/>'
                f'<text x="{x + w / 2}" y="{y + h / 2 + (-2 if sub else 4)}" text-anchor="middle">{t}</text>'
                + (f'<text class="sm" x="{x + w / 2}" y="{y + h / 2 + 14}" text-anchor="middle">{sub}</text>' if sub else ""))

    models = [("CatBoost", "native categories"), ("XGBoost", "gradient boosting"),
              ("LightGBM", "leaf-wise trees"), ("ANN", "128 · 64 · 1")]
    parts = [box(10, 120, 130, 80, "12 answers", "survey inputs")]
    flows = []
    for i, (m, sub) in enumerate(models):
        y = 20 + i * 90
        parts.append(box(250, y, 150, 62, m, sub))
        flows.append(f'<path class="flow" d="M140 160 C195 160 195 {y + 31} 250 {y + 31}"/>')
        flows.append(f'<path class="flow" d="M400 {y + 31} C455 {y + 31} 455 160 510 160" style="animation-delay:{i * .3}s"/>')
    parts.append(box(510, 120, 150, 80, "Logistic head", "learns the blend", glow=True))
    for i, (t, sub) in enumerate((("Cognitive screen", "primary"), ("Sleep-risk screen", "secondary"))):
        y = 60 + i * 120
        parts.append(box(750, y, 170, 70, t, sub))
        flows.append(f'<path class="flow" d="M660 160 C705 160 705 {y + 35} 750 {y + 35}"/>')
    return ('<svg class="diagram" viewBox="0 0 930 380" role="img" aria-label="Pipeline: 12 answers go to four models, '
            'a logistic head blends them, producing two screens">'
            '<defs><linearGradient id="dg" x1="0" x2="1"><stop offset="0" stop-color="#8b7cf6"/>'
            '<stop offset="1" stop-color="#22d3ee"/></linearGradient></defs>'
            + "".join(flows) + "".join(parts) + "</svg>")


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

    c1, c2, _ = st.columns([1.1, 1.1, 2.6])
    with c1:
        if st.button("Start assessment", type="primary", use_container_width=True):
            nav.goto("assess")
    with c2:
        if st.button("How reliable is it?", use_container_width=True):
            nav.goto("insights")

    md('<div class="grid g4" style="margin-top:2rem">'
       + stat(f"{n:,}", "survey respondents", delay=1) + stat("12", "self-reported predictors", delay=2)
       + stat("4", "models in the stack", delay=3) + stat("5", "cross-validation folds", delay=4) + "</div>")

    chips = [esc(Q[k]["label"]) for k in Q]
    md('<div class="marq" aria-label="Topics the assessment covers"><div class="tr">'
       + "".join(f"<span>{c}</span>" for c in chips * 2) + "</div></div>")

    theme.section("How it works", "Tap a card to read more. Nothing you enter is stored.")
    md('<div class="grid g3">'
       + expand("📝", "1 · Answer 12 questions", "Age, habits, stress and workload; about two minutes.",
                "The questions come from a survey of Bangladeshi university students. <b>Nothing about your "
                "memory or sleep hours is asked</b>, because those items define the labels. Using them as "
                "inputs would let the model read the answer back.")
       + expand("🧠", "2 · Four models vote", "CatBoost, XGBoost, LightGBM and a neural network each score you.",
                "CatBoost reads categories natively; XGBoost and LightGBM work on a one-hot encoding; the ANN is "
                "a 128-64-1 network with dropout. A logistic regression learns how to blend their scores from "
                "out-of-fold predictions.")
       + expand("🔎", "3 · See the why", "SHAP shows which answers pushed the score up or down.",
                "Each answer gets a signed contribution from the CatBoost model. Your result is shown next to "
                "the cohort average, so you can see how far you sit from a typical respondent.")
       + "</div>")

    theme.section("Inside the stack", "Every answer set takes this exact path. The dashes show the flow.")
    md(f'<div class="card flat rv"><div class="scroll-x">{_diagram()}</div></div>')

    theme.section("Two screens, one stack")
    md('<div class="grid g2">'
       + expand("💭", "Cognitive-symptom screen", "Primary task. Trained on students who meet the sleep-risk rule.",
                "Label: a symptom score of 10 or more out of 17 (the 72nd percentile of the training split). "
                "The score sums seven self-reported items: mental stamina, spacing out, audio lag, forgetfulness, "
                "reliance on reminders, brain fog and decision difficulty.")
       + expand("😴", "Sleep-risk screen", "Secondary task. Trained on all respondents.",
                "Label: under 6 hours of sleep a night, or sleep quality rated 3 out of 5 or worse. Around "
                "three in four respondents meet it, so the model's baseline estimate is high for everyone.")
       + "</div>")

    theme.section("Be clear-eyed about it")
    md(theme.callout(
        "<b>Honest note.</b> An earlier version of this project reported ~95% accuracy because the symptom "
        "questions were fed in as inputs. With that leak removed, cross-validated results sit close to the "
        "majority-class baseline. The <b>Reliability</b> page shows those numbers openly. Treat the output as "
        "a conversation starter, not a verdict.", "info", "🔬"))

    theme.section("Questions people ask")
    md('<div class="grid" style="grid-template-columns:1fr;max-width:820px">'
       + "".join(f'<details class="card rv flat"><summary style="padding:.2rem 2.2rem .2rem 0"><h4 style="margin:0">'
                 f'{esc(q)}</h4></summary><div class="more" style="padding:.8rem 0 0">{esc(a)}</div></details>'
                 for q, a in FAQ) + "</div>")

    st.write("")
    md('<div class="card rv" style="text-align:center;padding:2.2rem 1.4rem">'
       '<div class="ns-display" style="font-size:1.6rem;font-weight:700">Ready? It takes about two minutes.</div>'
       '<p style="margin:.6rem auto 0;max-width:460px">No account and no data stored.</p></div>')
    _, mid, _ = st.columns([1.3, 1, 1.3])
    with mid:
        st.write("")
        if st.button("Start assessment →", type="primary", use_container_width=True, key="cta_bottom"):
            nav.goto("assess")
    st.write("")
    md(theme.disclaimer())
    theme.footer()
