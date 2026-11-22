"""Results page: animated gauges, model breakdown, drivers, tips."""

from __future__ import annotations

import streamlit as st

from modules import predictor
from modules.predictor import TaskResult
from modules.ui import nav, theme
from modules.ui.theme import esc, md

COLORS = {"above": "#fb7185", "inline": "#fbbf24", "below": "#34d399"}
WORDS = {"above": "Above cohort average", "inline": "In line with cohort average",
         "below": "Below cohort average"}


def _gauge_card(r: TaskResult, title: str, blurb: str, delay: int) -> str:
    color = COLORS[r.band]
    sign = "+" if r.delta_pp >= 0 else "−"
    return (f'<div class="card rise d{delay}" style="text-align:center">'
            f'<div class="kicker">{esc(title)}</div>'
            f"{theme.gauge(r.probability, r.base_rate, color, 'model estimate')}"
            f'<span class="verdict v-{r.band}">{WORDS[r.band]}</span>'
            f'<p style="margin-top:.9rem;font-size:.85rem">Cohort average {r.base_rate * 100:.0f}% '
            f"(white tick) · you are {sign}{abs(r.delta_pp):.1f} pts</p>"
            f'<p style="margin-top:.4rem;font-size:.8rem">{esc(blurb)}</p></div>')


def _tips(a: dict) -> list[str]:
    tips = []
    if a["devices"] in ("Sometimes", "Always"):
        tips.append("Try a screen-free wind-down for 30–60 minutes before bed.")
    if a["caffeine"] == "5 or more":
        tips.append("Move caffeine earlier in the day; after midday it can delay sleep.")
    if a["wake"] in ("Sometimes", "Always"):
        tips.append("Keep a regular sleep and wake time, and keep the room dark and cool.")
    if a["stress"] in ("Often", "Always") or a["load"] == "Always":
        tips.append("Schedule short recovery breaks; slow breathing or a walk helps between study blocks.")
    if a["deadlines"] == "5 or more" or a["fatigue"] in ("Often", "Always"):
        tips.append("Break big deadlines into smaller steps and tell a lecturer early if you are slipping.")
    tips.append("If memory or focus problems persist, talk to a doctor or your university counselling service.")
    return tips


def render() -> None:
    pred = st.session_state.get("prediction")
    if pred is None:
        md(theme.kicker("Results") + '<div class="step-title">Nothing to show yet</div>'
           '<p class="lead">Complete the short assessment to see your result.</p>')
        if st.button("Start assessment", type="primary"):
            nav.goto("assess")
        return

    cog, slp = pred.cognitive, pred.sleep
    md(theme.kicker("Your result") + '<div class="step-title rise">Here is what the stack sees</div>')
    md('<div class="grid g2">'
       + _gauge_card(cog, "Cognitive-symptom screen",
                     "Trained on students who meet the sleep-risk rule; the primary task.", 1)
       + _gauge_card(slp, "Sleep-risk screen",
                     "Does your profile resemble students under 6 h or with sleep rated ≤ 3/5?", 2)
       + "</div>")

    ev = predictor.evaluation().get("tasks", {}).get("cognitive", {}).get("cv", {})
    auc = ev.get("Stack", {}).get("auc", {}).get("mean")
    note = (f" Cross-validated AUC for this screen is {auc:.2f} (0.50 = chance)." if auc is not None else "")
    st.write("")
    md(theme.callout(
        "<b>Read this as a comparison, not a verdict.</b> The model's estimates stay close to the "
        "cohort average for almost everyone, because these twelve answers carry little signal about "
        "the symptom score." + note + " See the <b>Reliability</b> page for the full numbers.",
        "warn", "🧭"))

    theme.section("Why the score moved", "SHAP contributions from the CatBoost model for the cognitive screen. "
                  "Red raised the estimate, green lowered it.")
    top = cog.drivers[:6]
    mx = max(abs(d[2]) for d in top) or 1
    rows = [(f"{name}", v, f"{val}") for name, val, v in top]
    md('<div class="card rise">'
       + theme.bar_rows([(n, v, shown) for n, v, shown in
                         [(name, sv, f"{sv:+.2f}  ·  {val}") for name, val, sv in top]], maxv=mx, signed=True)
       + "</div>")

    theme.section("Model breakdown", "Each component's estimate for the cognitive screen, then the stack.")
    parts = cog.parts
    md('<div class="card rise">'
       + theme.bar_rows([(k, parts[k], f"{parts[k] * 100:.1f}%")
                         for k in ("CatBoost", "XGBoost", "LightGBM", "ANN", "Stack")])
       + "</div>")

    theme.section("Small things worth trying")
    md('<div class="card rise"><ul style="margin:0;padding-left:1.1rem;line-height:1.9;color:#c9d1ea">'
       + "".join(f"<li>{esc(t)}</li>" for t in _tips(st.session_state.get("answers", {})))
       + "</ul></div>")
    st.write("")
    md(theme.disclaimer())

    st.write("")
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("Retake", use_container_width=True):
            st.session_state["assess_step"] = 0
            nav.goto("assess")
    with c2:
        if st.button("Reliability →", use_container_width=True):
            nav.goto("insights")
    theme.footer()
