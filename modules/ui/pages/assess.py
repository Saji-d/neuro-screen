"""Assessment wizard: four steps, review, animated run."""

from __future__ import annotations

import time

import streamlit as st

from modules import predictor
from modules.ui import nav, theme
from modules.ui.questions import DEFAULTS, Q, STEPS, answers_to_columns
from modules.ui.theme import esc, md

STEP = "assess_step"
RESULT = "prediction"
ANSWERS = "answers"


def _init() -> None:
    st.session_state.setdefault(STEP, 0)
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(f"q_{k}", v)
    # Re-assigning keeps widget values alive when their step is not on screen.
    for k in DEFAULTS:
        st.session_state[f"q_{k}"] = st.session_state[f"q_{k}"]


def _question(key: str) -> None:
    q = Q[key]
    wid = f"q_{key}"
    md(f'<div class="q-label">{esc(q["label"])}</div>'
       + (f'<div class="q-help">{esc(q["help"])}</div>' if q["help"] else ""))
    kind = q["kind"]
    if kind == "slider":
        st.slider(q["label"], q["lo"], q["hi"], key=wid, label_visibility="collapsed")
    elif kind == "slider_opt":
        st.select_slider(q["label"], options=q["options"], key=wid, label_visibility="collapsed")
    elif kind == "radio":
        st.radio(q["label"], q["options"], key=wid, horizontal=True, label_visibility="collapsed",
                 format_func=lambda o, f=q.get("fmt", {}): f.get(o, o))
    else:
        opts = sorted(predictor.task_meta("cognitive").get("options", {}).get(q["col"], [q["default"]]))
        st.selectbox(q["label"], opts, key=wid, label_visibility="collapsed")


def _progress(step: int) -> None:
    total = len(STEPS) + 1
    bars = "".join(
        f'<div class="s {"done" if i < step else "now" if i == step else ""}"></div>' for i in range(total))
    md(f'<div class="steps">{bars}</div>')


def _run() -> None:
    slot = st.empty()
    stages = ["Encoding your answers", "Scoring with CatBoost, XGBoost and LightGBM",
              "Running the neural network", "Stacking and explaining"]
    answers = answers_to_columns({k: st.session_state[f"q_{k}"] for k in Q})
    result = None
    for i, label in enumerate(stages):
        slot.markdown(f'<div class="loader"><div class="ring"></div><p>{esc(label)}…</p></div>',
                      unsafe_allow_html=True)
        if i == 1:
            result = predictor.predict(answers)
        time.sleep(0.45)
    slot.empty()
    st.session_state[RESULT] = result
    st.session_state[ANSWERS] = {k: st.session_state[f"q_{k}"] for k in Q}
    nav.goto("results")


def render() -> None:
    if not predictor.models_ready():
        md(theme.callout("<b>Models not found.</b> Run <code>python scripts/train_models.py</code> first.",
                         "rose", "🛠️"))
        return
    _init()
    step = st.session_state[STEP]
    md(theme.kicker("Assessment") + f'<div class="step-title rise">'
       + (esc(STEPS[step][0]) if step < len(STEPS) else "Review and run") + "</div>")
    _progress(step)

    if step < len(STEPS):
        md(f'<p class="lead rise" style="font-size:.98rem">{esc(STEPS[step][1])}</p>')
        for key in STEPS[step][2]:
            _question(key)
    else:
        rows = "".join(
            f'<tr><td>{esc(Q[k]["label"])}</td><td><b>{esc(Q[k].get("fmt", {}).get(st.session_state[f"q_{k}"], st.session_state[f"q_{k}"]))}</b></td></tr>'
            for k in Q)
        md(f'<div class="card rise"><table class="tbl"><tr><th>Question</th><th>Your answer</th></tr>{rows}</table></div>')
        st.write("")
        md(theme.disclaimer())

    st.write("")
    back, _, fwd = st.columns([1, 3, 1.4])
    with back:
        if step > 0 and st.button("← Back", use_container_width=True):
            st.session_state[STEP] = step - 1
            st.rerun()
    with fwd:
        if step < len(STEPS):
            if st.button("Continue →", type="primary", use_container_width=True):
                st.session_state[STEP] = step + 1
                st.rerun()
        elif st.button("Run analysis ✦", type="primary", use_container_width=True):
            _run()
