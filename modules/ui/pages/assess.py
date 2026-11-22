"""Assessment wizard: four steps and a review, with a fixed action bar and a live side panel."""

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

STAGES = ["Encoding your answers", "Scoring: CatBoost, XGBoost, LightGBM",
          "Running the neural network", "Blending with the logistic head", "Explaining with SHAP"]


def _init() -> None:
    st.session_state.setdefault(STEP, 0)
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(f"q_{k}", v)
    # Re-assigning keeps widget values alive when their step is not on screen.
    for k in DEFAULTS:
        st.session_state[f"q_{k}"] = st.session_state[f"q_{k}"]


def _show(key: str) -> str:
    v = st.session_state[f"q_{key}"]
    return str(Q[key].get("fmt", {}).get(v, v))


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
    bars = "".join(
        f'<div class="s {"done" if i < step else "now" if i == step else ""}"></div>'
        for i in range(len(STEPS) + 1))
    md(f'<div class="steps">{bars}</div>')


def _profile_panel(step: int) -> str:
    current = set(STEPS[step][2]) if step < len(STEPS) else set()
    chips = "".join(
        f'<span class="chip{" now" if k in current else ""}">{esc(Q[k]["label"])}: <b>{esc(_show(k))}</b></span>'
        for k in Q)
    return ('<div class="card flat"><div class="panel-title">Your profile, live</div>'
            f'<div class="chips">{chips}</div>'
            '<p style="margin-top:1rem;font-size:.82rem">Highlighted chips are the ones on this step. '
            "Nothing leaves this session.</p></div>")


def _run_panel(done: int) -> str:
    items = "".join(
        f'<li class="{"done" if i < done else "cur" if i == done else ""}"><i>{"✓" if i < done else ""}</i>{esc(s)}</li>'
        for i, s in enumerate(STAGES))
    return (f'<div class="card flat run over"><div><div class="ring"></div>'
            f'<div class="panel-title">Analysing your answers</div><ul class="stg">{items}</ul></div></div>')


def _run(slot) -> None:
    answers = answers_to_columns({k: st.session_state[f"q_{k}"] for k in Q})
    result = None
    for i in range(len(STAGES)):
        slot.markdown(_run_panel(i), unsafe_allow_html=True)
        if i == 1:
            result = predictor.predict(answers)
        time.sleep(0.4)
    slot.markdown(_run_panel(len(STAGES)), unsafe_allow_html=True)
    time.sleep(0.25)
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
    last = len(STEPS)
    title = STEPS[step][0] if step < last else "Review and run"

    main, side = st.columns([1.6, 1], gap="large")
    with side:
        with st.container(key="side"):
            slot = st.empty()
            slot.markdown(_profile_panel(step), unsafe_allow_html=True)
    with main:
        md(theme.kicker("Assessment") + f'<div class="step-title">{esc(title)}</div>')
        _progress(step)
        if step < last:
            md(f'<p class="lead" style="font-size:.98rem">{esc(STEPS[step][1])}</p>')
            for key in STEPS[step][2]:
                _question(key)
        else:
            rows = "".join(f"<tr><td>{esc(Q[k]['label'])}</td><td><b>{esc(_show(k))}</b></td></tr>" for k in Q)
            md('<div class="card flat"><div class="scroll-x"><table class="tbl"><tr><th>Question</th>'
               f"<th>Your answer</th></tr>{rows}</table></div></div>")
            st.write("")
            md(theme.disclaimer())
        st.write("")
        st.write("")
        st.write("")

    # Fixed bottom bar: Back / Continue never move, whatever the step's height.
    with st.container(key="actionbar"):
        back, mid, fwd = st.columns([1, 2.4, 1.2])
        with back:
            if step > 0 and st.button("← Back", use_container_width=True, key="btn_back"):
                st.session_state[STEP] = step - 1
                st.rerun()
        with mid:
            md(f'<div class="meta">Step <b>{min(step + 1, last + 1)}</b> of {last + 1} · {esc(title)}</div>')
        with fwd:
            if step < last:
                if st.button("Continue →", type="primary", use_container_width=True, key="btn_next"):
                    st.session_state[STEP] = step + 1
                    st.rerun()
            elif st.button("Run analysis ✦", type="primary", use_container_width=True, key="btn_run"):
                _run(slot)
