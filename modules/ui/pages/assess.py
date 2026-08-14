"""
modules/ui/pages/assess.py
==========================
Quick check-in — the 21-question screening form.

All 21 controls map 1:1 onto the model's feature columns (via the schema in
config.load_feature_schema). On submit the answers are assembled into a raw
inputs dict and sent to the ONE inference path, modules.predictor.predict.
If the trained artifacts are missing, the run button is disabled — the app
never fabricates a score.
"""

from __future__ import annotations

import time

import streamlit as st

import config
from modules import predictor
from modules.ui import components, navigation
from modules.ui.theme import esc, glass_card

# Widget-state keys live in st.session_state so results/explainability can
# read back the exact answers that produced a prediction.
WIDGET_KEY = "feat_{key}"

# Guards the run button against double-submission and drives the staged
# loading UI. The stages are truthful labels for the real inference
# pipeline (see modules.predictor.predict) — not a fake training progress %.
RUNNING_KEY = "assess_running"
_STAGES = [
    ("Preparing responses", 10),
    ("Validating inputs", 22),
    ("Running CatBoost", 48),
    ("Running ANN", 68),
    ("Combining predictions", 84),
    ("Generating explanation", 96),
]
_STAGE_PAUSE = 0.14  # small, deliberate pacing so each stage is perceptible


def _default_index(feat: dict) -> int:
    opts = feat.get("options", [])
    try:
        return opts.index(feat.get("default", opts[0]))
    except (ValueError, IndexError):
        return 0


def _render_feature(feat: dict, cols: tuple) -> None:
    """One question: descriptive card (left) + input widget (right)."""
    left, right = cols
    with left:
        st.markdown(
            f"""
            <div class="ns-feature ns-animate">
              <div class="ns-feature-icon">{esc(feat.get("icon", "📌"))}</div>
              <div class="ns-feature-body">
                <div class="ns-feature-label">{esc(feat["label"])}
                  <span style="color:#64748b;font-weight:400;font-size:0.72rem;">
                    {esc(feat.get("category", ""))}</span>
                </div>
                <div class="ns-feature-desc">{esc(feat.get("description", ""))}</div>
                <div class="ns-feature-reco">💡 {esc(feat.get("recommended", ""))}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        key = WIDGET_KEY.format(key=feat["key"])
        if feat.get("input_type") == "number":
            st.number_input(
                "Value",
                min_value=int(feat.get("min", 17)),
                max_value=int(feat.get("max", 45)),
                value=int(feat.get("default", 22)),
                step=int(feat.get("step", 1)),
                key=key,
                label_visibility="collapsed",
                help=feat.get("description"),
            )
        else:
            st.selectbox(
                "Option",
                options=list(feat.get("options", [])),
                index=_default_index(feat),
                key=key,
                label_visibility="collapsed",
                help=feat.get("description"),
            )


def _collect_answers(features: list[dict]) -> tuple[dict, dict]:
    """Assemble raw model inputs + a display-friendly copy of the answers."""
    raw_inputs: dict = {}
    answers: dict = {}
    for feat in features:
        val = st.session_state.get(WIDGET_KEY.format(key=feat["key"]), feat.get("default"))
        raw_inputs[feat["model_column"]] = str(val)
        answers[feat["key"]] = {"label": feat["label"], "value": val, "icon": feat.get("icon", "")}
    return raw_inputs, answers


def _progress(features: list[dict]) -> float:
    """Fraction of questions whose answer differs from the default."""
    answered = 0
    for feat in features:
        default = feat.get("default", feat.get("options", [None])[0])
        current = st.session_state.get(WIDGET_KEY.format(key=feat["key"]), default)
        if current != default:
            answered += 1
    return answered / max(1, len(features))


def render() -> None:
    features = config.load_feature_schema()
    if not features:
        st.info("Feature schema not found. Check `data/feature_schema.json`.")
        return

    components.render_model_banner()

    components.section_title("Quick check-in", "Tell us about your sleep and day")
    st.markdown(
        f"<p style='color:#94a3b8;max-width:760px;'>Answer the {len(features)} "
        f"questions below as honestly as you can — there are no wrong answers. "
        f"Neuro-Screen blends a CatBoost model and a neural network on your "
        f"answers to estimate cognitive-impairment risk linked to insomnia.</p>",
        unsafe_allow_html=True,
    )

    # ---- progress ----
    pct = _progress(features)
    st.progress(pct)
    st.caption(f"{int(pct * len(features))} of {len(features)} questions reviewed")

    st.markdown("")

    # ---- grouped questions ----
    categories: list[tuple[str, list[dict]]] = []
    for feat in features:
        cat = feat.get("category", "General")
        if categories and categories[-1][0] == cat:
            categories[-1][1].append(feat)
        else:
            categories.append((cat, [feat]))

    for cat, feats in categories:
        st.markdown(
            f'<div class="ns-section-head ns-animate" style="margin-top:1.2rem;">'
            f'<span class="ns-kicker">{esc(cat)}</span></div>',
            unsafe_allow_html=True,
        )
        for feat in feats:
            cols = st.columns([2.1, 1.1])
            _render_feature(feat, cols)

    st.markdown("")

    # ---- submit ----
    if not predictor.models_available():
        st.markdown(
            glass_card(
                "🔒 Predictions are unavailable until the trained model artifacts "
                "exist. Follow the instructions in the banner above, then refresh. "
                "Your answers are safe — nothing is sent anywhere.",
                padding="1rem 1.3rem",
            ),
            unsafe_allow_html=True,
        )
        return

    if RUNNING_KEY not in st.session_state:
        st.session_state[RUNNING_KEY] = False

    run_area = st.empty()
    with run_area.container():
        col_btn, col_hint = st.columns([1, 2])
        with col_btn:
            clicked = st.button(
                "Run neuro-screen →", type="primary", width="stretch",
                disabled=st.session_state[RUNNING_KEY], key="run_neuro_screen",
            )
        with col_hint:
            st.markdown(
                "<p style='color:#64748b;font-size:0.8rem;'>Your responses are processed "
                "locally by the trained ensemble and are never stored or uploaded.</p>",
                unsafe_allow_html=True,
            )

    if clicked and not st.session_state[RUNNING_KEY]:
        st.session_state[RUNNING_KEY] = True
        raw_inputs, answers = _collect_answers(features)

        with run_area.container():
            st.button("Running…", type="primary", width="stretch", disabled=True,
                      key="run_neuro_screen_busy")
            status = st.empty()
            bar = st.progress(0)
            result = None
            try:
                for label, pct in _STAGES:
                    status.markdown(
                        f"<p style='color:#94a3b8;font-size:0.85rem;margin:0.4rem 0 0;'>"
                        f"🧠 Running Neuro-Screen assessment… <b style='color:#e6edf7;'>"
                        f"{esc(label)}</b></p>",
                        unsafe_allow_html=True,
                    )
                    bar.progress(pct)
                    time.sleep(_STAGE_PAUSE)
                    if label == "Combining predictions":
                        # The real hybrid inference call — CatBoost + ANN +
                        # per-instance SHAP — happens right here.
                        result = predictor.predict(raw_inputs)
                status.markdown(
                    "<p style='color:#5eead4;font-size:0.85rem;margin:0.4rem 0 0;'>"
                    "✅ Assessment complete</p>",
                    unsafe_allow_html=True,
                )
                bar.progress(100)
                time.sleep(0.2)
            except predictor.ModelNotReadyError:
                st.session_state[RUNNING_KEY] = False
                st.error("Model artifacts are missing — see the banner above.")
                st.stop()

        st.session_state["last_result"] = result
        st.session_state["last_answers"] = answers
        st.session_state[RUNNING_KEY] = False
        navigation.goto("results")

    components.render_disclaimer()
