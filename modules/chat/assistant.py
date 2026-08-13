"""
modules/chat/assistant.py
=========================
The conversational check-in.

The assistant collects the same 21 answers as the quick check-in, one question
at a time, by matching free-text replies to the question's allowed options.
Once every question is answered it calls the SAME single inference pipeline
(modules.predictor.predict) and stores the result for the Results page —
there is no separate mock logic here.

Safety: a small crisis detector intercepts statements about self-harm and
redirects to the AIUB Psychological Support Center. It is not a substitute
for professional help.
"""

from __future__ import annotations

import re

import streamlit as st

import config
from modules import predictor
from modules.ui import components
from modules.ui.theme import esc

# ------------------------------------------------------------------
# Session state keys
# ------------------------------------------------------------------
MSG = "chat_messages"
RESP = "chat_responses"
STEP = "chat_step"
DONE = "chat_done"

CRISIS_PATTERN = re.compile(
    r"\b(suicid\w*|self[- ]harm|kill\s+myself|end\s+my\s+life|"
    r"don'?t\s+want\s+to\s+live|want\s+to\s+die|hurt\s+myself|"
    r"no\s+point\s+in\s+living)\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _match_option(answer: str, options: list[str]):
    """Best-effort free-text -> option matching."""
    a = _norm(answer)
    if not a:
        return None
    for opt in options:
        o = _norm(opt)
        if o == a or o in a or a in o:
            return opt
    return None


def _reset() -> None:
    st.session_state[MSG] = []
    st.session_state[RESP] = {}
    st.session_state[STEP] = 0
    st.session_state[DONE] = False


def _append(role: str, content: str) -> None:
    st.session_state[MSG].append({"role": role, "content": content})


def _features() -> list[dict]:
    return config.load_feature_schema()


def _question(feat: dict) -> str:
    icon = feat.get("icon", "📌")
    opts = feat.get("options")
    if opts:
        opts_txt = "`" + "`, `".join(opts) + "`"
    else:
        opts_txt = f"a number between {feat.get('min')} and {feat.get('max')}"
    return (
        f"{icon} **{feat['label']}**\n\n{feat.get('description', '')}\n\n"
        f"Reply with: {opts_txt}"
    )


def _handle_answer(features: list[dict], user_text: str) -> None:
    """Validate and store the answer to the current question."""
    feat = features[st.session_state[STEP]]
    options = feat.get("options")

    if feat.get("input_type") == "number":
        try:
            value = int(float(user_text.strip()))
        except ValueError:
            _append("assistant", "That doesn't look like a number — could you "
                                 "type just the number of years? 🙂")
            return
        if value < int(feat.get("min", 0)) or value > int(feat.get("max", 120)):
            _append("assistant", f"Please enter an age between "
                                 f"{feat.get('min')} and {feat.get('max')}.")
            return
        st.session_state[RESP][feat["key"]] = str(value)
    else:
        match = _match_option(user_text, options or [])
        if match is None:
            _append("assistant", "I didn't catch that — pick one of: "
                                 + ", ".join(f"`{o}`" for o in (options or [])) + " 🙂")
            return
        st.session_state[RESP][feat["key"]] = match

    st.session_state[STEP] += 1
    if st.session_state[STEP] >= len(features):
        st.session_state[DONE] = True


def _summary_and_predict(features: list[dict]) -> None:
    raw_inputs = {f["model_column"]: st.session_state[RESP][f["key"]] for f in features}
    answers = {f["key"]: {"label": f["label"], "value": st.session_state[RESP][f["key"]],
                          "icon": f.get("icon", "")} for f in features}

    if not predictor.models_available():
        _append("assistant",
                "Thank you — I've collected all 21 answers. The trained model "
                "artifacts aren't loaded on this machine yet, so I can't run the "
                "inference. Follow the instructions on the Home page to train the "
                "model, then come back and type **restart** to re-run your check-in.")
        return

    try:
        result = predictor.predict(raw_inputs)
    except predictor.ModelNotReadyError:
        _append("assistant", "The model isn't available right now — please check "
                             "the status banner on the Home page.")
        return

    st.session_state["last_result"] = result
    st.session_state["last_answers"] = answers

    tier = result.severity
    _append(
        "assistant",
        f"All done! Here is your neuro-screen:\n\n"
        f"- **Prediction:** {result.prediction_label} (`{result.risk_score}` / 100, {tier})\n"
        f"- **Blended probability:** {result.proba_impaired*100:.1f}%\n"
        f"- **Model confidence:** {result.confidence}%\n\n"
        f"Open the **Results** page (🧭 → Results) for your contributing factors "
        f"and recommendations. Type **restart** to go again.",
    )


def _handle_command(user_text: str, features: list[dict]) -> bool:
    t = _norm(user_text)
    if t in ("restart", "start over", "reset", "start again", "new"):
        _reset()
        _append("assistant", "Got it — let's start over. " + _question(features[0]))
        return True
    if t in ("help", "what can you do", "hi", "hello", "hey"):
        _append("assistant",
                "I run your **cognitive-impairment check-in** by chatting. "
                "I'll ask you about your sleep, lifestyle and a few everyday "
                "cognitive symptoms (21 short questions), then run the same "
                "CatBoost + ANN hybrid model as the quick check-in. "
                "Type **answers** to see what I've collected so far, or "
                "**restart** to begin again.")
        return True
    if t in ("answers", "what have you got", "status"):
        collected = {f["key"]: v for f in features if f["key"] in st.session_state[RESP]}
        if not collected:
            _append("assistant", "Nothing collected yet — we haven't started!")
        else:
            lines = "\n".join(
                f"- {f['label']}: **{st.session_state[RESP][f['key']]}**"
                for f in features if f["key"] in collected
            )
            _append("assistant", f"Here's what you've told me so far:\n\n{lines}")
        return True
    return False


def render() -> None:
    features = _features()
    if not features:
        st.info("Feature schema not found.")
        return

    if MSG not in st.session_state:
        _reset()
        _append("assistant",
                "Hi! I'm the **Neuro-Screen assistant** 🧠 — I'll walk you through "
                "a 21-question check-in and run the hybrid model on your answers. "
                "Shall we start? (Just say **hello**, or reply with your answer to "
                "the first question below.)\n\n"
                + _question(features[0]))

    for message in st.session_state[MSG]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_text = st.chat_input("Type your answer…")
    if not user_text:
        return

    _append("user", esc(user_text))

    if CRISIS_PATTERN.search(user_text):
        _append(
            "assistant",
            "I'm really glad you told me. You are not alone, and this matters.\n\n"
            "Please reach out right now to the **AIUB Psychological Support Center** "
            "(your Student Affairs office has the contact details) — or call a "
            "trusted friend, family member, or the national emergency line. "
            "If you are in immediate danger, contact emergency services in "
            "Bangladesh on **999**.\n\n"
            "I'm here for the check-in whenever you're ready — type **restart** "
            "to continue, or take a break. 💙",
        )
    elif st.session_state[DONE]:
        _append("assistant",
                "You've already completed this check-in! Type **restart** to run "
                "it again.")
    elif _handle_command(user_text, features):
        pass
    else:
        _handle_answer(features, user_text)
        if not st.session_state[DONE]:
            _append("assistant",
                    _question(features[st.session_state[STEP]]))
        else:
            _summary_and_predict(features)
