"""
modules/chat/assistant.py
=========================
The conversational check-in.

The assistant collects the same 21 answers as the quick check-in, one question
at a time. Replies are matched loosely to each question's allowed options so
users can answer naturally ("around 5 hours", "male", "i'm male"); genuinely
unclear answers get a short clarification instead of a long list of accepted
values. The university question accepts any reasonable institution name —
known variants are mapped to the model's vocabulary when possible, otherwise
the user's own name is preserved.

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

# ------------------------------------------------------------------
# Light text normalisation — tolerate natural human phrasing
# ------------------------------------------------------------------
_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ORDINAL_WORDS = {
    "first": "1st", "second": "2nd", "third": "3rd",
    "fourth": "4th", "fifth": "5th",
}
_STOPWORDS = {"a", "an", "the", "of", "at", "in", "for", "to", "and",
              "or", "my", "i'm", "im", "about"}

# Bare numbers (no sign) for safe range/option parsing — "5-6" must yield
# [5, 6], never [5, -6]. The signed form is only used when a sign is wanted.
_NUM_BARE = r"\d+(?:\.\d+)?"
_NUM_RE = r"[-+]?" + _NUM_BARE


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _words_to_digits(text: str) -> str:
    """Replace number words (and compound tens) with digits."""
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        key = tokens[i].strip(".,-")
        if key in _NUM_WORDS:
            val = _NUM_WORDS[key]
            if val >= 20 and i + 1 < len(tokens):
                nxt = tokens[i + 1].strip(".,-")
                if nxt in _NUM_WORDS and _NUM_WORDS[nxt] < 10:
                    out.append(str(val + _NUM_WORDS[nxt]))
                    i += 2
                    continue
            out.append(str(val))
            i += 1
            continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


def _prepare(text: str) -> str:
    """Normalise: lowercase, ordinal words -> digits, number words -> digits."""
    t = " " + _norm(text) + " "
    for word, repl in _ORDINAL_WORDS.items():
        t = t.replace(f" {word} ", f" {repl} ")
    t = _words_to_digits(t.strip())
    return re.sub(r"\s+", " ", t).strip()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _norm(text)))


# ------------------------------------------------------------------
# Option matching
# ------------------------------------------------------------------
# Single-letter / shorthand answers that are unambiguous for a given question.
_ALIASES = {
    "gender": {"m": "Male", "man": "Male", "guy": "Male", "boy": "Male",
               "f": "Female", "woman": "Female", "girl": "Female"},
    "audio_lag": {"y": "Yes", "yeah": "Yes", "yep": "Yes",
                  "n": "No", "nope": "No", "nah": "No"},
}


def _token_match(answer: str, options: list[str]) -> list[str]:
    """Weak fuzzy match: every significant token of the answer appears in one
    option's tokens (e.g. "dhaka university" -> "University of Dhaka")."""
    at = _tokens(answer) - _STOPWORDS
    if not at:
        return []
    out = []
    for opt in options:
        ot = _tokens(opt) - _STOPWORDS
        if ot and at.issubset(ot):
            out.append(opt)
    return out


def _parse_numeric(text: str):
    """Infer a numeric intent from prepared text.

    Returns one of ``("value", n)``, ``("range", lo, hi)``, ``("below", x)``
    or ``("above", x)``, or ``None`` when no usable number is present.
    """
    t = _prepare(text)
    below = re.search(
        r"(?:less\s*than|under|below|fewer\s*than|at\s*most|no\s*more\s*than)\s*(" + _NUM_BARE + ")", t)
    above = re.search(
        r"(?:more\s*than|over|above|greater\s*than|at\s*least|no\s*less\s*than)\s*(" + _NUM_BARE + ")", t)
    if below:
        return ("below", float(below.group(1)))
    if above:
        return ("above", float(above.group(1)))
    # "-5 hours" -> the user means "less than 5 hours"
    neg = re.search(r"(?<!\d)-\s*(" + _NUM_BARE + ")", t)
    if neg:
        return ("below", float(neg.group(1)))
    nums = [float(n) for n in re.findall(_NUM_BARE, t)]
    if not nums:
        return None
    if len(nums) == 1:
        return ("value", nums[0])
    return ("range", nums[0], nums[1])


def _option_bounds(opt: str):
    """Numeric bounds encoded by an option string, or ``None``."""
    o = _prepare(opt)
    nums = [float(n) for n in re.findall(_NUM_BARE, o)]
    if not nums:
        return None
    if re.search(r"less\s*than|under|below|fewer\s*than", o):
        return ("below", nums[0])
    if re.search(r"more\s*than|over|above|greater\s*than|or\s*more|at\s*least", o):
        return ("above", nums[0])
    if len(nums) >= 2:
        return ("range", nums[0], nums[1])
    return ("value", nums[0])


def _interval(b) -> tuple[float, float]:
    kind = b[0]
    if kind == "below":
        return (float("-inf"), b[1])
    if kind == "above":
        return (b[1], float("inf"))
    if kind == "range":
        return (b[1], b[2])
    return (b[1], b[1])


def _center(b) -> float:
    lo, hi = b
    if lo == float("-inf"):
        lo = 0.0
    if hi == float("inf"):
        hi = lo + 4.0
    return (lo + hi) / 2.0


def _overlap(i1: tuple[float, float], i2: tuple[float, float]) -> float:
    return max(0.0, min(i1[1], i2[1]) - max(i1[0], i2[0]))


def _contains(itv: tuple[float, float], oiv: tuple[float, float]) -> int:
    """1 when the intent interval is fully inside the option interval."""
    return 1 if itv[0] >= oiv[0] and itv[1] <= oiv[1] else 0


def _match_numeric_option(answer: str, options: list[str]):
    """Pick the option whose numeric range best fits the answer's number(s)."""
    intent = _parse_numeric(answer)
    if intent is None:
        return None
    itv = _interval(intent)
    best = None
    best_score = (float("-inf"), float("-inf"), float("-inf"))
    for opt in options:
        b = _option_bounds(opt)
        if b is None:
            continue
        oiv = _interval(b)
        score = (_overlap(itv, oiv),
                 _contains(itv, oiv),
                 -abs(_center(itv) - _center(oiv)))
        if score > best_score:
            best_score = score
            best = opt
    return best


def _resolve_select(answer: str, feat: dict) -> tuple[str | None, list[str]]:
    """Map free text to one allowed option.

    Returns ``(value, ambiguous)`` — ``ambiguous`` is non-empty when the
    answer maps to several options and no numeric signal decides between them.
    """
    options = feat.get("options") or []
    p = _prepare(answer)

    aliases = _ALIASES.get(feat.get("key", ""), {})
    if aliases and p in aliases:
        return aliases[p], []

    exact = [o for o in options if _prepare(o) == p]
    if len(exact) == 1:
        return exact[0], []

    # Answers that carry a number resolve numerically first, so "5 hours"
    # lands on "5-6 hours" rather than on the "less than 5 hours" substring.
    if _parse_numeric(answer) is not None:
        numeric = _match_numeric_option(answer, options)
        if numeric is not None:
            return numeric, []

    strong = [o for o in options if _prepare(o) in p or p in _prepare(o)]
    if len(strong) == 1:
        return strong[0], []
    if len(strong) > 1:
        return None, strong

    fuzzy = _token_match(answer, options)
    if len(fuzzy) == 1:
        return fuzzy[0], []
    if len(fuzzy) > 1:
        return None, fuzzy
    return None, []


def _parse_number(text: str):
    m = re.search(_NUM_RE, _prepare(text))
    return float(m.group(0)) if m else None


def _clean_free_text(text: str) -> str | None:
    """Sanitise an arbitrary free-form answer (used for the university name)."""
    t = re.sub(r"[^A-Za-z0-9&\-\. ]+", " ", text)
    t = re.sub(r"\s+", " ", t).strip(" .-")
    if len(t) < 2 or not any(c.isalpha() for c in t):
        return None
    return t


def _clarify(matches: list[str]) -> str:
    shown = matches[:3]
    return ("Just to be sure — did you mean " +
            ", ".join(f"**{m}**" for m in shown) + "? 🙂")


# ------------------------------------------------------------------
# Question prompts — short, natural, no long option lists
# ------------------------------------------------------------------
_HINTS = {
    "age": "a number between 17 and 28",
    "gender": "e.g. Male or Female",
    "university": "Just type the name of your university.",
    "academic_year": "e.g. 2nd year",
    "sleep_hours": "e.g. around 5 hours",
    "sleep_quality": "e.g. good, poor, or a rating from 1 to 5",
    "awakenings": "e.g. sometimes",
    "caffeine_intake": "e.g. 2 cups a day",
    "bedtime_device_use": "e.g. sometimes",
    "cognitive_load": "e.g. sometimes",
    "stress_frequency": "e.g. often",
    "fatigue_level": "e.g. often",
    "gpa_impact": "e.g. minor impact",
    "missed_deadlines": "e.g. 2 deadlines",
    "mental_stamina": "e.g. full energy",
    "spacing_out": "e.g. rarely",
    "audio_lag": "e.g. no, maybe, or yes",
    "forgetfulness": "e.g. sometimes",
    "reminder_reliance": "e.g. sometimes",
    "brain_fog": "e.g. sometimes",
    "decision_making": "e.g. sometimes",
}


def _question(feat: dict) -> str:
    icon = feat.get("icon", "📌")
    hint = _HINTS.get(feat.get("key", ""), "Answer in your own words.")
    return (
        f"{icon} **{feat['label']}**\n\n{feat.get('description', '')}\n\n"
        f"_{hint}_"
    )


# ------------------------------------------------------------------
# Conversation state
# ------------------------------------------------------------------
def _reset() -> None:
    st.session_state[MSG] = []
    st.session_state[RESP] = {}
    st.session_state[STEP] = 0
    st.session_state[DONE] = False


def _append(role: str, content: str) -> None:
    st.session_state[MSG].append({"role": role, "content": content})


def _features() -> list[dict]:
    return config.load_feature_schema()


def _advance(features: list[dict]) -> None:
    st.session_state[STEP] += 1
    if st.session_state[STEP] >= len(features):
        st.session_state[DONE] = True


# ------------------------------------------------------------------
# Answer handling
# ------------------------------------------------------------------
def _handle_answer(features: list[dict], user_text: str) -> None:
    feat = features[st.session_state[STEP]]
    key = feat["key"]

    if feat.get("input_type") == "number":
        n = _parse_number(user_text)
        if n is None:
            _append("assistant", "That didn't look like a number — could you "
                                 "type it as digits? 🙂")
            return
        lo, hi = int(feat.get("min", 0)), int(feat.get("max", 120))
        if not lo <= n <= hi:
            _append("assistant",
                    f"Please enter a number between {lo} and {hi}.")
            return
        st.session_state[RESP][key] = str(int(n))
    else:
        value, ambiguous = _resolve_select(user_text, feat)
        if value is None:
            if key == "university":
                value = _clean_free_text(user_text)
                if value is None:
                    _append("assistant",
                            "Could you tell me the name of your university? 🙂")
                    return
            elif ambiguous:
                _append("assistant", _clarify(ambiguous))
                return
            else:
                hint = _HINTS.get(key, "phrasing it differently")
                _append("assistant",
                        f"I couldn't quite catch that — try {hint} 🙂")
                return
        st.session_state[RESP][key] = value

    _advance(features)


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
        with st.spinner("Running the hybrid model on your answers…"):
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
                "Hi! I'm the **Neuro-Screen assistant** 🧠 — I'll ask 21 short "
                "questions about your sleep and day, then run the risk model on "
                "your answers.\n\n"
                + _question(features[0]))

    for message in st.session_state[MSG]:
        avatar = "🧑" if message["role"] == "user" else "🧠"
        label = "You" if message["role"] == "user" else "Neuro-Screen Assistant"
        with st.chat_message(message["role"], avatar=avatar):
            st.caption(label)
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

    # Messages appended above were added AFTER the render loop already ran,
    # so without forcing a fresh rerun here they would stay invisible until
    # the user's *next* message — at which point both the missed response
    # and the new one would appear together, looking like a duplicate. This
    # rerun makes the message list (the single source of truth) the only
    # thing that ever gets rendered.
    st.rerun()
