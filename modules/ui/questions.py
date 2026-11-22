"""Assessment questions: UI metadata mapped onto the model's 12 predictors."""

from __future__ import annotations

from modules.ml.survey import AGE

FREQ4 = ["Never", "Rarely", "Sometimes", "Always"]
FREQ5 = ["Never", "Rarely", "Sometimes", "Often", "Always"]
BANDS = ["1-2", "3-4", "5 or more"]
BAND_LABEL = {"1-2": "1–2", "3-4": "3–4", "5 or more": "5 or more"}

# key -> spec. ``col`` is the exact training column name.
Q = {
    "age": dict(col=AGE, kind="slider", label="How old are you?", help="Respondents were aged 18–28.",
                lo=18, hi=28, default=23),
    "gender": dict(col="What is your gender?", kind="radio", label="Gender",
                   help="As reported in the survey.", options=["Female", "Male"], default="Female"),
    "university": dict(col="What is your University name?", kind="select", label="University",
                       help="One of the 31 universities in the survey.", default="AIUB"),
    "year": dict(col="Your current year of study?", kind="radio", label="Year of study",
                 help="", options=["1st Year", "2nd Year", "3rd Year", "4th Year", "Graduated"],
                 default="3rd Year"),
    "caffeine": dict(col="How many cups of caffeine (tea/coffee/energy drinks) do you have daily?",
                     kind="radio", label="Daily caffeine", help="Cups of tea, coffee or energy drinks.",
                     options=BANDS, default="1-2", fmt=BAND_LABEL),
    "devices": dict(col="How often do you use electronic devices (phone/laptop) in bed before sleep?",
                    kind="slider_opt", label="Screens in bed", help="Phone or laptop before sleep.",
                    options=FREQ4, default="Sometimes"),
    "wake": dict(col="How often do you wake up in the middle of the night?", kind="slider_opt",
                 label="Night wakings", help="Waking up after you have fallen asleep.",
                 options=["Never", "Sometimes", "Always"], default="Sometimes"),
    "load": dict(col="How often do you feel cognitive load?", kind="slider_opt",
                 label="Cognitive load", help="Feeling mentally overloaded.",
                 options=FREQ4, default="Sometimes"),
    "stress": dict(col="How often do you feel stressed?", kind="slider_opt", label="Stress",
                   help="", options=FREQ5, default="Sometimes"),
    "fatigue": dict(col="How often does physical or mental fatigue prevent you from completing your daily goals?",
                    kind="slider_opt", label="Fatigue getting in the way", help="",
                    options=FREQ5, default="Sometimes"),
    "deadlines": dict(col="How many academic deadlines have you missed or delayed in the last month?",
                      kind="radio", label="Missed or delayed deadlines", help="In the last month.",
                      options=BANDS, default="1-2", fmt=BAND_LABEL),
    "gpa": dict(col="To what extent do you feel sleep deprivation is lowering your GPA?",
                kind="slider_opt", label="Sleep loss and your GPA", help="Your own sense of the impact.",
                options=["No impact", "Minor impact", "Moderate impact", "High impact", "Severe impact"],
                default="Moderate impact"),
}

STEPS = [
    ("About you", "A little context so the model can compare you with the right cohort.",
     ["age", "gender", "university", "year"]),
    ("Sleep habits", "Everyday habits around bedtime.", ["caffeine", "devices", "wake"]),
    ("Mind and body", "How your days feel right now.", ["load", "stress", "fatigue"]),
    ("Academics", "How things are going at university.", ["deadlines", "gpa"]),
]

DEFAULTS = {k: v["default"] for k, v in Q.items()}


def answers_to_columns(state: dict) -> dict[str, object]:
    return {Q[k]["col"]: state[k] for k in Q}
