"""
app.py
======
Neuro-Screen: a four-model stack (CatBoost, XGBoost, LightGBM, ANN) that
screens sleep risk and cognitive-symptom risk from 12 self-reported answers.

    streamlit run app.py

This file owns the shell only: page config, theme and top navigation.
All logic lives in modules/ (see modules/ml, modules/predictor.py, modules/ui).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Neuro-Screen", page_icon="🧠", layout="wide",
                   initial_sidebar_state="collapsed")

from modules.ui import nav, theme
from modules.ui.pages import assess, home, insights, results

theme.inject()

specs = (
    ("home", home.render, "Home", ":material/home:", True),
    ("assess", assess.render, "Assess", ":material/psychology:", False),
    ("results", results.render, "Results", ":material/monitoring:", False),
    ("insights", insights.render, "Reliability", ":material/fact_check:", False),
)
pages = []
for key, fn, title, icon, default in specs:
    page = st.Page(fn, title=title, icon=icon, url_path=key if not default else "", default=default)
    nav.register(key, page)
    pages.append(page)

st.navigation(pages, position="top").run()
