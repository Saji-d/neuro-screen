"""
app.py
======
Neuro-Screen — a hybrid CatBoost + ANN screening tool for cognitive
impairment in insomniac university students.

Run with:
    streamlit run app.py

This entry point owns the app shell only:
  * global page config + theme injection (assets/styles.css)
  * sidebar (brand, native navigation, model status)
  * native multipage routing via st.navigation() + st.switch_page()
    (see modules/ui/navigation.py) — every page change happens inside the
    SAME browser tab, with clean URLs and working Back/Forward buttons.

All logic lives in modules/:
  modules/core/preprocessing.py   — notebook-exact feature engineering
  modules/predictor.py            — the ONE inference path (CatBoost + ANN hybrid)
  modules/core/risk.py            — severity tiers, drivers, recommendations
  modules/ui/pages/*              — page renderers (landing, assess, results, …)
  modules/chat/assistant.py       — conversational check-in (same pipeline)
  modules/charts/charts.py        — plotly figures for explainability
"""

from __future__ import annotations

import streamlit as st

from modules.ui import components, navigation
from modules.ui.pages import landing, assess, results, explainability, chat
from modules.ui.theme import inject_theme


def _build_pages() -> list:
    """Build the canonical st.Page set and register it with the router.

    Each page keeps the same stable ``url_path`` so the browser URL is clean
    (``/``, ``/assessment``, ``/results``, ``/explainability``, ``/assistant``)
    and Back/Forward navigation resolves correctly.
    """
    specs = (
        ("home", landing.render, "Home", "🏠", True),
        ("assessment", assess.render, "Assessment", "🧭", False),
        ("results", results.render, "Results", "📊", False),
        ("explainability", explainability.render, "Explainability", "🔬", False),
        ("assistant", chat.render, "Assistant", "💬", False),
    )
    pages = []
    for key, render_fn, title, icon, default in specs:
        page = st.Page(render_fn, title=title, icon=icon, url_path=key, default=default)
        navigation.register(key, page)
        pages.append(page)
    return pages


st.set_page_config(
    page_title="Neuro-Screen · Cognitive Impairment Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

pages = _build_pages()
pg = st.navigation(pages, position="hidden")
active = navigation.active_key(pg)
components.render_sidebar(active)

# Scroll to top only on an actual page change, not on every rerun — a
# widget interaction inside a page (e.g. answering a question) also
# triggers a rerun, and unconditionally resetting scroll there would yank
# the user back to the top mid-form.
if st.session_state.get("_current_page") != active:
    st.session_state["_current_page"] = active
    components.scroll_to_top()

pg.run()
