"""
modules/ui/navigation.py
========================
Native Streamlit multipage navigation for Neuro-Screen.

The router is built on Streamlit's own multipage framework:

* ``st.navigation`` (called from ``app.py``) registers the canonical pages so
  URLs stay clean (``/``, ``/assessment``, ``/results``, ``/explainability``,
  ``/assistant``) and the browser Back/Forward buttons navigate through the
  app's page history.
* ``st.switch_page`` (``goto``) performs every programmatic page change. It is
  native SPA navigation: the page swaps inside the SAME browser tab — no new
  tabs, no full page reloads, no query-param hacks.

``app.py`` registers the ``st.Page`` objects here once per script run; every
widget (sidebar items, CTA buttons) reuses them via :func:`goto`.
"""

from __future__ import annotations

import streamlit as st

# Canonical page keys. The sidebar order and every ``goto(...)`` call use
# these exact identifiers.
PAGE_KEYS = ("home", "assessment", "results", "explainability", "assistant")

# key -> st.Page, populated by app.py on every script run.
_pages: dict[str, "st.Page"] = {}


def register(key: str, page: "st.Page") -> None:
    """Associate a page key with its canonical ``st.Page`` object."""
    _pages[key] = page


def goto(key: str) -> None:
    """Navigate to an internal page in the SAME browser tab.

    Uses ``st.switch_page`` (native multipage routing). Raises ``ValueError``
    for unknown keys so a typo fails loudly instead of silently doing nothing.
    """
    page = _pages.get(key)
    if page is None:
        raise ValueError(f"Unknown page key: {key!r}")
    st.switch_page(page)


def active_key(current: "st.Page") -> str:
    """Resolve which page key a ``st.navigation`` result corresponds to.

    Page identity is stable across reruns because ``st.Page`` hashes are
    derived from ``url_path``, so both object identity and hash comparison
    work here.
    """
    for key, page in _pages.items():
        if page is current or page._script_hash == current._script_hash:
            return key
    return PAGE_KEYS[0]
