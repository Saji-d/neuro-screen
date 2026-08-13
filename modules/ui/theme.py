"""
modules/ui/theme.py
===================
Design-system entry point for Neuro-Screen.

Loads the global stylesheet (assets/styles.css), injects it into the app,
and exposes small helpers used by every page: tier colours, HTML escaping,
and a reusable HTML-shell function for custom components.
"""

from __future__ import annotations

import html as _html

import streamlit as st

import config

# Tier -> accent colour map (also used by charts).
TIER_COLORS = {
    "Low Risk": "#34d399",
    "Moderate Risk": "#fbbf24",
    "High Risk": "#fb7185",
}

TIER_GLOWS = {
    "Low Risk": "rgba(52, 211, 153, 0.35)",
    "Moderate Risk": "rgba(251, 191, 36, 0.35)",
    "High Risk": "rgba(251, 113, 133, 0.35)",
}


def _read_css() -> str:
    css_path = config.ASSETS_DIR / "styles.css"
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""


def inject_theme() -> None:
    """Apply the Neuro-Screen design system once at app startup."""
    st.markdown(f"<style>{_read_css()}</style>", unsafe_allow_html=True)


def esc(text: object) -> str:
    """HTML-escape a value for safe interpolation into custom markup."""
    return _html.escape(str(text), quote=True)


def html(inner: str, css_class: str = "", style: str = "") -> str:
    """Wrap arbitrary HTML in a generic <div> for st.markdown(unsafe_allow_html)."""
    cls = f' class="{css_class}"' if css_class else ""
    stl = f' style="{style}"' if style else ""
    return f'<div{cls}{stl}>{inner}</div>'


def glass_card(inner: str, padding: str = "1.6rem 1.8rem", extra: str = "") -> str:
    """Standard glass card shell used across pages."""
    return (
        f'<div class="glass glass-hover" style="padding:{padding};{extra}">'
        f"{inner}</div>"
    )
