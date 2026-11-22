"""Tiny page registry so any page can ``goto('results')`` inside the same tab."""

from __future__ import annotations

import streamlit as st

_pages: dict[str, "st.Page"] = {}


def register(key: str, page: "st.Page") -> None:
    _pages[key] = page


def goto(key: str) -> None:
    st.switch_page(_pages[key])
