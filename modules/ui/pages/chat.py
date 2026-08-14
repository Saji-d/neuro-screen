"""
modules/ui/pages/chat.py
========================
Chat page — thin wrapper that renders the conversational assistant and the
standard status banner/disclaimer around it.
"""

from __future__ import annotations

import streamlit as st

from modules.chat import assistant
from modules.ui import components


def render() -> None:
    components.render_model_banner()
    components.section_title("Assistant", "Chat through your check-in")
    components.render_disclaimer()
    assistant.render()
    components.scroll_to_latest_chat()
