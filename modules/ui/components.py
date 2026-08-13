"""
modules/ui/components.py
========================
Reusable visual building blocks for Neuro-Screen.

Everything here is presentation only — the components never compute risk
scores; they only render what the core modules give them. That keeps the
"single inference pipeline" rule enforced in one place (modules/predictor.py).
"""

from __future__ import annotations

import base64

import streamlit as st

import config
from modules import predictor
from modules.ui import navigation
from modules.ui.theme import TIER_COLORS, esc, glass_card

NAV_ITEMS = [
    ("home", "🏠", "Home"),
    ("assessment", "🧭", "Assessment"),
    ("results", "📊", "Results"),
    ("explainability", "🔬", "Explainability"),
    ("assistant", "💬", "Assistant"),
]

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
def _logo_html() -> str:
    logo = config.ASSETS_DIR / "aiub_logo.png"
    if not logo.exists():
        return ""
    b64 = base64.b64encode(logo.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="AIUB logo"/>'


def nav_button(label: str, page: str, key: str, variant: str = "ghost") -> None:
    """CTA-style in-app navigation button.

    Navigates via ``st.switch_page`` (native multipage routing) so the page
    swaps inside the same browser tab — never a new tab or reload.
    """
    if st.button(
        label,
        key=key,
        type="primary" if variant == "primary" else "secondary",
        width="stretch",
    ):
        navigation.goto(page)


def render_sidebar(active: str) -> None:
    """Brand header, native navigation, model status and footer inside the sidebar.

    Navigation buttons call ``navigation.goto`` -> ``st.switch_page``, the
    native multipage router, so every click navigates in the same tab.
    """
    ready, label = _model_status()
    dot_color = "#34d399" if ready else "#fbbf24"
    status_text = "Model active" if ready else "Models not loaded"

    with st.sidebar:
        st.markdown(
            f'<div class="ns-brand">{_logo_html()}'
            f'<div><div class="ns-brand-name">{config.BRAND["name"]}</div>'
            f'<div class="ns-brand-sub">AIUB · CSE</div></div></div>',
            unsafe_allow_html=True,
        )

        for key, icon, label in NAV_ITEMS:
            is_active = key == active
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                type="primary" if is_active else "secondary",
                width="stretch",
            ):
                navigation.goto(key)

        st.markdown(
            f'<div style="font-size:0.78rem;color:#94a3b8;padding:0.2rem 0.2rem 1.1rem;">'
            f'<span class="ns-status-dot" style="background:{dot_color};'
            f'box-shadow:0 0 10px {dot_color};"></span>{status_text}'
            f'<div style="font-size:0.7rem;color:#64748b;margin-top:4px;">{esc(label)}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:0.9rem;">'
            f'<div style="color:#64748b;font-size:0.7rem;line-height:1.5;">'
            f'Designed &amp; developed by<br/>'
            f'<span style="color:#94a3b8;font-weight:500;">Sajidur Rahman Sajid</span></div></div>',
            unsafe_allow_html=True,
        )


def _model_status() -> tuple[bool, str]:
    if predictor.models_available():
        return True, "CatBoost + ANN hybrid loaded"
    missing = ", ".join(predictor.missing_artifacts())
    return False, missing or "Run scripts/train_models.py"


# ------------------------------------------------------------------
# Status / banner
# ------------------------------------------------------------------
def models_ready() -> bool:
    """True when the real model artifacts are present."""
    return predictor.models_available()


def render_model_banner() -> None:
    """Honest, prominent banner when the trained artifacts are missing.

    Shown on every page that would otherwise display a prediction, so the
    app never pretends to run a model it does not have.
    """
    if predictor.models_available():
        return
    st.markdown(
        f"""
        <div class="ns-banner warn ns-animate">
          <div style="font-size:1.3rem;">🛠️</div>
          <div style="flex:1;">
            <div style="font-weight:600;color:#fbbf24;">Trained model weights are not loaded yet</div>
            <div style="color:#cbd5e1;font-size:0.9rem;margin-top:4px;">
              Neuro-Screen will <b>not fabricate a score</b>. To enable real predictions:
              place the survey CSV in <code>data/</code> and run
              <code>python scripts/train_models.py</code> — it reproduces the
              Thesis3.ipynb pipeline (CatBoost + ANN + hybrid blending) and exports
              the model artifacts. Then refresh this page.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        glass_card(
            "⚠️&nbsp; Neuro-Screen is a research prototype for screening cognitive-"
            "impairment risk factors linked to insomnia. It is <b>not a medical "
            "diagnosis</b> and does not replace professional care. If you are "
            "struggling, please reach out to the <b>AIUB Psychological Support "
            "Center</b> or someone you trust.",
            padding="0.9rem 1.2rem",
        ),
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Risk meter (pure SVG, animated arc)
# ------------------------------------------------------------------
def risk_meter_html(score: int, tier: str) -> str:
    """Animated semicircular risk meter. 0-100, color-coded by tier."""
    clamped = max(0, min(100, score))
    arc_length = 301.59  # semicircle, r=96
    dash = arc_length * clamped / 100.0
    color = TIER_COLORS.get(tier, "#5eead4")
    return f"""
    <div class="ns-meter-wrap ns-animate">
      <div class="ns-meter-ring" style="width:250px;height:175px;">
        <svg viewBox="0 0 240 140" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="meterGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="#22d3ee"/>
              <stop offset="55%" stop-color="#5eead4"/>
              <stop offset="100%" stop-color="#818cf8"/>
            </linearGradient>
          </defs>
          <path d="M 24 120 A 96 96 0 0 1 216 120" fill="none"
                stroke="rgba(255,255,255,0.07)" stroke-width="15" stroke-linecap="round"/>
          <path d="M 24 120 A 96 96 0 0 1 216 120" fill="none"
                stroke="url(#meterGrad)" stroke-width="15" stroke-linecap="round"
                stroke-dasharray="{dash:.1f} 400"
                style="filter:drop-shadow(0 0 10px {color}99);"/>
          <text x="24" y="137" font-size="11" fill="#64748b" text-anchor="middle">0</text>
          <text x="120" y="30" font-size="11" fill="#64748b" text-anchor="middle">50</text>
          <text x="216" y="137" font-size="11" fill="#64748b" text-anchor="middle">100</text>
        </svg>
        <div class="ns-meter-num" style="position:absolute;left:0;right:0;top:52%;bottom:0;
             display:flex;align-items:flex-start;justify-content:center;flex-direction:column;">
          <span style="font-size:2.5rem;line-height:1;">{clamped}</span>
          <small style="font-size:0.72rem;color:#64748b;letter-spacing:.12em;
               text-transform:uppercase;">Risk score</small>
        </div>
      </div>
      <div style="margin-top:14px;">
        <span class="ns-chip" style="color:{color};background:{color}1f;border:1px solid {color}55;">
          {esc(tier)}
        </span>
      </div>
    </div>
    """


# ------------------------------------------------------------------
# Stat / info cards
# ------------------------------------------------------------------
def stat_card(icon: str, label: str, value: str, sub: str | None = None) -> None:
    sub_html = f'<div class="ns-stat-sub">{esc(sub)}</div>' if sub else ""
    st.markdown(
        f'<div class="glass glass-hover ns-stat ns-animate">'
        f'<div class="ns-stat-icon">{icon}</div>'
        f'<div class="ns-stat-label">{esc(label)}</div>'
        f'<div class="ns-stat-value">{esc(value)}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def section_title(kicker: str, title: str) -> None:
    st.markdown(
        f'<div class="ns-section-head ns-animate">'
        f'<span class="ns-kicker">{esc(kicker)}</span></div>'
        f'<h2 style="margin-bottom:1rem;">{esc(title)}</h2>',
        unsafe_allow_html=True,
    )


def driver_row(name: str, value: str, pct: float) -> None:
    """Horizontal contribution bar for a top feature driver."""
    pct = max(0.0, min(100.0, pct))
    st.markdown(
        f"""
        <div class="ns-driver">
          <div style="flex:1;min-width:0;">
            <div style="color:#e6edf7;font-size:0.9rem;font-weight:500;">{esc(name)}</div>
            <div style="color:#64748b;font-size:0.76rem;">{esc(value)}</div>
          </div>
          <div class="ns-driver-bar" style="max-width:45%;">
            <div class="ns-driver-fill" style="width:{pct:.0f}%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
