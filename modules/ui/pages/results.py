"""
modules/ui/pages/results.py
===========================
Results page — renders the RiskResult produced by the single inference
pipeline: risk meter, blended probabilities, prediction verdict, top
feature drivers and tiered recommendations.

The page is a pure renderer: it never calls the model itself, it only
displays whatever is stored in ``st.session_state["last_result"]`` by the
assessment (or chat) flow.
"""

from __future__ import annotations

import streamlit as st

from modules.core.risk import recommendations_for
from modules.ui import components, navigation
from modules.ui.theme import TIER_COLORS, esc, glass_card


def _result_missing() -> bool:
    return "last_result" not in st.session_state


def render() -> None:
    if _result_missing():
        st.markdown(
            glass_card(
                f"<div style='font-size:1.15rem;font-weight:600;'>No result yet</div>"
                f"<p style='color:#94a3b8;margin-top:6px;'>Complete a quick check-in "
                f"to run the hybrid model and see your risk assessment here.</p>",
                padding="1.8rem 2rem",
            ),
            unsafe_allow_html=True,
        )
        components.nav_button("Start the check-in", "assessment",
                              "cta_checkin", variant="primary")
        return

    result = st.session_state["last_result"]

    components.section_title("Your result", "Neuro-screen assessment")

    color = TIER_COLORS.get(result.severity, "#5eead4")

    # ---- verdict row: meter + headline numbers ----
    left, right = st.columns([1.1, 1.6])
    with left:
        st.markdown(components.risk_meter_html(result.risk_score, result.severity),
                    unsafe_allow_html=True)
    with right:
        st.markdown(
            glass_card(
                f"""
                <div class="ns-kicker" style="margin-bottom:8px;">Prediction</div>
                <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                  <span class="ns-chip"
                        style="color:{color};background:{color}1f;border:1px solid {color}55;
                               font-size:1rem;padding:7px 16px;">
                    {esc(result.prediction_label)}
                  </span>
                  <span class="ns-chip"
                        style="color:#e6edf7;background:rgba(255,255,255,.05);
                               border:1px solid rgba(255,255,255,.14);">
                    Confidence {result.confidence}%
                  </span>
                </div>
                <p style="color:#94a3b8;font-size:0.9rem;margin-top:12px;">
                  Blended probability of cognitive impairment:
                  <b style="color:#e6edf7;">{result.proba_impaired*100:.1f}%</b>
                  (decision threshold {result.threshold*100:.0f}%).
                  <span style="color:#64748b;">{esc(result.model_name)}</span>
                </p>
                """,
                padding="1.5rem 1.7rem",
            ),
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            components.stat_card("🎯", "Hybrid", f"{result.proba_impaired*100:.1f}%", "P(impaired)")
        with m2:
            components.stat_card("🌲", "CatBoost", f"{result.cat_proba*100:.1f}%",
                                 "P(impaired)")
        with m3:
            components.stat_card("🧠", "ANN", f"{result.ann_proba*100:.1f}%", "P(impaired)")

        if result.trained_at:
            st.caption(f"Model trained on {esc(result.trained_at)} — single live inference, no mock values.")

    st.markdown("")

    # ---- drivers + recommendations ----
    d_left, d_right = st.columns(2)
    with d_left:
        components.section_title("Top drivers", "What moved your score")
        if result.drivers:
            for driver in result.drivers:
                components.driver_row(
                    driver.feature,
                    driver.value,
                    pct=min(100.0, driver.magnitude * 900),
                )
            st.caption("Bars reflect per-instance SHAP contributions from the trained "
                       "CatBoost model (weighted).")
        else:
            st.markdown(
                glass_card("No driver data available for this prediction.",
                           padding="1.2rem 1.4rem"),
                unsafe_allow_html=True,
            )

    with d_right:
        components.section_title("Recommendations", "What you can try")
        recs = recommendations_for(result)
        rec_html = "".join(
            f"<li style='margin-bottom:10px;color:#cbd5e1;font-size:0.92rem;'>"
            f"<span style='color:{color};'>→</span>&nbsp; {esc(rec)}</li>"
            for rec in recs
        )
        st.markdown(
            glass_card(
                f"<ul style='padding-left:4px;list-style:none;margin:0;'>{rec_html}</ul>",
                padding="1.4rem 1.6rem",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")
    components.render_disclaimer()

    # ---- further actions ----
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Re-run check-in", width="stretch"):
            navigation.goto("assessment")
    with c2:
        if st.button("🔬 Why this score?", width="stretch"):
            navigation.goto("explainability")
    with c3:
        if st.button("💬 Ask the assistant", width="stretch"):
            navigation.goto("assistant")
