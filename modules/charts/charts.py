"""
modules/charts/charts.py
========================
Plotly figure builders for the Explainability page.

Charts are pure renderers: they take data (reported paper numbers or live
evaluation artifacts) and return a Plotly figure styled for the Neuro-Screen
dark theme. They never load models or compute scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

GRID = "rgba(148,163,184,0.12)"
TICK = "#94a3b8"
TEXT = "#e6edf7"
MUTED = "#64748b"
TEAL = "#22d3ee"
INDIGO = "#818cf8"
GOOD = "#34d399"
WARN = "#fbbf24"
DANGER = "#fb7185"

_MODEL_COLORS = {"CatBoost": TEAL, "ANN": INDIGO, "Hybrid": "#5eead4"}


def _style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": TEXT},
        height=height,
        margin=dict(l=30, r=20, t=46, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=dict(font=dict(size=14, color=TEXT), x=0.02),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(color=TICK))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(color=TICK))
    return fig


@st.cache_data(show_spinner=False)
def confusion_matrix(tn: float, fp: float, fn: float, tp: float,
                     labels=("Healthy", "Impaired"), height: int = 380) -> go.Figure:
    """Heatmap with value + percentage annotations.

    Cell text color is chosen per-cell from the cell's own value (not a
    single fixed color) — the brightest cell in this colorscale is a light
    cyan, and light text on it (the previous fixed color) was unreadable.
    """
    cm = np.array([[tn, fp], [fn, tp]])
    total = cm.sum()
    cm_max = cm.max() or 1
    text = np.array([
        [
            (f"<span style='color:{('#0b1120' if v / cm_max > 0.6 else TEXT)}'>{v}</span>"
             f"<br><span style='font-size:11px;"
             f"color:{('#0b1120bb' if v / cm_max > 0.6 else '#94a3b8')}'>"
             f"{v/total*100:.1f}%</span>")
            for v in row
        ]
        for row in cm
    ])
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted Healthy", "Predicted Impaired"],
        y=["Actual Healthy", "Actual Impaired"],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=15),
        colorscale=[[0, "#0f172a"], [0.45, "#155e75"], [1, "#22d3ee"]],
        showscale=False,
        hovertemplate="%{x}<br>%{y}<br>count: %{z}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": TEXT}, height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Confusion Matrix (held-out test set)", x=0.02,
                   font=dict(size=14, color=TEXT)),
    )
    return fig


@st.cache_data(show_spinner=False)
def metrics_grouped(rows: dict[str, list[float]], metric_names: list[str],
                    height: int = 380) -> go.Figure:
    """Grouped bar chart comparing CatBoost / ANN / Hybrid across metrics."""
    df = pd.DataFrame(rows, index=metric_names).reset_index().melt(
        id_vars="index", var_name="Model", value_name="Score")
    df = df.rename(columns={"index": "Metric"})
    df["Score"] = df["Score"] * 100
    fig = px.bar(df, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_map=_MODEL_COLORS,
                 text_auto=".1f")
    fig.update_yaxes(range=[0, 100])
    fig.update_traces(textposition="outside", cliponaxis=False,
                      marker_line_width=0, hovertemplate="%{y:.2f}%<extra></extra>")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": TEXT}, height=height,
        margin=dict(l=30, r=20, t=76, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=dict(text="Model Performance Comparison (%)", x=0.02, y=0.98,
                   yanchor="top", font=dict(size=14, color=TEXT)),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=TICK))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(color=TICK))
    return fig


@st.cache_data(show_spinner=False)
def feature_importance(items: list[dict], height: int = 480) -> go.Figure:
    """Horizontal bar chart of CatBoost feature importances."""
    df = pd.DataFrame(items)
    df = df.sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"], orientation="h",
        marker=dict(color=[TEAL], opacity=0.9),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        text=[f"{v*100:.1f}%" for v in df["importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": TEXT}, height=height,
        margin=dict(l=10, r=60, t=46, b=30),
        title=dict(text="CatBoost Feature Importance", x=0.02, font=dict(size=14, color=TEXT)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(color=TICK))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=TICK))
    return fig


@st.cache_data(show_spinner=False)
def shap_contrib(names: list[str], values: list[str], contributions: list[float],
                 height: int = 360) -> go.Figure:
    """Per-instance SHAP contributions (signed) for the current prediction."""
    fig = go.Figure(go.Bar(
        y=names, x=contributions, orientation="h",
        marker=dict(color=[GOOD if v >= 0 else DANGER for v in contributions]),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        text=[f"{v:+.3f}" for v in contributions],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": TEXT}, height=height,
        margin=dict(l=10, r=70, t=46, b=30),
        title=dict(text="Feature contributions for this prediction (SHAP)",
                   x=0.02, font=dict(size=14, color=TEXT)),
    )
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(color=TICK))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=TICK))
    return fig


@st.cache_data(show_spinner=False)
def roc_curve(fpr: list[float], tpr: list[float], auc: float,
              height: int = 380) -> go.Figure:
    """ROC curve from live evaluation artifacts (or paper-reported point)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=TEAL, width=3),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.08)",
        name=f"Hybrid (AUC={auc:.4f})",
        hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="rgba(148,163,184,0.4)", width=1.5, dash="dash"),
        name="Chance", showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": TEXT}, height=height,
        margin=dict(l=10, r=10, t=76, b=30),
        title=dict(text="ROC Curve", x=0.02, y=0.98, yanchor="top", font=dict(size=14, color=TEXT)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="False Positive Rate", gridcolor=GRID, zeroline=False, tickfont=dict(color=TICK)),
        yaxis=dict(title="True Positive Rate", gridcolor=GRID, zeroline=False, tickfont=dict(color=TICK)),
    )
    return fig
