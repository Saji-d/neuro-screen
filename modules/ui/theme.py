"""Design-system helpers: CSS injection, HTML escaping and reusable fragments."""

from __future__ import annotations

import html as _html
import math
import textwrap

import streamlit as st

import config


def esc(text: object) -> str:
    return _html.escape(str(text), quote=True)


def md(markup: str) -> None:
    """Render raw HTML. Dedented and de-blanked: Markdown treats 4-space
    indents and blank lines inside HTML as code / paragraph breaks."""
    lines = [ln.strip() for ln in textwrap.dedent(markup).splitlines() if ln.strip()]
    st.markdown(" ".join(lines), unsafe_allow_html=True)


def inject() -> None:
    css = (config.ASSETS_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def kicker(text: str) -> str:
    return f'<div class="kicker">{esc(text)}</div>'


def section(title: str, sub: str = "") -> None:
    md(f'<div class="h-section rise">{esc(title)}</div>'
       + (f'<div class="sub-section rise d1">{esc(sub)}</div>' if sub else ""))


def callout(body: str, kind: str = "warn", icon: str = "⚠️") -> str:
    cls = "" if kind == "warn" else f" {kind}"
    return f'<div class="callout{cls}"><div>{icon}</div><div>{body}</div></div>'


def card(icon: str, title: str, body: str, delay: int = 1) -> str:
    return (f'<div class="card rv"><div class="icon-chip">{icon}</div>'
            f"<h4>{esc(title)}</h4><p>{esc(body)}</p></div>")


def expand(icon: str, title: str, summary: str, detail: str, delay: int = 1, cls: str = "rv") -> str:
    """Click-to-expand card (native <details>, works without JavaScript)."""
    return (f'<details class="card {cls}"><summary><div class="icon-chip">{icon}</div>'
            f"<h4>{esc(title)}</h4><p>{esc(summary)}</p><div class=\"hint\">Tap to read more</div></summary>"
            f'<div class="more">{detail}</div></details>')


def stat(value: str, label: str, suffix: str = "", delay: int = 1) -> str:
    num = value.replace(",", "")
    inner = (f'<span class="count" style="--v:{num}"></span>{esc(suffix)}'
             if num.isdigit() and "," not in value else esc(value))
    return (f'<div class="card stat flat rise d{delay}"><div class="num grad-text">{inner}</div>'
            f'<div class="lab">{esc(label)}</div></div>')


def brain_svg() -> str:
    """Animated neural-network motif for the hero."""
    pts = [(60, 90), (60, 170), (60, 250), (60, 330),
           (160, 60), (160, 140), (160, 220), (160, 300), (160, 380),
           (260, 100), (260, 190), (260, 280), (260, 360),
           (350, 160), (350, 250), (350, 330)]
    layers = [pts[0:4], pts[4:9], pts[9:13], pts[13:16]]
    edges = "".join(
        f'<line class="edge" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
        f'style="animation-delay:{(i * 0.17) % 3:.2f}s"/>'
        for la, lb in zip(layers, layers[1:]) for i, (a, b) in enumerate((a, b) for a in la for b in lb))
    nodes = "".join(
        f'<circle class="node" cx="{x}" cy="{y}" r="9"/>'
        f'<circle class="pulse" cx="{x}" cy="{y}" r="3.2" style="animation-delay:{(i * 0.37) % 2.4:.2f}s"/>'
        for i, (x, y) in enumerate(pts))
    return ('<svg class="brain" viewBox="0 0 410 440" xmlns="http://www.w3.org/2000/svg">'
            '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#8b7cf6"/><stop offset=".55" stop-color="#22d3ee"/>'
            '<stop offset="1" stop-color="#34d399"/></linearGradient></defs>'
            f"{edges}{nodes}</svg>")


def gauge(prob: float, base: float, color: str, caption: str) -> str:
    r = 80
    circ = 2 * math.pi * r
    off = circ * (1 - prob)
    angle = base * 360
    return (
        f'<div class="gauge" style="--len:{circ:.1f};--off:{off:.1f};--c:{color}">'
        '<svg viewBox="0 0 200 200" width="200" height="200">'
        f'<circle class="trk" cx="100" cy="100" r="{r}"/>'
        f'<circle class="bar" cx="100" cy="100" r="{r}" stroke="{color}"/>'
        f'<g transform="rotate({angle:.1f} 100 100)"><line class="tick" x1="{100 + r - 13}" y1="100" '
        f'x2="{100 + r + 13}" y2="100"/></g></svg>'
        f'<div class="mid"><div class="val"><span class="count" style="--v:{round(prob * 100)}"></span>'
        f'<small>%</small></div><div class="cap">{esc(caption)}</div></div></div>')


def bar_rows(rows: list[tuple[str, float, str]], maxv: float = 1.0, signed: bool = False) -> str:
    """rows: (label, value, display). Width scales value/maxv (abs for signed)."""
    out = []
    for label, value, shown in rows:
        w = min(abs(value) / maxv, 1) * 100
        cls = ("pos" if value > 0 else "neg") if signed else ""
        out.append(f'<div class="bar-row"><div>{esc(label)}</div>'
                   f'<div class="t"><i class="{cls}" style="--w:{w:.1f}%"></i></div>'
                   f'<div class="n">{esc(shown)}</div></div>')
    return "".join(out)


def disclaimer() -> str:
    return callout("<b>Research prototype, not a medical device.</b> Results come from self-reported "
                   "survey answers and are not a diagnosis. If sleep, memory or mood is affecting your "
                   "life, please talk to a qualified professional or your university's counselling service.")


def footer() -> None:
    md('<div class="foot">Neuro-Screen · Sajidur Rahman Sajid · Research prototype, not for clinical use</div>')
