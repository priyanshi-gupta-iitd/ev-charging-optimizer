"""Page-level styling and the header banner."""

from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1400px; }
[data-testid="stSidebarContent"] { padding-top: 0.4rem; }
[data-testid="stSidebar"] h4 { margin: 0.6rem 0 0.1rem 0; font-size: 0.95rem; letter-spacing: .02em; }
.brand { font-size: 1.15rem; font-weight: 700; color: #fff; margin: .2rem 0 .1rem 0; }
.brand-sub { font-size: .78rem; color: #9fb8c9; margin-bottom: .6rem; }

.hero {
  background: linear-gradient(120deg, #0f2a3d 0%, #135e63 55%, #0f9d76 100%);
  border-radius: 14px; padding: 1.3rem 1.6rem; color: #fff; margin-bottom: 1rem;
  box-shadow: 0 4px 18px rgba(15, 42, 61, .18);
}
.hero h1 { color: #fff; font-size: 1.75rem; margin: 0 0 .2rem 0; padding: 0; }
.hero p { color: #d5ece6; margin: 0 0 .8rem 0; font-size: .95rem; }
.chip {
  display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
  border-radius: 999px; padding: .18rem .7rem; margin: 0 .35rem .3rem 0; font-size: .82rem; color: #fff;
}
.chip.ok { background: rgba(46, 204, 113, .28); border-color: rgba(46, 204, 113, .55); }
.chip.warn { background: rgba(241, 196, 15, .28); border-color: rgba(241, 196, 15, .6); }

button[data-baseweb="tab"] p { font-size: 1rem; font-weight: 600; }
div[data-testid="stMetric"] { background: #fff; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }

.insight {
  background: #eaf7f2; border-left: 4px solid #0f9d76; border-radius: 8px;
  padding: .8rem 1rem; margin: .4rem 0 1rem 0; color: #1f2933; font-size: .95rem;
}
</style>
"""


def apply() -> None:
    st.html(CSS)


def sidebar_brand() -> None:
    st.sidebar.html('<div class="brand">⚡ EV Charge Optimizer</div>'
                    '<div class="brand-sub">MSL304 · Group 13 · ToU cost minimisation</div>')


def hero(chips: list[tuple[str, str]]) -> None:
    """chips: (text, kind) with kind in {"", "ok", "warn"}."""
    chip_html = "".join(f'<span class="chip {k}">{html.escape(t)}</span>' for t, k in chips)
    st.html(
        '<div class="hero"><h1>Cost-Optimal EV Charging Scheduler</h1>'
        "<p>Schedule EV charging to minimise electricity cost under time-of-use tariffs, "
        f"site power limits and vehicle deadlines.</p>{chip_html}</div>"
    )


def insight(text: str) -> None:
    st.html(f'<div class="insight">{text}</div>')
