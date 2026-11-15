"""Plotly figures used across the app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from core.data import Instance, tariff_by_minute

OPT_COLOR = "#0f9d76"
BASE_COLOR = "#94a3b8"
LIMIT_COLOR = "#e03131"
PRICE_COLOR = "#f08c00"
PEAK_FILL = "rgba(224, 49, 49, 0.07)"
IDLE_BG = "#eef1f4"
FONT = "Source Sans Pro, Segoe UI, sans-serif"

pio.templates["evopt"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=13, color="#1f2933"),
        title=dict(font=dict(size=15), x=0, xanchor="left", pad=dict(l=4)),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        colorway=["#0f9d76", "#3b82f6", "#f08c00", "#8b5cf6", "#ef4444", "#14b8a6", "#eab308", "#64748b"],
        xaxis=dict(gridcolor="#eef1f4", linecolor="#cbd5e1", zeroline=False),
        yaxis=dict(gridcolor="#eef1f4", linecolor="#cbd5e1", zeroline=False),
        hoverlabel=dict(font=dict(family=FONT)),
    )
)
pio.templates.default = "plotly_white+evopt"


def _step(x: pd.DatetimeIndex, y: np.ndarray, end: pd.Timestamp) -> tuple[list, list]:
    return list(x) + [end], list(y) + [y[-1]]


def _price_runs(inst: Instance) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Contiguous runs of slots with the same price."""
    slot = pd.Timedelta(minutes=inst.site["slot_min"])
    runs, start = [], 0
    for k in range(1, inst.n_slots + 1):
        if k == inst.n_slots or not np.isclose(inst.price[k], inst.price[start]):
            runs.append((inst.times[start], inst.times[k - 1] + slot, float(inst.price[start])))
            start = k
    return runs


def _tariff_band(fig: go.Figure, inst: Instance, plot_px: float) -> None:
    """Colour-coded tariff strip along the top edge of the plot (green = cheap, red = expensive)."""
    lo, hi = float(inst.price.min()), float(inst.price.max())
    y0 = 1 + 4 / plot_px
    y1 = y0 + 18 / plot_px
    for x0, x1, price in _price_runs(inst):
        t = 0.5 if hi == lo else (price - lo) / (hi - lo)
        color = px.colors.sample_colorscale("RdYlGn_r", [0.1 + 0.8 * t])[0]
        fig.add_shape(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=color, opacity=0.85, line=dict(width=1, color="white"), layer="above")
        if (x1 - x0) >= pd.Timedelta(hours=1):
            fig.add_annotation(x=x0 + (x1 - x0) / 2, y=(y0 + y1) / 2, xref="x", yref="paper", text=f"<b>₹{price:g}/kWh</b>",
                               showarrow=False, yanchor="middle", font=dict(size=11, color="#1f2933"))


def _shade_peaks(fig: go.Figure, inst: Instance) -> None:
    if np.allclose(inst.price, inst.price[0]):
        return
    for x0, x1, price in _price_runs(inst):
        if np.isclose(price, inst.price.max()):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=PEAK_FILL, line_width=0, layer="below")


def _time_axis(inst: Instance) -> dict:
    return dict(range=[inst.times[0], inst.end_time], tickformat="%H:%M", title=None, showgrid=False)


def schedule_heatmap(inst: Instance, P: np.ndarray) -> go.Figure:
    """Gantt-style view: rows are vehicles, columns are slots, colour is kW delivered."""
    V = inst.vehicles
    half = pd.Timedelta(minutes=inst.site["slot_min"] / 2)
    z = np.where(inst.availability(), P, np.nan)
    labels = [f"{vid} · {t}" for vid, t in zip(V["vehicle_id"], V["type"])]

    fig = go.Figure(
        go.Heatmap(
            z=z, x=inst.times + half, y=labels,
            colorscale=[[0, "#ffffff"], [0.02, "#d3f2e7"], [0.5, "#3cbf95"], [1, "#0b5e47"]],
            zmin=0, zmax=float(V["rmax"].max()),
            colorbar=dict(title=dict(text="kW", side="top"), thickness=12, len=0.6, y=0.5),
            xgap=1, ygap=3,
            hovertemplate="%{y}<br>%{x|%H:%M}<br><b>%{z:.2f} kW</b><extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(
        x=V["arr_ts"], y=labels, mode="markers", name="Arrival",
        marker=dict(symbol="triangle-right", size=9, color="#3b82f6"),
        hovertemplate="%{y}<br>arrives %{x|%H:%M}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=V["dep_ts"].clip(upper=inst.end_time), y=labels, mode="markers", name="Departure",
        marker=dict(symbol="line-ns", size=12, line=dict(width=3, color=LIMIT_COLOR)),
        hovertemplate="%{y}<br>departs %{x|%H:%M}<extra></extra>",
    ))
    height = max(320, 24 * len(V) + 160)
    _tariff_band(fig, inst, plot_px=height - 115)
    fig.update_layout(
        height=height, plot_bgcolor=IDLE_BG,
        xaxis=_time_axis(inst), yaxis=dict(autorange="reversed", title=None, showgrid=False),
        legend=dict(orientation="h", y=-0.06, x=0, yanchor="top"),
        margin=dict(l=10, r=10, t=65, b=50),
    )
    return fig


def load_profile(
    inst: Instance,
    P: np.ndarray,
    title: str = "",
    base_P: np.ndarray | None = None,
    stacked: bool = True,
    label: str = "Optimized load",
    color: str = OPT_COLOR,
    y_max: float | None = None,
) -> go.Figure:
    """Site load vs the power limit, with the tariff on a secondary axis."""
    V = inst.vehicles
    half = pd.Timedelta(minutes=inst.site["slot_min"] / 2)
    width_ms = inst.site["slot_min"] * 60 * 1000 * 0.92
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if stacked:
        palette = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
        for i, vid in enumerate(V["vehicle_id"]):
            if P[i].sum() <= 0:
                continue
            fig.add_trace(go.Bar(
                x=inst.times + half, y=P[i], name=vid, width=width_ms, marker_color=palette[i % len(palette)],
                hovertemplate=f"{vid}<br>%{{x|%H:%M}}<br>%{{y:.2f}} kW<extra></extra>",
            ), secondary_y=False)
    else:
        fig.add_trace(go.Bar(
            x=inst.times + half, y=P.sum(axis=0), name=label, width=width_ms, marker_color=color,
            hovertemplate="%{x|%H:%M}<br>%{y:.1f} kW<extra></extra>",
        ), secondary_y=False)

    if base_P is not None:
        bx, by = _step(inst.times, base_P.sum(axis=0), inst.end_time)
        fig.add_trace(go.Scatter(
            x=bx, y=by, name="Baseline load", line=dict(color="#334155", width=2, dash="dot", shape="hv"), legendrank=2,
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=[inst.times[0], inst.end_time], y=[inst.site["p_max"]] * 2, name="Site limit",
        mode="lines", line=dict(color=LIMIT_COLOR, width=2, dash="dash"), legendrank=1,
    ), secondary_y=False)

    px_, py_ = _step(inst.times, inst.price, inst.end_time)
    fig.add_trace(go.Scatter(
        x=px_, y=py_, name="Tariff (₹/kWh)", line=dict(color=PRICE_COLOR, width=2.5, shape="hv"), legendrank=3,
        hovertemplate="₹%{y:.2f}/kWh<extra></extra>",
    ), secondary_y=True)

    _shade_peaks(fig, inst)
    legend = (dict(orientation="v", x=1.06, y=1, font=dict(size=11)) if stacked
              else dict(orientation="h", x=0, y=-0.12, yanchor="top"))
    fig.update_layout(
        title=title or None, barmode="stack", bargap=0, height=420, legend=legend,
        margin=dict(l=10, r=10, t=50 if title else 20, b=40), xaxis=_time_axis(inst), hovermode="x unified",
    )
    top = y_max or max(inst.site["p_max"], P.sum(axis=0).max(initial=0),
                       0 if base_P is None else base_P.sum(axis=0).max(initial=0))
    fig.update_yaxes(title_text="Power (kW)", range=[0, top * 1.12], secondary_y=False)
    fig.update_yaxes(title_text="₹/kWh", range=[0, inst.price.max() * 1.3], secondary_y=True,
                     showgrid=False, tickprefix="₹", tickformat=".1f")
    return fig


def tariff_preview(tariff: pd.DataFrame) -> go.Figure:
    price, _ = tariff_by_minute(tariff)
    x = pd.date_range("2026-01-01", periods=24 * 60 + 1, freq="min")
    y = np.append(price, price[-1])
    fig = go.Figure(go.Scatter(
        x=x, y=y, line=dict(color=PRICE_COLOR, width=3, shape="hv"), fill="tozeroy",
        fillcolor="rgba(240,140,0,0.14)", hovertemplate="%{x|%H:%M}: ₹%{y:.2f}/kWh<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=40, b=30), title="Price over the day",
        xaxis=dict(tickformat="%H:%M", dtick=3 * 3600 * 1000, showgrid=False),
        yaxis=dict(title="₹/kWh", rangemode="tozero", tickprefix="₹"),
    )
    return fig


def cost_by_level(df: pd.DataFrame, value: str, title: str) -> go.Figure:
    fig = px.bar(
        df, x="price_level", y=value, color="schedule", barmode="group", title=title, text_auto=".0f",
        color_discrete_map={"Optimized": OPT_COLOR, "Baseline": BASE_COLOR},
        category_orders={"schedule": ["Baseline", "Optimized"]},
        labels={"price_level": "Tariff level", "energy_kwh": "Energy (kWh)", "cost_inr": "Cost (₹)"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=40), legend=dict(title=None, orientation="h", y=1.08, x=1, xanchor="right"))
    return fig


def sensitivity_lines(df: pd.DataFrame, param: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df[param], y=df["base_cost"], name="Baseline cost", mode="lines+markers", line=dict(color=BASE_COLOR, width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df[param], y=df["opt_cost"], name="Optimized cost", mode="lines+markers", line=dict(color=OPT_COLOR, width=3),
                             fill="tonexty", fillcolor="rgba(15,157,118,0.10)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df[param], y=df["saving_pct"], name="Saving (%)", mode="lines+markers", line=dict(color=PRICE_COLOR, dash="dot")), secondary_y=True)
    fig.update_layout(height=400, title=f"Cost vs {param}", margin=dict(l=10, r=10, t=50, b=40),
                      legend=dict(orientation="h", y=-0.18), hovermode="x unified")
    fig.update_xaxes(title_text=param)
    fig.update_yaxes(title_text="Cost (₹)", secondary_y=False, rangemode="tozero", tickprefix="₹")
    fig.update_yaxes(title_text="Saving (%)", secondary_y=True, rangemode="tozero", showgrid=False, ticksuffix="%")
    return fig


def sensitivity_shortfall(df: pd.DataFrame, param: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[param], y=df["base_unserved"], name="Baseline", marker_color=BASE_COLOR))
    fig.add_trace(go.Bar(x=df[param], y=df["opt_unserved"], name="Optimized", marker_color=OPT_COLOR))
    fig.update_layout(height=400, barmode="group", title=f"Vehicles not fully charged vs {param}",
                      margin=dict(l=10, r=10, t=50, b=40), yaxis=dict(title="Vehicles", rangemode="tozero"),
                      xaxis=dict(title=param), legend=dict(orientation="h", y=-0.18))
    return fig
