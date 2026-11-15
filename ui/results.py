"""Schedule and comparison tabs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import metrics
from ui import charts, style


def _no_results() -> None:
    with st.container(border=True):
        st.markdown("#### Nothing to show yet")
        st.markdown("Check your inputs on the **📥 Inputs** tab, then press **▶ Optimize** (sidebar or bottom of the Inputs tab).")


def _guard(res: dict | None, stale: bool) -> bool:
    if res is None:
        _no_results()
        return False
    if stale:
        st.warning("Inputs have changed since this run – press **▶ Optimize** to refresh.", icon="⚠️")
    if not res["opt"]["ok"]:
        st.error(f"Solver returned no solution (status: {res['opt']['status']}). Try the LP model or a longer time limit.")
        return False
    return True


def _delta(value: float, fmt: str) -> str | None:
    return None if abs(value) < 1e-6 else fmt.format(value)


def kpi_row(k: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Optimized cost", f"₹{k['opt_cost']:,.0f}", border=True)
    c2.metric("Charge-immediately cost", f"₹{k['base_cost']:,.0f}", border=True)
    c3.metric("Saving", f"₹{k['saving']:,.0f}", _delta(k["saving_pct"], "{:.1f}%"), border=True)
    c4.metric("Peak site load", f"{k['opt_peak']:,.0f} kW", _delta(k["opt_peak"] - k["base_peak"], "{:+,.0f} kW vs baseline"),
              delta_color="inverse", border=True)
    c5.metric("Fully charged", f"{k['opt_served']} / {k['n_vehicles']}",
              _delta(k["opt_served"] - k["base_served"], "{:+.0f} vs baseline"), border=True)


def _insight(res: dict) -> str:
    inst, k = res["inst"], res["kpis"]
    peak = np.isclose(inst.price, inst.price.max()) & ~np.allclose(inst.price, inst.price[0])
    dt = inst.dt

    def peak_kwh(P):
        return float(P.sum(axis=0)[peak].sum() * dt)

    o_peak, b_peak = peak_kwh(res["opt"]["power"]), peak_kwh(res["base"]["power"])
    avg_o = k["opt_cost"] / k["opt_grid_kwh"] if k["opt_grid_kwh"] else 0
    avg_b = k["base_cost"] / k["base_grid_kwh"] if k["base_grid_kwh"] else 0
    parts = [f"<b>Optimized charging saves ₹{k['saving']:,.0f} ({k['saving_pct']:.1f}%)</b> compared with charging every vehicle on arrival."]
    if peak.any():
        parts.append(f"Energy drawn in the peak tariff block drops from <b>{b_peak:,.0f} kWh</b> to <b>{o_peak:,.0f} kWh</b>.")
    parts.append(f"The average price paid falls from ₹{avg_b:.2f} to <b>₹{avg_o:.2f}/kWh</b>.")
    if k["opt_served"] < k["n_vehicles"]:
        parts.append(f"⚠️ {k['n_vehicles'] - k['opt_served']} vehicle(s) cannot be fully charged with the current site limits.")
    else:
        parts.append(f"All {k['n_vehicles']} vehicles reach their target charge before departure.")
    return " ".join(parts)


def show_schedule(res: dict | None, stale: bool) -> None:
    if not _guard(res, stale):
        return
    inst, opt, base, k = res["inst"], res["opt"], res["base"], res["kpis"]

    kpi_row(k)
    style.insight(_insight(res))

    vs = res["vehicle_summary"]
    short = vs[~vs["target_met"]]
    if len(short):
        lines = "\n".join(f"- **{r.vehicle_id}** short by {r.shortfall_kwh:.2f} kWh ({r.delivered_kwh:.1f} of {r.need_kwh:.1f} kWh)" for r in short.itertuples())
        with st.expander(f"⚠️ {len(short)} vehicle(s) not fully charged – details", expanded=True):
            st.markdown(lines)
            st.caption("The model minimises total unmet energy (kWh), so it may spread a small shortfall over several vehicles. "
                       "Raise the site limit or chargers, or extend stays, to serve everyone.")

    with st.container(border=True):
        h1, h2 = st.columns([3, 2])
        h1.markdown("#### 🗓️ Charging schedule")
        which = h2.segmented_control("View", ["Optimized", "Baseline"], default="Optimized", key="sched_view",
                                     label_visibility="collapsed") or "Optimized"
        P = opt["power"] if which == "Optimized" else base["power"]
        st.caption("Each row is a vehicle; coloured cells show kW delivered while parked (grey = not at the site). "
                   "The strip on top shows the tariff – green is cheap, red is expensive.")
        st.plotly_chart(charts.schedule_heatmap(inst, P), width="stretch")

    with st.container(border=True):
        h1, h2, h3 = st.columns([3, 1.2, 1.2])
        h1.markdown(f"#### ⚡ Site load – {which.lower()}")
        stacked = h2.toggle("By vehicle", value=True)
        overlay = h3.toggle("Show baseline", value=which == "Optimized", disabled=which != "Optimized")
        st.plotly_chart(
            charts.load_profile(inst, P, base_P=base["power"] if overlay and which == "Optimized" else None, stacked=stacked,
                                label=f"{which} load", color=charts.OPT_COLOR if which == "Optimized" else charts.BASE_COLOR),
            width="stretch",
        )

    with st.container(border=True):
        st.markdown("#### 🚗 Per-vehicle summary")
        st.dataframe(
            vs, hide_index=True, width="stretch",
            column_config={
                "vehicle_id": "Vehicle", "type": "Type", "arrival": "Arrives", "departure": "Departs",
                "need_kwh": st.column_config.NumberColumn("Need (kWh)", format="%.1f"),
                "delivered_kwh": st.column_config.ProgressColumn("Delivered (kWh)", format="%.1f", min_value=0,
                                                                 max_value=float(max(vs["need_kwh"].max(), 1))),
                "shortfall_kwh": st.column_config.NumberColumn("Short (kWh)", format="%.2f"),
                "target_met": st.column_config.CheckboxColumn("Target met"),
                "charge_start": "Starts", "charge_end": "Ends",
                "grid_kwh": st.column_config.NumberColumn("Grid (kWh)", format="%.1f"),
                "cost_inr": st.column_config.NumberColumn("Cost", format="₹%.2f"),
                "avg_inr_per_kwh": st.column_config.NumberColumn("Avg ₹/kWh", format="%.2f"),
                "baseline_cost_inr": st.column_config.NumberColumn("Baseline cost", format="₹%.2f"),
                "saving_inr": st.column_config.NumberColumn("Saving", format="₹%.2f"),
            },
        )
        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "⬇️ Schedule (CSV)", metrics.schedule_table(inst, opt["power"]).to_csv(index=False).encode(),
            file_name="optimized_schedule.csv", mime="text/csv", width="stretch",
        )
        d2.download_button(
            "⬇️ Vehicle summary (CSV)", vs.to_csv(index=False).encode(),
            file_name="vehicle_summary.csv", mime="text/csv", width="stretch",
        )
        d3.download_button(
            "⬇️ Full report (Excel)", metrics.excel_report(inst, res, res["vehicles_in"], res["tariff_in"]),
            file_name="ev_charging_report.xlsx", width="stretch", type="primary",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.caption(
        f"Solver: {'MILP' if opt['milp'] else 'LP'} · {opt['status']} · {opt['solve_time']:.2f} s · "
        f"{opt['n_vars']:,} variables · {opt['n_cons']:,} constraints · {inst.n_slots} slots of {inst.site['slot_min']} min"
    )


def show_comparison(res: dict | None, stale: bool) -> None:
    if not _guard(res, stale):
        return
    inst, opt, base, k = res["inst"], res["opt"], res["base"], res["kpis"]
    kpi_row(k)

    y_max = max(inst.site["p_max"], opt["power"].sum(axis=0).max(), base["power"].sum(axis=0).max())
    with st.container(border=True):
        st.markdown("#### ⚡ Site load: charge immediately vs optimized")
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.load_profile(inst, base["power"], "Charge immediately (baseline)", stacked=False,
                                            label="Baseline load", color=charts.BASE_COLOR, y_max=y_max), width="stretch")
        c2.plotly_chart(charts.load_profile(inst, opt["power"], "Optimized", stacked=False,
                                            label="Optimized load", color=charts.OPT_COLOR, y_max=y_max), width="stretch")

    levels = metrics.cost_by_price_level(inst, opt, base)
    with st.container(border=True):
        st.markdown("#### 💡 Where the energy is bought")
        c3, c4 = st.columns(2)
        c3.plotly_chart(charts.cost_by_level(levels, "energy_kwh", "Energy drawn at each tariff level (kWh)"), width="stretch")
        c4.plotly_chart(charts.cost_by_level(levels, "cost_inr", "Cost at each tariff level (₹)"), width="stretch")

    avg_o = k["opt_cost"] / k["opt_grid_kwh"] if k["opt_grid_kwh"] else 0
    avg_b = k["base_cost"] / k["base_grid_kwh"] if k["base_grid_kwh"] else 0
    table = pd.DataFrame(
        {
            "Metric": ["Total cost (₹)", "Grid energy (kWh)", "Average price paid (₹/kWh)", "Peak load (kW)",
                       "Vehicles fully charged", "Total shortfall (kWh)"],
            "Charge immediately": [f"{k['base_cost']:,.2f}", f"{k['base_grid_kwh']:,.1f}", f"{avg_b:.2f}", f"{k['base_peak']:,.1f}",
                                   f"{k['base_served']}/{k['n_vehicles']}", f"{k['base_shortfall_kwh']:,.2f}"],
            "Optimized": [f"{k['opt_cost']:,.2f}", f"{k['opt_grid_kwh']:,.1f}", f"{avg_o:.2f}", f"{k['opt_peak']:,.1f}",
                          f"{k['opt_served']}/{k['n_vehicles']}", f"{k['opt_shortfall_kwh']:,.2f}"],
        }
    )
    with st.container(border=True):
        st.markdown("#### 📋 Side-by-side")
        st.dataframe(table, hide_index=True, width="stretch")
        if k["opt_shortfall_kwh"] < k["base_shortfall_kwh"] - 1e-3:
            st.caption("The optimized schedule also delivers more energy than the baseline, so part of its cost is energy the baseline never supplied.")
