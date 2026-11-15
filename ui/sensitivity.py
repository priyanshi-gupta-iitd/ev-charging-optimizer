"""Scenario and sensitivity analysis tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import pipeline
from ui import charts

PARAMS = {
    "Site power limit (kW)": "p_max",
    "Number of chargers": "n_chargers",
    "Peak price multiplier": "peak_mult",
}


def _scaled_peak(tariff: pd.DataFrame, mult: float) -> pd.DataFrame:
    t = tariff.copy()
    peak = t["price_inr_per_kwh"].max()
    t.loc[np.isclose(t["price_inr_per_kwh"], peak), "price_inr_per_kwh"] = peak * mult
    return t


def _summary_row(label, value, res: dict) -> dict:
    k = res["kpis"]
    return {
        label: value,
        "opt_cost": round(k["opt_cost"], 2),
        "base_cost": round(k["base_cost"], 2),
        "saving_pct": round(k["saving_pct"], 2),
        "opt_peak_kw": round(k["opt_peak"], 2),
        "opt_unserved": k["n_vehicles"] - k["opt_served"],
        "base_unserved": k["n_vehicles"] - k["base_served"],
        "opt_shortfall_kwh": round(k["opt_shortfall_kwh"], 2),
        "status": res["opt"]["status"],
    }


def page(vehicles: pd.DataFrame, tariff: pd.DataFrame, settings: dict, site: dict, errors: list[str], run_cached) -> None:
    ss = st.session_state
    if errors:
        st.error("Fix the input errors on the Inputs tab first.")
        return

    with st.container(border=True):
        st.markdown("#### 🔬 Parameter sweep")
        st.caption("Re-solve the problem for a range of values of one parameter, keeping everything else as in the sidebar.")
        c1, c2, c3, c4, c5 = st.columns([1.6, 1, 1, 0.8, 1], vertical_alignment="bottom")
        label = c1.selectbox("Parameter to vary", list(PARAMS))
        key = PARAMS[label]
        if key == "p_max":
            lo, hi, step_default = 0.4 * site["p_max"], 1.6 * site["p_max"], 7
        elif key == "n_chargers":
            lo, hi, step_default = 1.0, float(max(2, 2 * site["n_chargers"])), 8
        else:
            lo, hi, step_default = 0.5, 2.0, 7
        vmin = c2.number_input("From", value=float(round(lo, 1)), key=f"from_{key}")
        vmax = c3.number_input("To", value=float(round(hi, 1)), key=f"to_{key}")
        steps = int(c4.number_input("Steps", 2, 25, step_default, key=f"steps_{key}"))
        run = c5.button("Run sweep", type="primary", width="stretch")

        if run:
            values = np.linspace(vmin, vmax, steps)
            if key == "n_chargers":
                values = np.unique(np.maximum(1, np.round(values)).astype(int))
            rows, bar = [], st.progress(0.0, text="Solving scenarios…")
            for j, val in enumerate(values):
                s, t = dict(site), tariff
                if key == "peak_mult":
                    t = _scaled_peak(tariff, float(val))
                else:
                    s[key] = int(val) if key == "n_chargers" else float(val)
                res = run_cached(vehicles, t, s, settings["milp"], settings["time_limit"], settings["gap"])
                rows.append(_summary_row(label, round(float(val), 3), res))
                bar.progress((j + 1) / len(values), text=f"Solved {j + 1}/{len(values)}")
            bar.empty()
            ss.sweep = {"label": label, "df": pd.DataFrame(rows)}

        sweep = ss.get("sweep")
        if sweep:
            df, lab = sweep["df"], sweep["label"]
            g1, g2 = st.columns(2)
            g1.plotly_chart(charts.sensitivity_lines(df, lab), width="stretch")
            g2.plotly_chart(charts.sensitivity_shortfall(df, lab), width="stretch")
            if (df["opt_unserved"] > 0).any() or (df["base_unserved"] > 0).any():
                st.caption("Where vehicles are left short, the two policies deliver different amounts of energy, so their costs "
                           "(and the saving %) are not like-for-like. The optimizer minimises total unmet kWh first, then cost.")
            with st.expander("Sweep results table"):
                st.dataframe(df, hide_index=True, width="stretch")
                st.download_button("⬇️ Sweep results (CSV)", df.to_csv(index=False).encode(), file_name="sensitivity.csv", mime="text/csv")
        else:
            st.info("Pick a parameter and press **Run sweep** to see how cost and service change.", icon="💡")

    with st.container(border=True):
        st.markdown("#### 📁 Saved scenarios")
        st.caption("Save the latest optimization run, change inputs, run again, and compare the setups side by side.")
        res = ss.get("results")
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        name = c1.text_input("Scenario name", value=f"Scenario {len(ss.get('scenarios', [])) + 1}")
        if c2.button("💾 Save current run", disabled=res is None, width="stretch"):
            row = _summary_row("scenario", name, res)
            row.update({"p_max_kw": res["inst"].site["p_max"], "chargers": res["inst"].site["n_chargers"],
                        "vehicles": len(res["inst"].vehicles), "slot_min": res["inst"].site["slot_min"]})
            ss.setdefault("scenarios", []).append(row)
        if c3.button("Clear all", width="stretch"):
            ss.scenarios = []
        if ss.get("scenarios"):
            st.dataframe(pd.DataFrame(ss.scenarios), hide_index=True, width="stretch")
