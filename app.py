"""Cost-Optimal EV Charging Scheduler – Streamlit app.

Run with:  streamlit run app.py
"""

import hashlib

import pandas as pd
import streamlit as st

from core import data, pipeline
from ui import inputs, results, sensitivity, style

st.set_page_config(page_title="EV Charging Optimizer", page_icon="⚡", layout="wide")


@st.cache_data(show_spinner=False, max_entries=64)
def run_cached(vehicles: pd.DataFrame, tariff: pd.DataFrame, site: dict, milp: bool, time_limit: int, gap: float) -> dict:
    res = pipeline.run(vehicles, tariff, site, milp=milp, time_limit=time_limit, gap=gap)
    res["vehicles_in"], res["tariff_in"] = vehicles, tariff
    return res


def signature(vehicles: pd.DataFrame, tariff: pd.DataFrame, settings: dict) -> str:
    blob = vehicles.to_csv(index=False) + tariff.to_csv(index=False) + repr(sorted(settings.items()))
    return hashlib.md5(blob.encode()).hexdigest()


style.apply()
style.sidebar_brand()
run_slot = st.sidebar.container()
st.sidebar.divider()
hero_slot = st.container()

settings = inputs.sidebar()
site = inputs.site_params(settings)

tab_in, tab_sched, tab_cmp, tab_sens, tab_model = st.tabs(
    ["📥 Inputs", "📊 Optimized schedule", "⚖️ Baseline vs optimized", "🔬 Scenarios & sensitivity", "📘 Model & assumptions"]
)

with tab_in:
    vehicles = inputs.vehicle_section(settings)
    tariff = inputs.tariff_section()
    with st.container(border=True):
        errors = data.validate(vehicles, tariff, site)
        inputs.precheck_panel(vehicles, tariff, site, errors)
        main_run = st.button("▶ Optimize schedule", type="primary", disabled=bool(errors), key="run_main")

with run_slot:
    side_run = st.button("▶ Optimize", type="primary", disabled=bool(errors), width="stretch", key="run_side")
    run_note = st.empty()

if side_run or main_run:
    with st.spinner("Solving…"):
        st.session_state.results = run_cached(vehicles, tariff, site, settings["milp"], settings["time_limit"], settings["gap"])
        st.session_state.results_sig = signature(vehicles, tariff, settings)
    st.toast("Done – open the 📊 Optimized schedule tab", icon="✅")

res = st.session_state.get("results")
stale = res is not None and st.session_state.get("results_sig") != signature(vehicles, tariff, settings)

if errors:
    run_note.caption("⚠️ Fix input errors to enable optimization.")
elif res is None:
    run_note.caption("Press to compute the cost-optimal schedule.")
elif stale:
    run_note.caption("⚠️ Inputs changed – re-run to refresh results.")
else:
    run_note.caption(f"✅ Last run: {res['opt']['status']} in {res['opt']['solve_time']:.2f} s")

chips = [
    (f"🚗 {len(vehicles)} vehicles", ""),
    (f"🔌 {site['n_chargers']} chargers", ""),
    (f"⚡ {site['p_max']:,.0f} kW site limit", ""),
    (f"🕒 {site['start']} + {site['horizon_h']:g} h · {site['slot_min']}-min slots", ""),
]
if errors:
    chips.append(("⚠️ Input errors", "warn"))
elif res is None:
    chips.append(("Not optimized yet", ""))
elif stale:
    chips.append(("⚠️ Results out of date", "warn"))
elif res["opt"]["ok"]:
    k = res["kpis"]
    chips.append((f"💰 Saves ₹{k['saving']:,.0f} ({k['saving_pct']:.1f}%) vs charging immediately", "ok"))
with hero_slot:
    style.hero(chips)

with tab_sched:
    results.show_schedule(res, stale)

with tab_cmp:
    results.show_comparison(res, stale)

with tab_sens:
    sensitivity.page(vehicles, tariff, settings, site, errors, run_cached)

with tab_model:
    st.subheader("Mathematical model")
    st.markdown(
        "**Sets & data** – vehicles $v \\in V$, time slots $t \\in T$ of length $\\Delta t$ hours; "
        "tariff $c_t$ (₹/kWh); vehicle $v$ is parked in slots $T_v$ (arrival to departure), needs "
        "$E_v = B_v (\\text{SOC}^{target}_v - \\text{SOC}^{now}_v)$ kWh and can draw at most $R_v$ kW; "
        "site limit $P^{max}$ kW, $N$ chargers, efficiency $\\eta$."
    )
    st.markdown("**Decision variables** – $p_{v,t} \\ge 0$ power (kW), $y_{v,t} \\in \\{0,1\\}$ charger use, $u_v \\ge 0$ unmet energy (kWh).")
    st.latex(r"\min \; \sum_{v \in V} \sum_{t \in T_v} c_t \, p_{v,t} \, \Delta t \; + \; M \sum_{v \in V} u_v")
    st.latex(r"\eta \, \Delta t \sum_{t \in T_v} p_{v,t} + u_v = E_v \qquad \forall v")
    st.latex(r"0 \le p_{v,t} \le R_v \, y_{v,t} \qquad \forall v,\; t \in T_v")
    st.latex(r"\sum_{v} y_{v,t} \le N \qquad \forall t")
    st.latex(r"\sum_{v} p_{v,t} \le P^{max} \qquad \forall t")
    st.markdown(
        "$M$ is a large penalty (≫ any tariff), so $u_v > 0$ only when the site physically cannot "
        "serve vehicle $v$; the app then reports it as a shortfall instead of declaring the problem infeasible. "
        "In **LP mode** $y$ is dropped and the charger limit becomes $\\sum_v p_{v,t}/R_v \\le N$."
    )
    st.subheader("Baseline policy")
    st.markdown(
        "*Charge immediately*: in every slot, parked vehicles that still need energy are served "
        "first-come-first-served at full rate, subject to the same charger count and site limit."
    )
    st.subheader("Assumptions")
    st.markdown(
        "- Charging power is constant up to the target SOC (no CC-CV taper).\n"
        "- Tariff is applied using the price at the start of each slot.\n"
        "- Arrival earlier than the horizon start is taken as the next day; departure earlier than arrival is the next day.\n"
        "- Vehicle and site data are **synthetic but realistic**; preset tariffs are **illustrative** and should be replaced "
        "by the cited state electricity regulator ToU / EV tariff."
    )
    st.subheader("How to use")
    st.markdown(
        "1. Set the horizon, site limit, chargers and efficiency in the sidebar.\n"
        "2. Upload or edit the vehicle table and choose a tariff on **Inputs**.\n"
        "3. Press **▶ Optimize**, then review the schedule and comparison tabs and download the results.\n"
        "4. Use **Scenarios & sensitivity** to see how cost and service change with capacity, chargers or prices."
    )
