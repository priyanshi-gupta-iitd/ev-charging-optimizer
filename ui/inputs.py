"""Sidebar settings and the Inputs tab (vehicles, tariff, pre-check)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from core import data, generator, tariffs
from ui import charts

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"

VEHICLE_COLUMN_CONFIG = {
    "vehicle_id": st.column_config.TextColumn("Vehicle ID", required=True),
    "type": st.column_config.TextColumn("Type", help="e.g. e-rickshaw, car, e-bus"),
    "arrival": st.column_config.TextColumn("Arrival (HH:MM)", required=True),
    "departure": st.column_config.TextColumn("Departure (HH:MM)", required=True, help="Earlier than arrival = next day"),
    "battery_kwh": st.column_config.NumberColumn("Battery (kWh)", min_value=0.1, format="%.1f", required=True),
    "soc_now_pct": st.column_config.NumberColumn("Current SOC (%)", min_value=0, max_value=100, required=True),
    "soc_target_pct": st.column_config.NumberColumn("Target SOC (%)", min_value=0, max_value=100, required=True),
    "max_rate_kw": st.column_config.NumberColumn("Max rate (kW)", min_value=0.1, format="%.1f", required=True),
}

TARIFF_COLUMN_CONFIG = {
    "start": st.column_config.TextColumn("Start (HH:MM)", required=True),
    "end": st.column_config.TextColumn("End (HH:MM)", required=True, help="Use 24:00 for midnight; end < start wraps past midnight"),
    "price_inr_per_kwh": st.column_config.NumberColumn("Price (₹/kWh)", min_value=0.0, format="%.2f", required=True),
}


def _bump(key: str) -> None:
    st.session_state[key] = st.session_state.get(key, 0) + 1


def _normalise_times(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    def fix(x):
        try:
            minutes = data.parse_hhmm(x)
        except ValueError:
            return x
        return "24:00" if minutes == data.MIN_PER_DAY else data.fmt_hhmm(minutes)

    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(fix)
    return df


def _read_table(upload) -> pd.DataFrame:
    if upload.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    return pd.read_csv(upload, dtype=str)


def sidebar() -> dict:
    """Render the settings sidebar; the run button is placed separately at the top (see app.py)."""
    sb = st.sidebar
    sb.markdown("#### 🏭 Site")
    p_max = sb.number_input("Site power limit (kW)", min_value=1.0, value=150.0, step=10.0,
                            help="Sanctioned load available for EV charging.")
    c1, c2 = sb.columns(2)
    n_chargers = int(c1.number_input("Chargers", min_value=1, value=10, step=1))
    eta = c2.number_input("Efficiency", min_value=0.5, max_value=1.0, value=0.90, step=0.01,
                          help="Share of grid energy that ends up in the battery.")

    sb.markdown("#### 🕒 Planning horizon")
    c1, c2 = sb.columns(2)
    start = c1.time_input("Start", value=dt.time(18, 0), step=900)
    slot_min = c2.selectbox("Slot (min)", [15, 30, 60], index=1)
    horizon_h = sb.slider("Length (hours)", 4, 48, 16)

    with sb.expander("🧮 Solver settings"):
        mode = st.radio(
            "Model", ["MILP (exact charger limit)", "LP (relaxed, faster)"],
            help="The LP relaxes the charger count to 'fractional chargers' (Σ p/R ≤ N). Use it for very large instances.",
        )
        time_limit = int(st.number_input("Time limit (s)", min_value=5, max_value=600, value=30, step=5))
        gap_pct = st.number_input("Optimality gap (%)", min_value=0.0, max_value=10.0, value=0.5, step=0.1,
                                  help="Stop once the solution is provably within this % of optimal. 0 = prove exact optimality (slower).")
    return {
        "gap": float(gap_pct) / 100.0,
        "start": start.strftime("%H:%M"),
        "horizon_h": float(horizon_h),
        "slot_min": int(slot_min),
        "p_max": float(p_max),
        "n_chargers": n_chargers,
        "eta": float(eta),
        "milp": mode.startswith("MILP"),
        "time_limit": time_limit,
    }


def site_params(settings: dict) -> dict:
    return {k: settings[k] for k in ("start", "horizon_h", "slot_min", "p_max", "n_chargers", "eta")}


def _set_vehicles(df: pd.DataFrame) -> None:
    st.session_state.vehicles = df
    _bump("veh_ver")


def _demo(settings: dict) -> pd.DataFrame:
    return generator.generate(20, start=settings["start"], horizon_h=settings["horizon_h"], eta=settings["eta"])


def vehicle_section(settings: dict) -> pd.DataFrame:
    ss = st.session_state
    if "vehicles" not in ss:
        ss.vehicles, ss.veh_ver = _demo(settings), 0

    with st.container(border=True):
        st.markdown("### ① Vehicles")
        t1, t2, t3, t4, t5 = st.columns(5)
        with t1.popover("📤 Upload file", width="stretch"):
            upload = st.file_uploader("Vehicle table (CSV / Excel)", type=["csv", "xlsx", "xls"], key="veh_upload")
            st.caption("Columns: " + ", ".join(f"`{c}`" for c in data.VEHICLE_COLUMNS))
            if upload is not None and ss.get("veh_upload_id") != upload.file_id:
                try:
                    _set_vehicles(_normalise_times(data.clean_vehicles(_read_table(upload)), ["arrival", "departure"]))
                    ss.veh_upload_id = upload.file_id
                except Exception as exc:  # noqa: BLE001 - show any parse error to the user
                    st.error(f"Could not read file: {exc}")
        if t2.button("🎲 Demo scenario", width="stretch", help="20 vehicles, overnight depot"):
            _set_vehicles(_demo(settings))
        with t3.popover("🎛️ Random generator", width="stretch"):
            g1, g2 = st.columns(2)
            n = g1.number_input("Vehicles", 1, 300, 30)
            span = g2.number_input("Arrivals spread (h)", 0.5, 24.0, 4.0, 0.5)
            g3, g4 = st.columns(2)
            seed = int(g3.number_input("Random seed", 0, 10_000, 7))
            feasible = g4.checkbox("Long-enough stays", value=True, help="Each vehicle can be fully charged on its own")
            st.caption("Vehicle mix (%)")
            m1, m2, m3 = st.columns(3)
            mix = {
                "e-rickshaw": m1.number_input("e-rickshaw", 0, 100, 40),
                "car": m2.number_input("Car", 0, 100, 45),
                "e-bus": m3.number_input("e-bus", 0, 100, 15),
            }
            if st.button("Generate", type="primary", width="stretch"):
                if sum(mix.values()) == 0:
                    st.warning("Set at least one non-zero share.")
                else:
                    _set_vehicles(generator.generate(
                        int(n), mix, start=settings["start"], horizon_h=settings["horizon_h"],
                        arrival_span_h=span, seed=seed, ensure_feasible=feasible, eta=settings["eta"],
                    ))
            st.caption("Assumed: e-rickshaw 8–10 kWh @ 3.3 kW · car 30–60 kWh @ 7.2/30 kW · e-bus 200–300 kWh @ 60 kW")
        t4.download_button(
            "⬇️ CSV template", (SAMPLE_DIR / "vehicles_template.csv").read_bytes(),
            file_name="vehicles_template.csv", mime="text/csv", width="stretch",
        )
        if t5.button("🗑️ Clear", width="stretch"):
            _set_vehicles(pd.DataFrame(columns=data.VEHICLE_COLUMNS))

        edited = st.data_editor(
            ss.vehicles, key=f"veh_editor_{ss.veh_ver}", num_rows="dynamic", width="stretch", height=320,
            column_config=VEHICLE_COLUMN_CONFIG, hide_index=True,
        )
        vehicles = data.clean_vehicles(edited)
        if len(vehicles) and "type" in vehicles:
            counts = vehicles["type"].value_counts()
            st.caption(f"**{len(vehicles)} vehicles** · " + " · ".join(f"{c} {t}" for t, c in counts.items())
                       + " · edit cells directly, add rows at the bottom")
    return vehicles


def tariff_section() -> pd.DataFrame:
    ss = st.session_state
    if "tariff" not in ss:
        ss.tariff, ss.tariff_applied, ss.tar_ver = tariffs.get(tariffs.DEFAULT_PRESET), tariffs.DEFAULT_PRESET, 0

    with st.container(border=True):
        st.markdown("### ② Time-of-use tariff")
        c1, c2 = st.columns([1, 1.5], gap="large")
        with c1:
            names = list(tariffs.PRESETS) + ["Custom (edit table)"]
            choice = st.selectbox("Preset", names, index=0, key="tariff_choice")
            if choice in tariffs.PRESETS and choice != ss.tariff_applied:
                ss.tariff, ss.tariff_applied = tariffs.get(choice), choice
                _bump("tar_ver")
            elif choice not in tariffs.PRESETS:
                ss.tariff_applied = choice

            with st.popover("📤 Upload tariff file", width="stretch"):
                upload = st.file_uploader("Tariff (CSV / Excel)", type=["csv", "xlsx", "xls"], key="tar_upload")
                st.caption("Columns: `start`, `end`, `price_inr_per_kwh`")
                if upload is not None and ss.get("tar_upload_id") != upload.file_id:
                    try:
                        ss.tariff = _normalise_times(data.clean_tariff(_read_table(upload)), ["start", "end"])
                        ss.tar_upload_id = upload.file_id
                        _bump("tar_ver")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not read file: {exc}")

            edited = st.data_editor(
                ss.tariff, key=f"tar_editor_{ss.tar_ver}", num_rows="dynamic", width="stretch",
                column_config=TARIFF_COLUMN_CONFIG, hide_index=True,
            )
            st.caption("Preset values are illustrative – replace with the cited state-regulator ToU tariff.")
        tariff = data.clean_tariff(edited)
        with c2:
            try:
                _, cover = data.tariff_by_minute(tariff)
                if (cover == 1).all():
                    st.plotly_chart(charts.tariff_preview(tariff), width="stretch")
                else:
                    st.info("Tariff preview appears once the blocks cover the whole day without overlaps.")
            except (ValueError, KeyError, TypeError):
                st.info("Tariff preview appears once all times are valid HH:MM values.")
    return tariff


def precheck_panel(vehicles: pd.DataFrame, tariff: pd.DataFrame, site: dict, errors: list[str]) -> None:
    st.markdown("### ③ Check & run")
    if errors:
        st.error("Fix these before optimizing:\n\n" + "\n".join(f"- {e}" for e in errors))
        return
    inst = data.to_instance(vehicles, tariff, site)
    pc = data.precheck(inst)
    need_grid = float((inst.vehicles["need_kwh"] / site["eta"]).sum())
    capacity = site["p_max"] * site["horizon_h"]
    load = 100 * need_grid / capacity

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicles", len(pc), border=True)
    c2.metric("Energy required (grid)", f"{need_grid:,.0f} kWh", border=True)
    c3.metric("Site capacity over horizon", f"{capacity:,.0f} kWh", border=True)
    c4.metric("Demand / capacity", f"{load:.0f}%", border=True,
              help="Above ~70% the site is tight; above 100% some vehicles must fall short.")

    bad = pc[~pc["feasible_alone"]]
    beyond = pc[pc["beyond_horizon"]]
    if need_grid > capacity:
        st.warning("Total demand exceeds what the site can deliver in the horizon – some vehicles will fall short.")
    if len(bad):
        st.warning(f"{len(bad)} vehicle(s) cannot reach their target even with a dedicated charger (stay too short or rate too low):")
        st.dataframe(bad.drop(columns=["feasible_alone", "beyond_horizon"]), hide_index=True, width="stretch")
    if len(beyond):
        st.info(f"{len(beyond)} vehicle(s) depart after the horizon ends; they must finish by the horizon end: "
                + ", ".join(beyond["vehicle_id"]))
    if not len(bad) and need_grid <= capacity:
        st.success("Inputs look good – run the optimizer.", icon="✅")
