"""Turn raw schedules into KPIs, per-vehicle summaries and export tables."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .data import Instance


def schedule_cost(inst: Instance, P: np.ndarray) -> float:
    return float((P * inst.price[None, :]).sum() * inst.dt)


def vehicle_summary(inst: Instance, sched: dict, base: dict | None = None) -> pd.DataFrame:
    V, P, dt, eta = inst.vehicles, sched["power"], inst.dt, inst.site["eta"]
    grid_kwh = P.sum(axis=1) * dt
    cost = (P * inst.price[None, :]).sum(axis=1) * dt
    slot_td = pd.Timedelta(minutes=inst.site["slot_min"])

    starts, ends = [], []
    for row in P:
        idx = np.flatnonzero(row > 1e-6)
        starts.append(inst.times[idx[0]].strftime("%H:%M") if len(idx) else "–")
        ends.append((inst.times[idx[-1]] + slot_td).strftime("%H:%M") if len(idx) else "–")

    df = pd.DataFrame(
        {
            "vehicle_id": V["vehicle_id"],
            "type": V["type"],
            "arrival": V["arrival"],
            "departure": V["departure"],
            "need_kwh": V["need_kwh"].round(2),
            "delivered_kwh": (grid_kwh * eta).round(2),
            "shortfall_kwh": sched["shortfall"].round(2),
            "target_met": sched["shortfall"] <= 1e-3,
            "charge_start": starts,
            "charge_end": ends,
            "grid_kwh": grid_kwh.round(2),
            "cost_inr": cost.round(2),
            "avg_inr_per_kwh": np.where(grid_kwh > 0, cost / np.maximum(grid_kwh, 1e-9), np.nan).round(2),
        }
    )
    if base is not None:
        base_cost = (base["power"] * inst.price[None, :]).sum(axis=1) * dt
        df["baseline_cost_inr"] = base_cost.round(2)
        df["saving_inr"] = (base_cost - cost).round(2)
    return df


def kpis(inst: Instance, opt: dict, base: dict) -> dict:
    oc, bc = schedule_cost(inst, opt["power"]), schedule_cost(inst, base["power"])
    o_load, b_load = opt["power"].sum(axis=0), base["power"].sum(axis=0)
    n = len(inst.vehicles)
    return {
        "opt_cost": oc,
        "base_cost": bc,
        "saving": bc - oc,
        "saving_pct": 100.0 * (bc - oc) / bc if bc > 0 else 0.0,
        "opt_peak": float(o_load.max(initial=0)),
        "base_peak": float(b_load.max(initial=0)),
        "opt_served": int((opt["shortfall"] <= 1e-3).sum()),
        "base_served": int((base["shortfall"] <= 1e-3).sum()),
        "n_vehicles": n,
        "opt_shortfall_kwh": float(opt["shortfall"].sum()),
        "base_shortfall_kwh": float(base["shortfall"].sum()),
        "opt_grid_kwh": float(o_load.sum() * inst.dt),
        "base_grid_kwh": float(b_load.sum() * inst.dt),
    }


def cost_by_price_level(inst: Instance, opt: dict, base: dict) -> pd.DataFrame:
    rows = []
    for name, sched in (("Optimized", opt), ("Baseline", base)):
        load = sched["power"].sum(axis=0)
        for level in np.unique(inst.price):
            mask = inst.price == level
            kwh = float(load[mask].sum() * inst.dt)
            rows.append(
                {"schedule": name, "price_level": f"₹{level:.2f}/kWh", "price": level, "energy_kwh": kwh, "cost_inr": kwh * level}
            )
    return pd.DataFrame(rows).sort_values(["price", "schedule"])


def schedule_table(inst: Instance, P: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(P.round(3), columns=inst.slot_labels)
    df.insert(0, "vehicle_id", inst.vehicles["vehicle_id"].values)
    return df


def load_table(inst: Instance, opt: dict, base: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slot": inst.slot_labels,
            "price_inr_per_kwh": inst.price,
            "optimized_load_kw": opt["power"].sum(axis=0).round(3),
            "baseline_load_kw": base["power"].sum(axis=0).round(3),
            "site_limit_kw": inst.site["p_max"],
        }
    )


def excel_report(inst: Instance, res: dict, vehicles_in: pd.DataFrame, tariff_in: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    k = res["kpis"]
    summary = pd.DataFrame(
        [
            ("Optimized cost (INR)", round(k["opt_cost"], 2)),
            ("Baseline cost (INR)", round(k["base_cost"], 2)),
            ("Saving (INR)", round(k["saving"], 2)),
            ("Saving (%)", round(k["saving_pct"], 2)),
            ("Optimized peak load (kW)", round(k["opt_peak"], 2)),
            ("Baseline peak load (kW)", round(k["base_peak"], 2)),
            ("Vehicles fully served (optimized)", f"{k['opt_served']}/{k['n_vehicles']}"),
            ("Vehicles fully served (baseline)", f"{k['base_served']}/{k['n_vehicles']}"),
            ("Solver status", res["opt"]["status"]),
        ],
        columns=["metric", "value"],
    )
    site = pd.DataFrame(list(inst.site.items()), columns=["parameter", "value"])
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        schedule_table(inst, res["opt"]["power"]).to_excel(xw, sheet_name="Optimized schedule (kW)", index=False)
        schedule_table(inst, res["base"]["power"]).to_excel(xw, sheet_name="Baseline schedule (kW)", index=False)
        res["vehicle_summary"].to_excel(xw, sheet_name="Vehicle summary", index=False)
        load_table(inst, res["opt"], res["base"]).to_excel(xw, sheet_name="Site load", index=False)
        vehicles_in.to_excel(xw, sheet_name="Input vehicles", index=False)
        tariff_in.to_excel(xw, sheet_name="Input tariff", index=False)
        site.to_excel(xw, sheet_name="Input site", index=False)
    return buf.getvalue()
