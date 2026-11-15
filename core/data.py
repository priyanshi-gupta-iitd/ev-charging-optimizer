"""Input parsing, validation and conversion of clock times into time slots."""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

REF_DATE = pd.Timestamp("2026-01-01")
MIN_PER_DAY = 24 * 60

VEHICLE_COLUMNS = [
    "vehicle_id",
    "type",
    "arrival",
    "departure",
    "battery_kwh",
    "soc_now_pct",
    "soc_target_pct",
    "max_rate_kw",
]
VEHICLE_NUMERIC = ["battery_kwh", "soc_now_pct", "soc_target_pct", "max_rate_kw"]
TARIFF_COLUMNS = ["start", "end", "price_inr_per_kwh"]


@dataclass
class Instance:
    """A fully discretised problem instance, ready for the model and the baseline."""

    site: dict
    times: pd.DatetimeIndex  # start time of every slot
    price: np.ndarray  # INR/kWh in every slot
    vehicles: pd.DataFrame  # one row per vehicle, with slot indices a (inclusive) and d (exclusive)

    @property
    def n_slots(self) -> int:
        return len(self.times)

    @property
    def dt(self) -> float:
        return self.site["slot_min"] / 60.0

    @property
    def end_time(self) -> pd.Timestamp:
        return self.times[0] + pd.Timedelta(minutes=self.site["slot_min"] * self.n_slots)

    @property
    def slot_labels(self) -> list[str]:
        return [f"D{(t - REF_DATE).days + 1} {t:%H:%M}" for t in self.times]

    def availability(self) -> np.ndarray:
        avail = np.zeros((len(self.vehicles), self.n_slots), dtype=bool)
        for i, (a, d) in enumerate(zip(self.vehicles["a"], self.vehicles["d"])):
            avail[i, a:d] = True
        return avail


def parse_hhmm(value) -> int:
    """Return minutes after midnight for 'HH:MM', 'HH:MM:SS', datetime.time or Timestamp."""
    if isinstance(value, (_dt.time, _dt.datetime, pd.Timestamp)):
        return value.hour * 60 + value.minute
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        raise ValueError("missing time")
    parts = s.split(":")
    try:
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
    except ValueError as exc:
        raise ValueError(f"invalid time '{value}'") from exc
    if not (0 <= h <= 24 and 0 <= m < 60) or (h == 24 and m != 0):
        raise ValueError(f"invalid time '{value}'")
    return h * 60 + m


def fmt_hhmm(minutes: int) -> str:
    minutes = int(round(minutes)) % MIN_PER_DAY
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def clean_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, drop blank rows and coerce numeric columns."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(how="all")
    if "vehicle_id" in df.columns:
        df = df[df["vehicle_id"].astype(str).str.strip().replace({"nan": "", "None": ""}) != ""]
        df["vehicle_id"] = df["vehicle_id"].astype(str).str.strip()
    if "type" not in df.columns:
        df["type"] = "vehicle"
    df["type"] = df["type"].fillna("vehicle").astype(str)
    for col in VEHICLE_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def clean_tariff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(how="all")
    if "price_inr_per_kwh" in df.columns:
        df["price_inr_per_kwh"] = pd.to_numeric(df["price_inr_per_kwh"], errors="coerce")
    return df.reset_index(drop=True)


def tariff_by_minute(tariff: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Price for every minute of the day, plus how many blocks cover each minute."""
    price = np.full(MIN_PER_DAY, np.nan)
    cover = np.zeros(MIN_PER_DAY, dtype=int)
    for _, row in tariff.iterrows():
        s = parse_hhmm(row["start"]) % MIN_PER_DAY
        e = parse_hhmm(row["end"])
        e = MIN_PER_DAY if e == MIN_PER_DAY else e % MIN_PER_DAY
        spans = [(s, e)] if e > s else [(s, MIN_PER_DAY), (0, e)]
        for lo, hi in spans:
            price[lo:hi] = row["price_inr_per_kwh"]
            cover[lo:hi] += 1
    return price, cover


def validate(vehicles: pd.DataFrame, tariff: pd.DataFrame, site: dict) -> list[str]:
    """Return a list of human-readable errors. An empty list means inputs are usable."""
    errors: list[str] = []

    missing = [c for c in VEHICLE_COLUMNS if c not in vehicles.columns]
    if missing:
        errors.append(f"Vehicle table is missing columns: {', '.join(missing)}")
        return errors
    if vehicles.empty:
        errors.append("Vehicle table is empty.")
        return errors

    dup = vehicles["vehicle_id"][vehicles["vehicle_id"].duplicated()].unique()
    if len(dup):
        errors.append(f"Duplicate vehicle IDs: {', '.join(map(str, dup))}")

    for _, r in vehicles.iterrows():
        vid = r["vehicle_id"]
        for col in ("arrival", "departure"):
            try:
                parse_hhmm(r[col])
            except ValueError as exc:
                errors.append(f"{vid}: {col} – {exc}")
        for col in VEHICLE_NUMERIC:
            if pd.isna(r[col]):
                errors.append(f"{vid}: {col} is missing or not a number")
        if any(pd.isna(r[c]) for c in VEHICLE_NUMERIC):
            continue
        if r["battery_kwh"] <= 0:
            errors.append(f"{vid}: battery_kwh must be positive")
        if r["max_rate_kw"] <= 0:
            errors.append(f"{vid}: max_rate_kw must be positive")
        if not (0 <= r["soc_now_pct"] <= 100 and 0 <= r["soc_target_pct"] <= 100):
            errors.append(f"{vid}: SOC values must be between 0 and 100")
        elif r["soc_target_pct"] < r["soc_now_pct"]:
            errors.append(f"{vid}: target SOC is below current SOC")

    tmissing = [c for c in TARIFF_COLUMNS if c not in tariff.columns]
    if tmissing:
        errors.append(f"Tariff table is missing columns: {', '.join(tmissing)}")
    elif tariff.empty:
        errors.append("Tariff table is empty.")
    else:
        if tariff["price_inr_per_kwh"].isna().any():
            errors.append("Tariff has missing or non-numeric prices.")
        elif (tariff["price_inr_per_kwh"] < 0).any():
            errors.append("Tariff prices must be non-negative.")
        try:
            _, cover = tariff_by_minute(tariff)
            if (cover == 0).any():
                gap = int(np.argmax(cover == 0))
                errors.append(f"Tariff does not cover the whole day (first gap at {fmt_hhmm(gap)}).")
            if (cover > 1).any():
                ov = int(np.argmax(cover > 1))
                errors.append(f"Tariff blocks overlap (first overlap at {fmt_hhmm(ov)}).")
        except ValueError as exc:
            errors.append(f"Tariff: {exc}")

    if site["p_max"] <= 0:
        errors.append("Site power limit must be positive.")
    if site["n_chargers"] < 1:
        errors.append("At least one charger is required.")
    if not (0 < site["eta"] <= 1):
        errors.append("Efficiency must be in (0, 1].")
    return errors


def to_instance(vehicles: pd.DataFrame, tariff: pd.DataFrame, site: dict) -> Instance:
    """Discretise validated inputs into slots.

    Arrival times earlier than the horizon start are taken to be on the next day, and a
    departure at or before arrival is taken to be on the following day.
    """
    slot = int(site["slot_min"])
    start_min = parse_hhmm(site["start"])
    n = int(round(site["horizon_h"] * 60 / slot))
    t0 = REF_DATE + pd.Timedelta(minutes=start_min)
    times = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=slot * k) for k in range(n)])

    price_min, _ = tariff_by_minute(tariff)
    price = np.array([price_min[(start_min + slot * k) % MIN_PER_DAY] for k in range(n)])

    rows = []
    for _, r in vehicles.iterrows():
        arr = parse_hhmm(r["arrival"]) % MIN_PER_DAY
        dep = parse_hhmm(r["departure"]) % MIN_PER_DAY
        arr_off = (arr - start_min) % MIN_PER_DAY
        dwell = (dep - arr) % MIN_PER_DAY or MIN_PER_DAY
        dep_off = arr_off + dwell
        a = min(n, math.ceil(arr_off / slot - 1e-9))
        d = max(a, min(n, math.floor(dep_off / slot + 1e-9)))
        need = r["battery_kwh"] * max(0.0, r["soc_target_pct"] - r["soc_now_pct"]) / 100.0
        rows.append(
            {
                "vehicle_id": r["vehicle_id"],
                "type": r["type"],
                "arrival": fmt_hhmm(arr),
                "departure": fmt_hhmm(dep),
                "arr_ts": t0 + pd.Timedelta(minutes=arr_off),
                "dep_ts": t0 + pd.Timedelta(minutes=dep_off),
                "a": a,
                "d": d,
                "battery_kwh": float(r["battery_kwh"]),
                "need_kwh": float(need),
                "rmax": float(r["max_rate_kw"]),
                "beyond_horizon": dep_off > n * slot,
            }
        )
    return Instance(site=dict(site), times=times, price=price, vehicles=pd.DataFrame(rows))


def precheck(inst: Instance) -> pd.DataFrame:
    """Per-vehicle feasibility ignoring interactions with other vehicles."""
    v = inst.vehicles
    eta, dt = inst.site["eta"], inst.dt
    max_kwh = eta * dt * v["rmax"] * (v["d"] - v["a"])
    out = pd.DataFrame(
        {
            "vehicle_id": v["vehicle_id"],
            "need_kwh": v["need_kwh"].round(2),
            "max_deliverable_kwh": max_kwh.round(2),
            "parked_slots": v["d"] - v["a"],
            "feasible_alone": max_kwh + 1e-6 >= v["need_kwh"],
            "beyond_horizon": v["beyond_horizon"],
        }
    )
    return out
