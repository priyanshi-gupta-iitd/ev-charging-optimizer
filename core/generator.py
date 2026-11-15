"""Random but realistic vehicle scenarios (all values are assumed/synthetic)."""

import numpy as np
import pandas as pd

from .data import fmt_hhmm, parse_hhmm

VEHICLE_TYPES = {
    "e-rickshaw": dict(prefix="R", battery=(8, 10), rates=[3.3], soc_now=(10, 40), soc_target=(90, 100), dwell_h=(8, 12)),
    "car": dict(prefix="C", battery=(30, 60), rates=[7.2, 7.2, 7.2, 30.0], soc_now=(15, 50), soc_target=(80, 100), dwell_h=(4, 12)),
    "e-bus": dict(prefix="B", battery=(200, 300), rates=[60.0], soc_now=(20, 40), soc_target=(90, 100), dwell_h=(7, 10)),
}

DEFAULT_MIX = {"e-rickshaw": 40, "car": 45, "e-bus": 15}


def _round_to(x: float, step: float) -> float:
    return round(x / step) * step


def generate(
    n: int,
    mix: dict[str, float] | None = None,
    start: str = "18:00",
    horizon_h: float = 16,
    arrival_span_h: float = 4,
    seed: int = 42,
    ensure_feasible: bool = True,
    eta: float = 0.9,
) -> pd.DataFrame:
    """With ensure_feasible, each vehicle parks long enough to be charged on its own;
    shortfalls can then only come from shared site/charger limits."""
    mix = mix or DEFAULT_MIX
    names = [k for k, w in mix.items() if w > 0]
    weights = np.array([mix[k] for k in names], dtype=float)
    weights /= weights.sum()

    rng = np.random.default_rng(seed)
    types = rng.choice(names, size=n, p=weights)
    start_min = parse_hhmm(start)
    counters = {k: 0 for k in VEHICLE_TYPES}

    rows = []
    for vtype in types:
        spec = VEHICLE_TYPES[vtype]
        counters[vtype] += 1
        battery = round(float(rng.uniform(*spec["battery"])), 1)
        soc_now = int(rng.uniform(*spec["soc_now"]))
        soc_target = int(_round_to(rng.uniform(*spec["soc_target"]), 5))
        rate = float(rng.choice(spec["rates"]))

        arr_off = _round_to(rng.uniform(0, arrival_span_h) * 60, 15)
        dwell = _round_to(rng.uniform(*spec["dwell_h"]) * 60, 15)
        if ensure_feasible:
            charge_min = 60 * battery * (soc_target - soc_now) / 100 / (eta * rate)
            dwell = max(dwell, 15 * np.ceil(1.25 * charge_min / 15) + 30)
        dep_off = min(arr_off + dwell, horizon_h * 60)
        rows.append(
            {
                "vehicle_id": f"{spec['prefix']}{counters[vtype]:02d}",
                "type": vtype,
                "arrival": fmt_hhmm(start_min + arr_off),
                "departure": fmt_hhmm(start_min + dep_off),
                "battery_kwh": battery,
                "soc_now_pct": soc_now,
                "soc_target_pct": soc_target,
                "max_rate_kw": rate,
                "_arr_off": arr_off,
            }
        )
    df = pd.DataFrame(rows).sort_values("_arr_off", kind="stable")
    return df.drop(columns="_arr_off").reset_index(drop=True)
