"""'Charge immediately' baseline.

Each slot, vehicles that are parked and not yet full are served first-come-first-served
at their maximum rate, while respecting the number of chargers and the site power limit.
"""

import numpy as np

from .data import Instance


def simulate(inst: Instance) -> dict:
    V = inst.vehicles
    T = inst.n_slots
    dt, eta = inst.dt, inst.site["eta"]
    p_max, n_chargers = inst.site["p_max"], inst.site["n_chargers"]

    remaining = V["need_kwh"].to_numpy(dtype=float).copy()
    order = np.lexsort((V.index.to_numpy(), V["arr_ts"].to_numpy()))
    P = np.zeros((len(V), T))

    for t in range(T):
        cap, used = p_max, 0
        for i in order:
            if used >= n_chargers or cap <= 1e-9:
                break
            if not (V.at[i, "a"] <= t < V.at[i, "d"]) or remaining[i] <= 1e-9:
                continue
            pw = min(V.at[i, "rmax"], remaining[i] / (eta * dt), cap)
            P[i, t] = pw
            cap -= pw
            remaining[i] -= pw * eta * dt
            used += 1

    remaining[remaining < 1e-4] = 0.0
    return {"status": "Simulated", "ok": True, "power": P, "shortfall": remaining}
