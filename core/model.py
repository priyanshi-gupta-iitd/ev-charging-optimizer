"""Cost-minimising EV charging schedule as an LP / MILP (PuLP + CBC).

Decision variables
    p[v,t] >= 0   grid power to vehicle v in slot t (kW), only while v is parked
    y[v,t] in {0,1}  vehicle v occupies a charger in slot t (MILP only)
    u[v]   >= 0   unmet energy of vehicle v (kWh), heavily penalised

Objective
    min  sum_{v,t} c_t * p[v,t] * dt  +  M * sum_v u[v]

Constraints
    eta * dt * sum_t p[v,t] + u[v] = E_v           energy requirement
    p[v,t] <= R_v * y[v,t]                          rate limit / charger linking
    sum_v y[v,t] <= N                               charger count
    sum_v p[v,t] <= P_max                           site power limit

In LP mode the charger constraint is relaxed to sum_v p[v,t] / R_v <= N.
"""

from __future__ import annotations

import time

import numpy as np
import pulp

from .data import Instance

TIE_BREAK = 1e-6  # nudges otherwise-equal schedules to charge earlier


def solve(inst: Instance, milp: bool = True, time_limit: int = 30, gap: float = 0.005) -> dict:
    V = inst.vehicles
    T = inst.n_slots
    dt, eta = inst.dt, inst.site["eta"]
    price = inst.price
    penalty = max(1000.0, 100.0 * float(price.max()) / eta)

    prob = pulp.LpProblem("ev_charging", pulp.LpMinimize)
    p: dict[tuple[int, int], pulp.LpVariable] = {}
    y: dict[tuple[int, int], pulp.LpVariable] = {}
    u: dict[int, pulp.LpVariable] = {}

    for i, row in V.iterrows():
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=0, upBound=row["need_kwh"])
        if row["need_kwh"] <= 0:
            continue
        for t in range(row["a"], row["d"]):
            p[i, t] = pulp.LpVariable(f"p_{i}_{t}", lowBound=0, upBound=row["rmax"])
            if milp:
                y[i, t] = pulp.LpVariable(f"y_{i}_{t}", cat=pulp.LpBinary)

    energy_cost = pulp.lpSum(price[t] * dt * var for (i, t), var in p.items())
    prob += (
        energy_cost
        + penalty * pulp.lpSum(u.values())
        + TIE_BREAK * pulp.lpSum(t * var for (i, t), var in p.items())
    )

    by_vehicle: dict[int, list] = {i: [] for i in V.index}
    by_slot: dict[int, list] = {t: [] for t in range(T)}
    for (i, t), var in p.items():
        by_vehicle[i].append(var)
        by_slot[t].append((i, var))

    for i, row in V.iterrows():
        prob += eta * dt * pulp.lpSum(by_vehicle[i]) + u[i] == row["need_kwh"], f"energy_{i}"

    for t, items in by_slot.items():
        if not items:
            continue
        prob += pulp.lpSum(var for _, var in items) <= inst.site["p_max"], f"site_{t}"
        if milp:
            for i, var in items:
                prob += var <= V.at[i, "rmax"] * y[i, t], f"link_{i}_{t}"
            prob += pulp.lpSum(y[i, t] for i, _ in items) <= inst.site["n_chargers"], f"chargers_{t}"
        else:
            prob += (
                pulp.lpSum(var / V.at[i, "rmax"] for i, var in items) <= inst.site["n_chargers"],
                f"chargers_{t}",
            )

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=gap if milp else None)
    t0 = time.perf_counter()
    prob.solve(solver)
    elapsed = time.perf_counter() - t0

    optimal = "Optimal" if not milp or gap <= 0 else f"Optimal (within {100 * gap:g}% gap)"
    status = {1: optimal, 2: "Feasible (time limit reached)"}.get(prob.sol_status, pulp.LpStatus[prob.status])
    has_solution = prob.sol_status in (1, 2)

    P = np.zeros((len(V), T))
    if has_solution:
        for (i, t), var in p.items():
            P[i, t] = max(0.0, var.value() or 0.0)
        P[P < 1e-6] = 0.0
    shortfall = np.array([max(0.0, u[i].value() or 0.0) if has_solution else V.at[i, "need_kwh"] for i in V.index])
    shortfall[shortfall < 1e-4] = 0.0

    return {
        "status": status,
        "ok": has_solution,
        "power": P,
        "shortfall": shortfall,
        "solve_time": elapsed,
        "n_vars": len(prob.variables()),
        "n_cons": len(prob.constraints),
        "milp": milp,
    }
