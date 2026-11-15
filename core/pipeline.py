"""End-to-end run: inputs -> instance -> optimized + baseline -> results."""

import pandas as pd

from . import baseline, metrics, model
from .data import to_instance


def run(vehicles: pd.DataFrame, tariff: pd.DataFrame, site: dict, milp: bool = True, time_limit: int = 30, gap: float = 0.005) -> dict:
    inst = to_instance(vehicles, tariff, site)
    opt = model.solve(inst, milp=milp, time_limit=time_limit, gap=gap)
    base = baseline.simulate(inst)
    return {
        "inst": inst,
        "opt": opt,
        "base": base,
        "kpis": metrics.kpis(inst, opt, base),
        "vehicle_summary": metrics.vehicle_summary(inst, opt, base),
    }
