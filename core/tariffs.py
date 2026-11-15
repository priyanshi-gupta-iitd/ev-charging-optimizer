"""Preset time-of-use tariffs.

The values below are ILLUSTRATIVE placeholders with a realistic Indian ToU shape.
Replace them with the published EV / ToU tariff of the chosen state regulator and
cite the order in the final report.
"""

import pandas as pd

PRESETS: dict[str, pd.DataFrame] = {
    "Illustrative ToU (night off-peak, evening peak)": pd.DataFrame(
        [
            ("00:00", "06:00", 4.00),
            ("06:00", "09:00", 6.00),
            ("09:00", "17:00", 5.00),
            ("17:00", "23:00", 8.50),
            ("23:00", "24:00", 4.00),
        ],
        columns=["start", "end", "price_inr_per_kwh"],
    ),
    "Illustrative ToU (solar-hour discount)": pd.DataFrame(
        [
            ("00:00", "06:00", 5.50),
            ("06:00", "10:00", 6.50),
            ("10:00", "16:00", 3.50),
            ("16:00", "22:00", 9.00),
            ("22:00", "24:00", 6.00),
        ],
        columns=["start", "end", "price_inr_per_kwh"],
    ),
    "Illustrative 2-block (night discount)": pd.DataFrame(
        [("22:00", "06:00", 4.00), ("06:00", "22:00", 7.00)],
        columns=["start", "end", "price_inr_per_kwh"],
    ),
    "Flat tariff": pd.DataFrame(
        [("00:00", "24:00", 6.00)], columns=["start", "end", "price_inr_per_kwh"]
    ),
}

DEFAULT_PRESET = next(iter(PRESETS))


def get(name: str) -> pd.DataFrame:
    return PRESETS[name].copy()
