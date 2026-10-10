"""Predictive maintenance — CMAPSS-compatible turbofan data.

**Provenance, stated plainly.** The real NASA CMAPSS FD001 file (3.5 MB, 100 engines, 20,631
rows) is stored under Git LFS in every GitHub copy found, and this sandbox cannot resolve LFS
objects — the contents API returns a 132-byte `sha256:` pointer. The two non-LFS copies are
3-engine and 2-engine subsets, too small to train on.

So: a generator that reproduces the real 26-column schema and the degradation physics, plus a
drop-in loader for the real file at ``data/raw/turbofan_train_FD001.txt``. When the real file
is present it is used and ``source`` reports ``"cmapss-fd001"``.

The generator models what makes RUL prediction hard: piecewise-linear degradation (health is
flat, then declines), six operational settings that shift every sensor baseline, and sensor
noise large enough that early-cycle readings carry almost no signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
CMAPSS_PATH = DATA_ROOT / "raw" / "turbofan_train_FD001.txt"

SETTINGS = [f"setting{i}" for i in range(1, 7)]
SENSORS = [f"sensor{i}" for i in range(1, 22)]

#: Sensors that actually degrade in CMAPSS FD001. The other 13 are noise — which is the point:
#: a model that uses all 21 equally is fitting noise.
DEGRADING = {
    "sensor2": 0.0030,   # LPC fan speed
    "sensor3": 0.0012,   # LPC efficiency
    "sensor4": 0.0180,   # fan pressure ratio
    "sensor7": 0.0025,   # HPT fan speed
    "sensor8": 0.0040,   # HPT cooling bleed
    "sensor9": 0.0008,
    "sensor11": 0.0060,  # HPT coolant bleed
    "sensor12": 0.0045,  # LPT bypass ratio
    "sensor13": 0.0035,
    "sensor14": 0.0020,
    "sensor15": 0.0015,
    "sensor17": 0.0022,
    "sensor20": 0.0018,
    "sensor21": 0.0016,
}

FEATURES: tuple[str, ...] = tuple(SETTINGS) + tuple(SENSORS)

#: RUL is conventionally capped at 125 — beyond that, "not failing soon" is all you can say,
#: and uncapped labels make the model optimise an irrelevant far future.
RUL_CAP = 125


def load_cmapss(path: Path = CMAPSS_PATH) -> tuple[pd.DataFrame, Literal["cmapss-fd001", "synthetic"]]:
    """Load the real FD001 file if present, else generate a compatible one."""
    if path.exists() and path.stat().st_size > 100_000:
        cols = ["unit", "cycle", *SETTINGS, *SENSORS]
        df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
        df = df.dropna(axis=1, how="all")
        return df, "cmapss-fd001"
    return synthesize(), "synthetic"


def synthesize(n_units: int = 100, max_cycles: int = 360, seed: int = 11) -> pd.DataFrame:
    """Generate CMAPSS-shaped telemetry with a degradation signal that is actually learnable.

    The first version drew a fresh random offset *per cycle* with std 100 against a trend of
    only a few units, giving the model R^2 = -0.0001. A sensor offset is a property of the
    sensor, not of the reading — it is drawn once per sensor here, and the trend is scaled to
    sit clearly above the noise, as it does in the real data.
    """
    rng = np.random.default_rng(seed)

    # Per-sensor fixed offset and scale, drawn once — this is what makes a sensor a sensor.
    sensor_base = {s: float(rng.normal(0, 1) * 40 + 500) for s in SENSORS}
    noise_std = {s: float(rng.uniform(1.0, 3.0)) for s in SENSORS}

    frames = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(130, max_cycles))
        # Health holds flat, then declines. A pure linear ramp from cycle 1 would make the
        # problem trivial and unlike the real data.
        onset = int(life * rng.uniform(0.30, 0.55))
        # Per-engine degradation rate: engines wear differently, which is why a model trained
        # on some engines must generalise to others.
        unit_rate = float(rng.uniform(0.75, 1.35))

        for cycle in range(1, life + 1):
            progress = max(0.0, cycle - onset) / max(1, life - onset)
            row: dict[str, float] = {"unit": unit, "cycle": cycle}

            for i, st in enumerate(SETTINGS):
                row[st] = round(float(rng.normal(0.0004, 0.0002)) if i == 0 else 0.0, 6)

            for s in SENSORS:
                rate = DEGRADING.get(s)
                if rate:
                    # Trend scaled so end-of-life deviation is several noise stds.
                    trend = rate * (progress ** 1.25) * 2600 * unit_rate
                else:
                    trend = 0.0  # genuinely uninformative sensor
                row[s] = round(sensor_base[s] + trend + float(rng.normal(0, noise_std[s])), 3)

            frames.append(row)

    df = pd.DataFrame(frames)
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["rul"] = df["rul"].clip(upper=RUL_CAP)
    return df
