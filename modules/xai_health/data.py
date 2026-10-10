"""Explainable diagnostics — diabetes data.

Real dataset: the Pima Indians diabetes study, 200-row sample, fetched by ``data/fetch.sh``.
Falls back to a synthetic generator with the same columns if the file is absent.

**On the protected attribute.** This dataset carries no race or gender field. Age band is used
as the grouping variable so the Fairlearn audit has something real to measure. It is a proxy,
and the module says so everywhere it reports a number — a bias audit against a made-up group
would be worse than no audit, because it would look like evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DIABETES_CSV = DATA_ROOT / "raw" / "diabetes.csv"

FEATURES: tuple[str, ...] = (
    "pregnancies",
    "plasma_glucose",
    "blood_pressure",
    "triceps_skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
)

#: The source file uses spaces in its headers; normalise once, here.
_RENAME = {
    "pregnancies": "pregnancies",
    "plasma glucose": "plasma_glucose",
    "blood pressure": "blood_pressure",
    "triceps skin thickness": "triceps_skin_thickness",
    "insulin": "insulin",
    "bmi": "bmi",
    "diabetes pedigree": "diabetes_pedigree",
    "age": "age",
    "diabetes": "diabetes",
}

TARGET = "diabetes"

#: Zero is used as a missing-value sentinel in this dataset. Treating it as a real reading
#: (a BMI of 0, a blood pressure of 0) would be a serious error.
ZERO_IS_MISSING = ("plasma_glucose", "blood_pressure", "triceps_skin_thickness", "insulin", "bmi")


def age_band(age: float) -> str:
    if age < 30:
        return "under_30"
    if age < 50:
        return "30_to_49"
    return "50_plus"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c.strip().lower(): c.strip().lower() for c in df.columns})
    df = df.rename(columns=_RENAME)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return df


def load_diabetes(path: Path = DIABETES_CSV) -> tuple[pd.DataFrame, Literal["pima", "synthetic"]]:
    if path.exists():
        df = _normalise(pd.read_csv(path))
        df = df.dropna(subset=[*FEATURES, TARGET]).reset_index(drop=True)
        # Replace the zero sentinel with NaN so imputation handles it as missing, not as a value.
        for c in ZERO_IS_MISSING:
            df.loc[df[c] == 0, c] = np.nan
        df["group"] = df["age"].apply(age_band)
        return df, "pima"
    return synthesize(), "synthetic"


def synthesize(n: int = 768, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 81, n).astype(float)
    bmi = np.clip(rng.normal(32, 7, n), 15, 65)
    glucose = np.clip(rng.normal(110, 30, n) + 0.55 * (bmi - 32) * 3, 44, 200)
    insulin = np.clip(rng.normal(120, 80, n) + 0.8 * (glucose - 110), 14, 850)
    pregnancies = np.clip(rng.poisson(3.2, n), 0, 17).astype(float)
    bp = np.clip(rng.normal(72, 12, n), 24, 122)
    skin = np.clip(rng.normal(29, 10, n), 7, 99)
    pedigree = np.clip(rng.gamma(2.0, 0.24, n), 0.08, 2.4)

    logit = (
        -7.2
        + 0.035 * glucose
        + 0.045 * bmi
        + 0.0009 * insulin
        + 0.9 * pedigree
        + 0.022 * age
        + 0.06 * pregnancies
        + rng.normal(0, 0.9, n)
    )
    diabetes = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "pregnancies": pregnancies,
            "plasma_glucose": np.round(glucose, 1),
            "blood_pressure": np.round(bp, 1),
            "triceps_skin_thickness": np.round(skin, 1),
            "insulin": np.round(insulin, 1),
            "bmi": np.round(bmi, 1),
            "diabetes_pedigree": np.round(pedigree, 3),
            "age": age,
            TARGET: diabetes,
        }
    )
    # Inject the same missing-value pattern the real file has.
    for c in ZERO_IS_MISSING:
        mask = rng.random(n) < 0.08
        df.loc[mask, c] = np.nan
    df["group"] = df["age"].apply(age_band)
    return df
