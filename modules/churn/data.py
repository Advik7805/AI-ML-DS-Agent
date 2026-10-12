"""Churn module — data loading.

Two sources, one schema:

1. **Real**: ``data/raw/telco_churn.csv`` — the classic 3,333-row telecom churn dataset
   (state, area_code, international/voice-mail plans, day/eve/night/intl usage, service
   calls, churn label). Fetched into the repo from a committed copy on GitHub.
2. **Synthetic**: :func:`synthesize` reproduces the same schema with the same correlation
   structure, so every pipeline runs on a clean machine with no download.

The synthetic path is *not* the demo data — it is the fallback that keeps CI and a fresh
clone working. Real data wins whenever the file is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
RAW_CHURN_CSV = DATA_ROOT / "raw" / "telco_churn.csv"

TARGET = "churn"

#: The join key. Everything Northwind knows about a subscriber hangs off this, and it is the
#: same key space the fraud body uses — which is what makes cross-body questions ("this
#: customer is leaving *and* their card is being tested") answerable in SQL rather than in prose.
CUSTOMER_KEY = "customer_id"

#: Numeric features. Note we deliberately **exclude** the four ``*_charge`` columns: in this
#: dataset charge == minutes * rate, so they are deterministic functions of the minute
#: columns. Keeping them would give the model perfect collinearity and inflate importance.
NUMERIC_FEATURES: tuple[str, ...] = (
    "account_length",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_eve_minutes",
    "total_eve_calls",
    "total_night_minutes",
    "total_night_calls",
    "total_intl_minutes",
    "total_intl_calls",
    "number_customer_service_calls",
)

#: Binary yes/no features.
BINARY_FEATURES: tuple[str, ...] = ("international_plan", "voice_mail_plan")

#: High-cardinality categoricals. Kept, but see ``train.py`` for how they are handled.
CATEGORICAL_FEATURES: tuple[str, ...] = ("state", "area_code")

FEATURES: tuple[str, ...] = NUMERIC_FEATURES + BINARY_FEATURES

SourceKind = Literal["real", "synthetic"]


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """The raw CSV stores every value as a quoted string. Cast to real dtypes."""
    df = df.copy()
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Drop the unnamed leading index column if present.
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.strip('"').str.lower().map({"yes": 1, "no": 0})

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.strip('"')

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(str).str.strip().str.strip('"').str.lower().map({"yes": 1, "no": 0})

    # Attach the tenant key after every reshape, so a row dropped during cleaning cannot
    # silently shift every id below it.
    from core.tenant import customer_id

    df = df.reset_index(drop=True)
    df[CUSTOMER_KEY] = [customer_id(i) for i in range(len(df))]

    return df


def load_telco(path: Path | str = RAW_CHURN_CSV) -> tuple[pd.DataFrame, SourceKind]:
    """Load the real dataset when available, else synthesize. Never raises for a missing file."""
    p = Path(path)
    if p.exists():
        df = _coerce(pd.read_csv(p))
        missing = [c for c in (*FEATURES, TARGET) if c not in df.columns]
        if missing:
            raise ValueError(f"{p.name} is missing required columns: {', '.join(missing)}")
        df = df.dropna(subset=[*FEATURES, TARGET]).reset_index(drop=True)
        return df, "real"
    return synthesize(), "synthetic"


def synthesize(n: int = 3333, seed: int = 42) -> pd.DataFrame:
    """Generate a churn dataset with the same schema and correlation structure.

    Churn is driven by the levers that actually drive it in the real data: an international
    plan without many international calls, few voice-mail messages, high day usage, and many
    customer-service calls.
    """
    rng = np.random.default_rng(seed)

    states = rng.choice(["KS", "OH", "NJ", "OK", "AL", "MA", "MO", "LA", "WV", "IN"], size=n)
    area_codes = rng.choice(["area_code_415", "area_code_408", "area_code_510", "area_code_358"], size=n, p=[0.25, 0.25, 0.25, 0.25])

    international_plan = (rng.random(n) < 0.096).astype(int)
    voice_mail_plan = (rng.random(n) < 0.28).astype(int)
    number_vmail_messages = np.where(voice_mail_plan == 1, rng.integers(15, 52, n), 0)

    account_length = rng.integers(1, 244, n).astype(float)
    total_day_minutes = np.clip(rng.normal(180, 54, n), 0, None)
    total_eve_minutes = np.clip(rng.normal(200, 51, n), 0, None)
    total_night_minutes = np.clip(rng.normal(200, 51, n), 0, None)
    total_intl_minutes = np.clip(rng.normal(10.2, 2.8, n), 0, None)

    total_day_calls = rng.integers(70, 165, n).astype(float)
    total_eve_calls = rng.integers(70, 170, n).astype(float)
    total_night_calls = rng.integers(70, 170, n).astype(float)
    total_intl_calls = rng.integers(0, 20, n).astype(float)

    # Heavy service-call tail — the strongest churn signal in the real data.
    number_customer_service_calls = np.where(
        rng.random(n) < 0.20, rng.integers(4, 9, n), rng.integers(0, 4, n)
    ).astype(float)

    logit = (
        -2.4
        + 1.65 * international_plan
        - 0.55 * (international_plan * total_intl_minutes) / 5.0
        - 0.045 * number_vmail_messages
        + 0.30 * number_customer_service_calls
        + 0.006 * total_day_minutes
        - 0.002 * total_eve_minutes
        + 0.001 * account_length * 0.01
        + rng.normal(0, 0.55, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "state": states,
            "account_length": account_length,
            "area_code": area_codes,
            "international_plan": international_plan,
            "voice_mail_plan": voice_mail_plan,
            "number_vmail_messages": number_vmail_messages.astype(float),
            "total_day_minutes": np.round(total_day_minutes, 1),
            "total_day_calls": total_day_calls,
            "total_eve_minutes": np.round(total_eve_minutes, 1),
            "total_eve_calls": total_eve_calls,
            "total_night_minutes": np.round(total_night_minutes, 1),
            "total_night_calls": total_night_calls,
            "total_intl_minutes": np.round(total_intl_minutes, 1),
            "total_intl_calls": total_intl_calls,
            "number_customer_service_calls": number_customer_service_calls,
            TARGET: churn,
        }
    )
    # Mirror the real file's derived columns so the schemas match exactly.
    df["total_day_charge"] = np.round(df["total_day_minutes"] * 0.17, 2)
    df["total_eve_charge"] = np.round(df["total_eve_minutes"] * 0.085, 2)
    df["total_night_charge"] = np.round(df["total_night_minutes"] * 0.045, 2)
    df["total_intl_charge"] = np.round(df["total_intl_minutes"] * 0.27, 2)
    from core.tenant import customer_id

    df[CUSTOMER_KEY] = [customer_id(i) for i in range(len(df))]
    return df
