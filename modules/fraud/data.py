"""Fraud module — transaction data.

Generates a labelled transaction stream with the structure the real problem has: a small
fraudulent minority, velocity features that only exist once you look at a *window*, and a
class imbalance severe enough that accuracy is meaningless.

The real version of this body reads from Redis Streams or a CDC feed. What matters here is
that the downstream pipeline — windowed features, SMOTE, streaming scoring — is identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES: tuple[str, ...] = (
    "amount",
    "hour_of_day",
    "distance_from_home_km",
    "txn_count_1h",
    "distinct_merchants_24h",
    "card_age_days",
    "avg_amount_30d",
    "is_night",
    "is_foreign",
    "pin_used",
)

TARGET = "is_fraud"

#: Northwind Pay's ledger keys. ``customer_id`` is deliberately drawn from the *same* space as
#: the churn book (Northwind has ~3,333 subscription customers; cardholders are those same
#: people), so the two bodies join. That is a design decision, not a coincidence, and it is why
#: the hub agent can answer a question that spans retention and fraud in one call.
AUTH_KEY = "auth_id"
CUSTOMER_KEY = "customer_id"

#: Size of the shared customer base — matches modules/churn's postal extract.
CUSTOMER_BASE = 3333


def synthesize(n: int = 20000, fraud_rate: float = 0.012, seed: int = 7) -> pd.DataFrame:
    """Generate a transaction stream that is genuinely hard to separate.

    An earlier version made every fraudulent transaction larger than every legitimate one,
    which produced a perfect PR-AUC of 1.000 — a result that looks excellent and means
    nothing. Real fraud overlaps the legitimate distribution in both directions:

    - some fraud is **account takeover**: normal amounts, aged cards, PIN used, daytime;
    - some legitimate activity is **inherently suspicious**: travellers abroad, large
      purchases, late-night activity.

    Without that overlap the body cannot demonstrate the thing it exists to demonstrate.

    Note the flat-index style throughout: ``arr[mask_a][mask_b] = x`` writes to a temporary
    copy and silently discards the assignment, which is exactly how the first version of this
    generator ended up with no overlap at all.
    """
    rng = np.random.default_rng(seed)

    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    idx_fraud = np.where(is_fraud == 1)[0]
    idx_legit = np.where(is_fraud == 0)[0]
    n_f, n_l = len(idx_fraud), len(idx_legit)

    amount = np.empty(n)
    hour = np.empty(n)
    distance = np.empty(n)
    card_age = np.empty(n)
    avg_amount = np.empty(n)
    pin_used = np.empty(n)

    # --- legitimate behaviour, including the part that looks suspicious ---------
    amount[idx_legit] = np.clip(rng.lognormal(3.2, 1.0, n_l), 1, 8000)
    hour[idx_legit] = rng.integers(0, 24, n_l).astype(float)
    distance[idx_legit] = np.clip(rng.gamma(1.5, 8.0, n_l), 0, 600)
    card_age[idx_legit] = rng.integers(30, 2500, n_l).astype(float)
    avg_amount[idx_legit] = np.clip(rng.lognormal(3.2, 0.8, n_l), 5, 5000)
    pin_used[idx_legit] = (rng.random(n_l) < 0.85).astype(float)

    travellers = idx_legit[rng.random(n_l) < 0.04]
    distance[travellers] = np.clip(rng.gamma(3.0, 900.0, len(travellers)), 900, 14000)

    whales = idx_legit[rng.random(n_l) < 0.02]
    amount[whales] = avg_amount[whales] * rng.uniform(8, 25, len(whales))

    night_legit = idx_legit[rng.random(n_l) < 0.08]
    hour[night_legit] = rng.integers(0, 5, len(night_legit)).astype(float)

    # --- fraudulent behaviour, in two distinct modes ---------------------------
    # Mode A (~60%): card testing / stolen numbers. Loud.
    # Mode B (~40%): account takeover. Quiet, aged card, normal amounts, PIN known.
    mode_b_mask = rng.random(n_f) < 0.40
    idx_fb = idx_fraud[mode_b_mask]
    idx_fa = idx_fraud[~mode_b_mask]

    amount[idx_fa] = np.clip(rng.lognormal(5.4, 1.2, len(idx_fa)), 20, 20000)
    amount[idx_fb] = np.clip(rng.lognormal(3.4, 0.7, len(idx_fb)), 5, 2000)

    # Card testing skews to 3am; account takeover happens in normal hours.
    hour[idx_fa] = np.where(
        rng.random(len(idx_fa)) < 0.7,
        rng.integers(0, 5, len(idx_fa)),
        rng.integers(0, 24, len(idx_fa)),
    ).astype(float)
    hour[idx_fb] = rng.integers(7, 23, len(idx_fb)).astype(float)

    distance[idx_fa] = np.clip(rng.gamma(3.0, 120.0, len(idx_fa)), 0, 14000)
    distance[idx_fb] = np.clip(rng.gamma(1.5, 10.0, len(idx_fb)), 0, 200)

    card_age[idx_fa] = rng.integers(1, 400, len(idx_fa)).astype(float)
    card_age[idx_fb] = rng.integers(400, 2500, len(idx_fb)).astype(float)

    avg_amount[idx_fraud] = np.clip(rng.lognormal(3.2, 0.8, n_f), 5, 5000)

    pin_used[idx_fa] = (rng.random(len(idx_fa)) < 0.2).astype(float)
    pin_used[idx_fb] = (rng.random(len(idx_fb)) < 0.8).astype(float)  # the attacker has the PIN

    # --- velocity: the strongest signal, and invisible without a window --------
    txn_count_1h = np.clip(rng.poisson(1.3, n), 0, 20).astype(float)
    distinct_merchants_24h = np.clip(rng.poisson(2.6, n), 1, 25).astype(float)

    txn_count_1h[idx_fa] = np.clip(rng.poisson(9.5, len(idx_fa)), 1, 60).astype(float)
    distinct_merchants_24h[idx_fa] = np.clip(rng.poisson(12.0, len(idx_fa)), 1, 60).astype(float)
    txn_count_1h[idx_fb] = np.clip(rng.poisson(3.2, len(idx_fb)), 1, 15).astype(float)
    distinct_merchants_24h[idx_fb] = np.clip(rng.poisson(4.5, len(idx_fb)), 1, 20).astype(float)

    is_night = ((hour < 6) | (hour >= 23)).astype(float)
    is_foreign = (distance > 800).astype(float)

    df = pd.DataFrame(
        {
            "amount": np.round(amount, 2),
            "hour_of_day": hour,
            "distance_from_home_km": np.round(distance, 1),
            "txn_count_1h": txn_count_1h,
            "distinct_merchants_24h": distinct_merchants_24h,
            "card_age_days": card_age,
            "avg_amount_30d": np.round(avg_amount, 2),
            "is_night": is_night,
            "is_foreign": is_foreign,
            "pin_used": pin_used,
            TARGET: is_fraud,
        }
    )
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    from core.tenant import auth_id, customer_id

    df[AUTH_KEY] = [auth_id(i) for i in range(len(df))]
    # Cardholders are the same people as the subscribers in the churn book: same key space,
    # so a churn score and this person's authorisations can be joined on customer_id.
    df[CUSTOMER_KEY] = [customer_id(i % CUSTOMER_BASE) for i in range(len(df))]
    return df


def amount_anomaly_ratio(df: pd.DataFrame) -> float:
    """How many fraudulent transactions are >10x the customer's own 30-day average."""
    f = df[df[TARGET] == 1]
    if f.empty:
        return 0.0
    return float((f["amount"] > f["avg_amount_30d"] * 10).mean())
