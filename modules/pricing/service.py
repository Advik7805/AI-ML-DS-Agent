"""Dynamic Pricing — a Saros body.

Estimates own-price elasticity from a transaction history and recommends a price that
maximises revenue subject to a demand floor.

Two things this body refuses to do, because both are how pricing models cause real damage:

1. **Extrapolate outside the observed price range.** Elasticity estimated between $18 and $22
   says nothing about $40, so recommendations are clipped to the observed support.
2. **Ignore the confidence interval.** A wide interval means "we do not know", and the
   recommendation says so instead of returning a confident number.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.contract import Module
from core.types import AgentTool, Explanation, MetricsReport, ModuleManifest, Prediction, UploadAnalysis
from core.upload_utils import MAX_RETURNED_PREDICTIONS, coerce_numeric_columns, read_csv_table

ACCENT = "#34d399"


def synthesize_demand(n_skus: int = 24, n_obs: int = 400, seed: int = 3) -> pd.DataFrame:
    """Generate point-of-sale history with heterogeneous, realistic elasticities.

    Elasticities are drawn in the range retail actually sees (-0.6 for staples to -2.8 for
    discretionary), and price varies within a narrow band per SKU — which is what makes the
    estimation problem honest. A SKU whose price never moves has no identifiable elasticity,
    and the model must say so rather than invent one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sku in range(n_skus):
        sku_id = f"SKU-{sku:03d}"
        base_price = float(np.round(rng.uniform(8, 120), 2))
        base_demand = float(rng.uniform(40, 900))
        true_elasticity = float(rng.uniform(-2.8, -0.6))
        # Some SKUs barely change price — their elasticity is not identifiable.
        price_spread = float(rng.choice([0.02, 0.05, 0.12, 0.22]))

        for _ in range(n_obs):
            price = float(np.round(base_price * rng.uniform(1 - price_spread, 1 + price_spread), 2))
            log_ratio = np.log(price / base_price)
            season = 1 + 0.12 * np.sin(rng.uniform(0, 2 * np.pi))
            noise = float(rng.lognormal(0, 0.16))
            demand = base_demand * (price / base_price) ** true_elasticity * season * noise
            rows.append(
                {
                    "sku": sku_id,
                    "price": price,
                    "demand": int(max(0, round(demand))),
                    "true_elasticity": true_elasticity,
                }
            )
    return pd.DataFrame(rows)


@dataclass
class ElasticityModel:
    coefficients: dict[str, float] = field(default_factory=dict)
    std_errors: dict[str, float] = field(default_factory=dict)
    r2: dict[str, float] = field(default_factory=dict)
    price_support: dict[str, tuple[float, float]] = field(default_factory=dict)
    n_obs: dict[str, int] = field(default_factory=dict)
    intercepts: dict[str, float] = field(default_factory=dict)
    mean_demand: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "ElasticityModel":
        """Log-log regression per SKU: log(demand) = a + e * log(price)."""
        self.coefficients, self.std_errors, self.r2 = {}, {}, {}
        self.price_support, self.n_obs, self.intercepts, self.mean_demand = {}, {}, {}, {}

        for sku, g in df.groupby("sku"):
            g = g[(g["price"] > 0) & (g["demand"] > 0)]
            if len(g) < 10:
                continue
            x = np.log(g["price"].to_numpy()).reshape(-1, 1)
            y = np.log(g["demand"].to_numpy())
            reg = LinearRegression().fit(x, y)
            resid = y - reg.predict(x)
            dof = max(1, len(g) - 2)
            se = float(np.sqrt((resid @ resid) / dof / max(1e-12, ((x - x.mean()) ** 2).sum())))
            self.coefficients[sku] = float(reg.coef_[0])
            self.intercepts[sku] = float(reg.intercept_)
            self.std_errors[sku] = se
            self.r2[sku] = float(reg.score(x, y))
            self.price_support[sku] = (float(g["price"].min()), float(g["price"].max()))
            self.n_obs[sku] = int(len(g))
            self.mean_demand[sku] = float(g["demand"].mean())
        return self


class PricingModule(Module):
    id = "pricing"

    def __init__(self) -> None:
        self._model: ElasticityModel | None = None
        self._df: pd.DataFrame | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> ElasticityModel:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._df = synthesize_demand()
                    self._model = ElasticityModel().fit(self._df)
        return self._model

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            _ = self.model
        assert self._df is not None
        return self._df

    def analyze_upload(
        self,
        filename: str | None,
        content: bytes | None,
        *,
        session_id: str | None = None,
        query: str = "",
    ) -> UploadAnalysis:
        del session_id, query
        if not filename or not filename.lower().endswith(".csv") or content is None:
            raise ValueError("Pricing analysis requires a CSV with sku, price, and demand columns.")
        required = ["sku", "price", "demand"]
        frame = read_csv_table(
            content,
            filename,
            module_id=self.id,
            expected_columns=required,
        )
        frame = coerce_numeric_columns(frame, ("price", "demand"))
        frame["sku"] = frame["sku"].astype("string").str.strip()
        if frame["sku"].isna().any() or (frame["sku"] == "").any():
            raise ValueError("Pricing CSV column 'sku' cannot contain blank values.")

        original_rows = len(frame)
        usable = frame.dropna(subset=["price", "demand"])
        usable = usable[(usable["price"] > 0) & (usable["demand"] > 0)].copy()
        n_skus = usable["sku"].nunique()
        if n_skus > 500:
            raise ValueError("Pricing uploads are capped at 500 distinct SKUs per analysis.")
        if usable.empty:
            raise ValueError("Pricing CSV needs positive price and demand values for at least 10 rows per SKU.")

        uploaded_model = ElasticityModel().fit(usable[required])
        if not uploaded_model.coefficients:
            raise ValueError(
                "No SKU has 10 or more usable observations. Provide at least 10 positive-price, "
                "positive-demand rows per SKU."
            )
        analyzer = PricingModule()
        analyzer._model = uploaded_model
        analyzer._df = usable[required]
        predictions = [
            analyzer.predict({"sku": sku})
            for sku in sorted(uploaded_model.coefficients)
        ]
        identified = [
            sku for sku in uploaded_model.coefficients
            if abs(uploaded_model.coefficients[sku]) / max(uploaded_model.std_errors[sku], 1e-9) > 2.0
        ]
        n_fitted = len(uploaded_model.coefficients)
        scores = {
            "observations_used": float(len(usable)),
            "sku_count": float(n_fitted),
            "identified_share": len(identified) / max(1, n_fitted),
            "mean_absolute_elasticity": float(np.mean([abs(uploaded_model.coefficients[s]) for s in uploaded_model.coefficients])),
            "mean_r2": float(np.mean(list(uploaded_model.r2.values()))),
        }
        notes = [
            "Elasticity was fitted from the uploaded POS history using log(demand) ~ log(price), separately per SKU.",
            "Recommendations stay within each SKU's observed price range; no uploaded data is saved and the shared model is unchanged.",
            "positive_rate is the share of fitted SKUs whose elasticity differs from zero by more than two standard errors.",
        ]
        dropped = original_rows - len(usable)
        if dropped:
            notes.append(f"Ignored {dropped:,} rows with missing or non-positive price/demand values.")
        under_sampled = sorted(set(frame["sku"].astype(str)) - set(uploaded_model.coefficients))
        if under_sampled:
            notes.append(
                "Skipped SKUs with fewer than 10 usable observations: "
                + ", ".join(under_sampled[:20])
                + ("…" if len(under_sampled) > 20 else ".")
            )
        predictions.sort(
            key=lambda item: float(item.meta.get("estimated_revenue_uplift", 0.0)),
            reverse=True,
        )
        return UploadAnalysis(
            module_id=self.id,
            filename=filename,
            n_rows=original_rows,
            metrics=MetricsReport(
                n_samples=len(usable),
                positive_rate=len(identified) / max(1, n_fitted),
                scores=scores,
                threshold=0.0,
                notes=notes,
            ),
            predictions=predictions[:MAX_RETURNED_PREDICTIONS],
            predictions_truncated=len(predictions) > MAX_RETURNED_PREDICTIONS,
        )

    @property
    def manifest(self) -> ModuleManifest:
        m = self.model
        return ModuleManifest(
            id=self.id,
            title="Dynamic Pricing",
            summary=(
                "Own-price elasticity per SKU from point-of-sale history, then revenue-optimal "
                "price subject to a demand floor. Recommendations are clipped to the observed "
                "price range and carry their confidence interval."
            ),
            accent=ACCENT,
            status="live",
            capabilities=["regression", "elasticity", "optimisation", "uncertainty"],
            routes={"ui": "/pricing", "api": "/api/v1/pricing"},
            tools=self.tools(),
        )

    def predict(self, payload: dict[str, Any]) -> Prediction:
        sku = str(payload.get("sku", "")).strip()
        if not sku:
            raise ValueError("Missing required features: sku")
        m = self.model
        if sku not in m.coefficients:
            raise ValueError(f"Unknown SKU '{sku}'. Known: {', '.join(sorted(m.coefficients)[:8])}…")

        e = m.coefficients[sku]
        se = m.std_errors[sku]
        lo_p, hi_p = m.price_support[sku]
        floor = float(payload.get("min_demand_share", 0.8))  # keep at least 80% of baseline demand

        # Revenue = price * demand(price); demand(p) = k * p^e. With e < -1 revenue rises with
        # price; with e > -1 it falls. Search the observed support rather than solving in
        # closed form, so the demand floor can be enforced directly.
        grid = np.linspace(lo_p, hi_p, 200)
        k = float(np.exp(m.intercepts[sku]))
        demand = k * grid ** e
        revenue = grid * demand
        feasible = demand >= floor * m.mean_demand[sku]
        if feasible.any():
            best = int(np.argmax(np.where(feasible, revenue, -np.inf)))
        else:
            best = int(np.argmin(np.abs(demand - floor * m.mean_demand[sku])))

        best_price = float(grid[best])
        best_revenue = float(revenue[best])
        baseline_revenue = float(m.mean_demand[sku] * (lo_p + hi_p) / 2)
        uplift = (best_revenue - baseline_revenue) / baseline_revenue if baseline_revenue else 0.0

        # Clamp the score: a recommendation is not a probability, but the contract wants 0..1.
        score = float(np.clip(abs(e) / 3.0, 0.0, 1.0))

        ci_lo, ci_hi = e - 1.96 * se, e + 1.96 * se
        identified = abs(e) / se > 2.0

        if not identified:
            label, decision = "Elasticity not identified", (
                "This SKU's price barely varies, so elasticity is not statistically distinguishable "
                "from zero. Run a price test before acting."
            )
        elif uplift > 0.03:
            label, decision = "Price increase available", (
                f"Raise to ${best_price:.2f} for an estimated {uplift*100:.1f}% revenue lift within "
                f"the {floor*100:.0f}% demand floor."
            )
        elif uplift < -0.03:
            label, decision = "Price cut indicated", (
                f"Lower to ${best_price:.2f}; current pricing is leaving {abs(uplift)*100:.1f}% on the table."
            )
        else:
            label, decision = "Near optimal", "Current pricing is within noise of optimal. No change."

        return Prediction(
            module_id=self.id,
            score=round(score, 4),
            label=label,
            decision=decision,
            inputs={"price_low": lo_p, "price_high": hi_p},
            meta={
                "sku": sku,
                "elasticity": round(e, 3),
                "elasticity_ci": [round(ci_lo, 3), round(ci_hi, 3)],
                "identified": bool(identified),
                "r2": round(m.r2[sku], 4),
                "n_obs": m.n_obs[sku],
                "recommended_price": round(best_price, 2),
                "observed_price_range": [round(lo_p, 2), round(hi_p, 2)],
                "estimated_revenue_uplift": round(uplift, 4),
                "demand_floor": floor,
            },
        )

    def explain(self, payload: dict[str, Any]) -> Explanation:
        from core.types import FeatureAttribution

        sku = str(payload.get("sku", "")).strip()
        if not sku:
            raise ValueError("Missing required features: sku")
        p = self.predict(payload)
        meta = p.meta
        items = [
            FeatureAttribution.of("elasticity", meta["elasticity"], meta["elasticity"]),
            FeatureAttribution.of(
                "price_range_width",
                round(meta["observed_price_range"][1] - meta["observed_price_range"][0], 3),
                round((meta["observed_price_range"][1] - meta["observed_price_range"][0]) / 10, 4),
            ),
            FeatureAttribution.of("r2", meta["r2"], meta["r2"]),
        ]
        return Explanation(
            kind="heuristic",
            space="unspecified",
            attributions=items,
            summary=(
                f"Elasticity {meta['elasticity']:.2f} "
                f"(95% CI {meta['elasticity_ci'][0]:.2f} to {meta['elasticity_ci'][1]:.2f}), "
                f"R²={meta['r2']:.2f} on {meta['n_obs']} observations."
            ),
            caveats=[
                "Recommendations are clipped to the observed price range. Extrapolating "
                "elasticity beyond it is how pricing models destroy demand.",
                "Own-price only: competitor response, cross-elasticity and seasonality are not "
                "modelled. A large recommended move is a hypothesis, not a decision.",
                "Elasticity is estimated from historical variation, so it inherits whatever "
                "confounds were present — promotions, stockouts, weather.",
            ],
        )

    def evaluate(self) -> MetricsReport:
        """Recover the true elasticities — possible only because the data is generated."""
        m = self.model
        true = self.df.groupby("sku")["true_elasticity"].first()
        common = [s for s in m.coefficients if s in true.index]
        errs = np.array([m.coefficients[s] - true[s] for s in common])
        identified = [s for s in common if abs(m.coefficients[s]) / max(m.std_errors[s], 1e-9) > 2.0]
        ident_errs = np.array([m.coefficients[s] - true[s] for s in identified])

        return MetricsReport(
            n_samples=len(common),
            positive_rate=len(identified) / max(1, len(common)),
            scores={
                "mae_elasticity": float(np.abs(errs).mean()),
                "bias_elasticity": float(errs.mean()),
                "mae_when_identified": float(np.abs(ident_errs).mean()) if len(ident_errs) else float("nan"),
                "share_identified": len(identified) / max(1, len(common)),
                "mean_r2": float(np.mean([m.r2[s] for s in common])),
            },
            notes=[
                "This audit is only possible because the demand data is generated and the true "
                "elasticities are known. On real POS data there is no ground truth, so the "
                "honest check is a held-out price test.",
                f"{len(identified)} of {len(common)} SKUs have a statistically identified "
                "elasticity (|e|/SE > 2). The rest should get a price test, not a recommendation.",
                "Error is much smaller on identified SKUs than overall, which is the point of "
                "reporting the confidence interval at all.",
            ],
        )

    def tools(self) -> list[AgentTool]:
        return [
            AgentTool(
                name="optimise_price",
                description=(
                    "Recommend a revenue-optimal price for a SKU subject to a demand floor. "
                    "Returns elasticity with a confidence interval and whether it is identified."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string", "description": "SKU id, e.g. SKU-007"},
                        "min_demand_share": {"type": "number", "default": 0.8},
                    },
                    "required": ["sku"],
                },
            ),
            AgentTool(
                name="list_skus_by_opportunity",
                description="Rank all SKUs by estimated revenue opportunity and flag those whose elasticity is not identified.",
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {"n": {"type": "integer", "default": 10}},
                },
            ),
        ]

    def tool_optimise_price(self, sku: str, min_demand_share: float = 0.8) -> dict[str, Any]:
        return self.predict({"sku": sku, "min_demand_share": min_demand_share}).model_dump()

    def tool_list_skus_by_opportunity(self, n: int = 10) -> dict[str, Any]:
        rows = []
        for sku in self.model.coefficients:
            try:
                meta = self.predict({"sku": sku}).meta
            except ValueError:
                continue
            rows.append(
                {
                    "sku": sku,
                    "elasticity": meta["elasticity"],
                    "identified": meta["identified"],
                    "recommended_price": meta["recommended_price"],
                    "estimated_revenue_uplift": meta["estimated_revenue_uplift"],
                    "r2": meta["r2"],
                }
            )
        rows.sort(key=lambda r: -abs(r["estimated_revenue_uplift"]))
        unidentified = [r["sku"] for r in rows if not r["identified"]]
        return {
            "n_skus": len(rows),
            "n_not_identified": len(unidentified),
            "not_identified": unidentified,
            "top": rows[:n],
            "note": "SKUs flagged not-identified have prices that barely move; they need a price test, not a recommendation.",
        }


MODULE = PricingModule()
