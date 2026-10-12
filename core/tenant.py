"""The demo tenant: one company, seven bodies, one joinable key space.

**Why this file exists.** A portfolio project that shows seven models on seven unrelated
datasets forces the person reviewing it to ask "so what is this, actually?" — and there is no
good answer, because nothing connects. Every body here belongs to a single fictional enterprise,
and every dataset is named, sourced and keyed:

    Northwind Global plc — a fictional multinational, 48,000 people, four business units:
      Northwind Connect     telecom + broadband subscriptions   -> churn
      Northwind Pay         card acquiring + payments          -> fraud
      Northwind Industrial  aircraft-engine MRO contracts      -> predictive maintenance
      Northwind Retail      own-brand consumer electronics     -> dynamic pricing
      Northwind Care        employee health screening pilot    -> explainable diagnostics
      Northwind Knowledge   group policy corpus                -> RAG
      Northwind Labs        public-web acquisition             -> scraper

**Why a fictional company and not a real one.** The provenance has to be exact, and two of the
datasets are genuinely public (IBM/Kaggle Telco Customer Churn; NASA CMAPSS FD001; Pima Indians
Diabetes), the rest are generated. Attributing those to a real named corporation would be a
fabrication — the one thing that would actually damage a candidate in an interview. So the tenant
is clearly labelled fictional and each card says exactly what the numbers are and where they came
from. "IBM Telco Customer Churn, 3,333 rows, standing in for Northwind Connect's billing extract"
is defensible in a room. "Amazon's churn data" is not.

**The connect-the-dots part that is real, not narrative.** ``customer_id`` is a single key space
shared by :mod:`modules.churn` and :mod:`modules.fraud`, so a subscriber ranked as a churn risk
can be joined to that same person's card activity in the fraud stream. Engines carry asset ids,
SKUs carry SKU ids. That is what "these are one company's systems" means in SQL rather than in a
README.
"""

from __future__ import annotations

from typing import Any, Literal

from core.types import DatasetCard

#: The tenant every body is described against. Rendered verbatim in the UI.
TENANT: dict[str, Any] = {
    "id": "northwind",
    "name": "Northwind Global",
    "legal_name": "Northwind Global plc (fictional)",
    "tagline": "A fictional multinational, used as one consistent tenant across all seven bodies.",
    "founded": "1974",
    "hq": "Rotterdam · Singapore · Austin",
    "headcount": "48,000",
    "revenue": "€11.4bn (illustrative)",
    "why_fictional": (
        "Three of the seven datasets below are real, public and citable (IBM Telco Customer "
        "Churn, NASA CMAPSS FD001, Pima Indians Diabetes) and the other four are generated. "
        "Labelling them as a named real company's internal data would be a fabrication. The "
        "tenant is therefore explicitly fictional, while every dataset keeps its true origin "
        "and its real row count — so nothing in this product claims something it cannot back up."
    ),
    "units": {
        "churn": "Northwind Connect",
        "fraud": "Northwind Pay",
        "maintenance": "Northwind Industrial",
        "pricing": "Northwind Retail",
        "xai_health": "Northwind Care",
        "rag": "Northwind Knowledge",
        "scraper": "Northwind Labs",
    },
}

Origin = Literal["public-dataset", "generated", "corpus", "uploaded", "none"]

#: Static provenance per body. ``rows`` is what the body scores as shipped; ``origin`` never
#: lies about whether the numbers are real, and the two things are reported separately on
#: purpose — a synthetic dataset standing in for a real system is a normal engineering
#: situation, but it is not the same thing as real data and the card must not blur them.
DATASETS: dict[str, dict[str, Any]] = {
    "churn": {
        "dataset": "Subscriber billing & behaviour extract",
        "rows": 3333,
        "grain": "one row per subscription per billing month",
        "key": "customer_id (NW-C-######)",
        "target": "churn — the subscription was cancelled within 30 days of the extract date",
        "period": "rolling 90-day window; monthly refresh",
        "origin": "public-dataset",
        "origin_detail": (
            "IBM/Kaggle Telco Customer Churn — 3,333 subscribers, 20 columns, 14.5% positive "
            "rate. A real, citable dataset, used here as the stand-in for Northwind Connect's "
            "billing extract."
        ),
        "origin_url": "https://github.com/ywchiu/python_for_data_science/blob/master/Data/customer_churn.csv",
        "file": "data/raw/telco_churn.csv (fetched by data/fetch.sh; a seeded generator with the "
        "same schema and correlation structure is used when the file is absent)",
        "system_of_record": "cdw.subscriber_monthly (Snowflake), fed by the Connect billing gateway",
        "why": (
            "Retention spend is finite. The body ranks the whole book of business so the save "
            "desk works the accounts that are actually worth working, and attaches an action "
            "to each score instead of a bare probability."
        ),
        "connect": (
            "customer_id is the same key space as the fraud body, so a subscriber flagged here "
            "can be joined to that person's card activity in Northwind Pay. Engines and SKUs "
            "carry their own asset/SKU ids."
        ),
        "caveats": [
            "Charge columns are excluded: charge == minutes x rate, so keeping them would give "
            "the model perfect collinearity and inflate feature importance.",
            "The public extract has no acquisition channel or contract type, which is why "
            "'why did they leave' is answered from behaviour, not from intent.",
        ],
    },
    "fraud": {
        "dataset": "Card authorisation ledger",
        "rows": 20000,
        "grain": "one row per authorisation attempt",
        "key": "auth_id (NW-A-#######) + customer_id (shared with the churn book)",
        "target": "is_fraud — the authorisation was later confirmed fraudulent by the chargeback desk",
        "period": "24-hour rolling stream; scored per event",
        "origin": "generated",
        "origin_detail": (
            "A seeded generator (modules/fraud/data.py) that deliberately overlaps the "
            "legitimate and fraudulent distributions, including two distinct fraud modes — "
            "card testing and account takeover. An earlier version separated them cleanly and "
            "scored PR-AUC 1.000, which measures nothing."
        ),
        "origin_url": "",
        "file": "generated in memory (20,000 authorisations by default); the real body consumes "
        "Redis Streams 'pay.auth.events' or a CDC feed from the processor",
        "system_of_record": "Redis Streams pay.auth.events -> Kafka topic auth.scored",
        "why": (
            "A card decline is a customer-visible decision. The body reports the operating "
            "point it would actually run at, keeps false declines on the ledger as a real cost, "
            "and pairs a supervised model with an isolation forest so novel fraud still trips."
        ),
        "connect": (
            "Transactions share customer_id with the churn book: the same 48,000-person "
            "customer base appears in both, which is what makes 'this customer is leaving *and* "
            "their card is being tested' an answerable question rather than two screens."
        ),
        "caveats": [
            "1.2% positive rate makes accuracy meaningless; the card leads with PR-AUC and lift.",
            "The isolation-forest half is unsupervised by construction — it is tuned for "
            "recall of novel patterns, not calibrated to a probability.",
        ],
    },
    "maintenance": {
        "dataset": "Engine telemetry (sensor telemetry per operating cycle)",
        "rows": 20631,
        "grain": "one row per engine per flight cycle",
        "key": "unit (NW-E-##) + cycle",
        "target": "rul — remaining useful life in cycles, capped at 125",
        "period": "per flight cycle; monthly model refresh",
        "origin": "public-dataset",
        "origin_detail": (
            "NASA CMAPSS FD001 (100 engines, 20,631 cycles) when the file is present; otherwise "
            "a generator reproducing the real 26-column schema and the degradation physics. "
            "13 of the 21 sensors are pure noise in the real data, and the generator keeps that "
            "property — a model that uses all 21 equally is fitting noise."
        ),
        "origin_url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
        "file": "data/raw/turbofan_train_FD001.txt (drop the real file here; the loader prefers it "
        "and reports which source it used)",
        "system_of_record": "Northwind Industrial MRO: engine health-monitoring gateway",
        "why": (
            "An unplanned engine removal costs a contracted airline roughly €250k per event. "
            "The body answers 'which unit, how many cycles, how sure' and reports its error per "
            "remaining-life band, because overall RMSE hides where it fails."
        ),
        "connect": (
            "Engines belong to Northwind Industrial's MRO contracts; the same hub agent can pair "
            "a unit's RUL with the SKU price of the replacement part from Northwind Retail."
        ),
        "caveats": [
            "Grouped cross-validation by engine, never random splits — random splits leak the "
            "same engine into train and test and report a fantasy RMSE.",
            "The RUL label is capped at 125: past that, 'not failing soon' is all the data says.",
        ],
    },
    "pricing": {
        "dataset": "Point-of-sale demand history",
        "rows": 9600,
        "grain": "one row per SKU per week per region",
        "key": "sku (NW-SKU-###)",
        "target": "units_sold — used to estimate own-price elasticity per SKU",
        "period": "104 weeks of history; weekly re-estimation",
        "origin": "generated",
        "origin_detail": (
            "A generated demand surface (modules/pricing/service.py) with per-SKU elasticity "
            "and noise. Generated *on purpose*: the true elasticity is therefore known, which "
            "is the only way to audit that the estimator recovers it. The audit reports the "
            "recovery error, so the body can prove its own maths instead of asserting it."
        ),
        "origin_url": "",
        "file": "generated in memory; the real body reads the POS feed in Northwind Retail",
        "system_of_record": "Retail POS feed -> demand_history (weekly aggregate)",
        "why": (
            "Pricing is where a data scientist most easily loses money by trusting a model. "
            "Recommendations are clipped to the observed price range, carry a confidence "
            "interval, and respect a demand floor."
        ),
        "connect": (
            "SKUs are Northwind Retail's own-brand electronics; the price of a replacement part "
            "quoted against a Northwind Industrial engine is the same SKU table."
        ),
        "caveats": [
            "Elasticity is estimated per SKU; SKUs with too little price variation are reported "
            "as unidentifiable rather than given a confident number.",
            "No competitor price enters the model — the scraper body exists to supply that, and "
            "is not wired in yet. That gap is stated on the card, not hidden.",
        ],
    },
    "rag": {
        "dataset": "Group policy & procedure corpus",
        "rows": None,
        "grain": "one chunk per section, heading-aware",
        "key": "source file + heading path",
        "target": "not a supervised target — retrieval ranked by TF-IDF, answered with citations",
        "period": "corpus re-indexed on change; documents carry an effective date",
        "origin": "corpus",
        "origin_detail": (
            "Seven short Northwind Global policy documents shipped in the repo at "
            "modules/rag/corpus/ (refunds, data privacy, fraud protocol, retention playbook, "
            "model card, SLA, plus a README). Short on purpose: every answer can be read and "
            "checked by hand, which is the point of a citation-first RAG demo."
        ),
        "origin_url": "",
        "file": "modules/rag/corpus/*.md — or your own .md/.txt/.pdf, added per session via "
        "'Analyze your data' without touching the shipped corpus",
        "system_of_record": "Confluence / SharePoint export (or Supabase, when SAROS_SUPABASE_LIVE=1)",
        "why": (
            "Policy questions are the highest-volume question type in any large company and the "
            "one where a confident wrong answer does real damage. Every answer here carries the "
            "passages it came from and a grounding verdict."
        ),
        "connect": (
            "The corpus is Northwind's own policy set, so its answers are about the same company "
            "whose churn and fraud decisions the other bodies make — including the retention "
            "playbook those churn decisions are supposed to follow."
        ),
        "caveats": [
            "TF-IDF, not embeddings: no model download, no 512 MB problem, and it degrades "
            "predictably. The trade-off is recall on paraphrase, which is stated in the UI.",
            "Zero hits is reported as zero hits. The retriever never pads an answer with an "
            "irrelevant passage to look confident.",
        ],
    },
    "scraper": {
        "dataset": "Public web pages (competitive price intel)",
        "rows": None,
        "grain": "one document per fetched URL",
        "key": "url + fetch timestamp",
        "target": "not a prediction — acquisition for the other bodies",
        "period": "on demand, host-allowlisted and size-capped",
        "origin": "none",
        "origin_detail": (
            "No dataset: this body fetches public pages. It runs against a labelled mock unless "
            "a backend is configured, and an empty allowlist blocks every live fetch — so the "
            "agent cannot be talked into hammering an arbitrary site."
        ),
        "origin_url": "",
        "file": "live HTTP (Jina Reader by default), or the labelled mock",
        "system_of_record": "Northwind Labs' competitive-intel crawler",
        "why": (
            "Acquisition is the unglamorous half of most real data work. It is a first-class "
            "body here so the agent can pull a page and hand it to another body as input."
        ),
        "connect": (
            "Feeds Northwind Retail's price comparison and Northwind Knowledge's document "
            "ingestion: scraped text can be re-indexed into the RAG corpus."
        ),
        "caveats": [
            "Robots/ToS compliance is the operator's responsibility; the allowlist is the "
            "technical control, not a legal one.",
        ],
    },
    "xai_health": {
        "dataset": "Employee health screening pilot",
        "rows": 768,
        "grain": "one row per screened employee",
        "key": "screening_id",
        "target": "diabetes_risk — screening outcome, triaged for follow-up",
        "period": "annual screening cycle",
        "origin": "public-dataset",
        "origin_detail": (
            "Pima Indians Diabetes study (768 records, 9 columns) — a real, public, citable "
            "dataset, used as the clinical stand-in for Northwind Care's screening pilot. Zero "
            "is a missing-value sentinel in five of its columns and is imputed, not averaged."
        ),
        "origin_url": "https://github.com/bigmlcom/python/blob/master/data/diabetes.csv",
        "file": "data/raw/diabetes.csv (fetched by data/fetch.sh; synthetic fallback with the "
        "same columns when absent)",
        "system_of_record": "Northwind Care occupational-health screening system",
        "why": (
            "Screening inverts the usual objective: a missed case is the harm that matters, so "
            "the operating point is chosen for false-negative rate rather than accuracy. Every "
            "score ships with its SHAP drivers and a Fairlearn group audit, because a clinical "
            "model that is unfair by age band is a legal and ethical problem, not a metric."
        ),
        "connect": (
            "The only body whose data is about people's health, which is why it is the one that "
            "ships a fairness audit; the model card in the RAG corpus is its governance record."
        ),
        "caveats": [
            "The public dataset has no race or gender field; age band is used as a proxy and "
            "labelled as a proxy, not as the protected attribute it stands in for.",
            "A screening model is decision support, never a diagnosis.",
        ],
    },
}

#: What a body says when it has no dataset of its own.
NO_DATASET = DatasetCard(
    module_id="",
    company=TENANT["name"],
    business_unit="",
    dataset="(none)",
    grain="",
    key="",
    target="",
    period="",
    origin="none",
    origin_detail="",
    file="",
    rows=None,
    system_of_record="",
    why="",
)


def dataset_for(module_id: str, **runtime: Any) -> DatasetCard:
    """Build the provenance card for a body, optionally enriched with runtime row counts.

    ``runtime`` carries whatever the module actually observed — ``rows``, a ``source`` of
    ``"real"``/``"synthetic"``/``"cmapss-fd001"``, a filename. Passing it in keeps this function
    pure and keeps the card honest: a body that has not loaded its data yet reports the shipped
    expectation, and a body that has, reports what it really read.
    """
    static = DATASETS.get(module_id)
    if static is None:
        return NO_DATASET.model_copy(
            update={"module_id": module_id, "business_unit": TENANT["units"].get(module_id, "")}
        )

    observed_rows = runtime.pop("rows", None)
    observed_source = runtime.pop("source", None)
    observed_basis = runtime.pop("basis", "")

    card = DatasetCard(
        module_id=module_id,
        company=TENANT["name"],
        company_note=TENANT["tagline"],
        business_unit=TENANT["units"].get(module_id, ""),
        observed_rows=observed_rows,
        observed_source=observed_source or "",
        observed_basis=observed_basis,
        **static,
    )
    return card


def prediction_meta(module_id: str, **extra: Any) -> dict[str, Any]:
    """The tenant stamp every prediction carries.

    Small on purpose — four short fields — because it travels with every scored record and the
    point is that a *single* prediction can be traced back to a named dataset without a second
    API call. The full card lives at /api/v1/modules/{id}/dataset.
    """
    d = DATASETS.get(module_id)
    if d is None:
        return extra
    return {
        "tenant": TENANT["name"],
        "business_unit": TENANT["units"].get(module_id, ""),
        "dataset": d["dataset"],
        "record_key": d["key"],
        **extra,
    }


def provenance_block(module_id: str, **extra: Any) -> dict[str, Any]:
    """The attribution stamp that travels with a ranked list or a stream summary.

    Every tool that returns *rows* — a target list, an engine ranking, a fraud stream — says in the
    same payload whose data it is, what the dataset is called, where it really came from, how many
    rows it covers and what the join key is. The point is that a list cannot be read without the
    answer to "what am I looking at?" sitting in the same object, rather than three screens away
    or in someone's head.

    ``rows`` here is the shipped expectation from this registry; a body that observed something
    different reports that separately through its dataset card.
    """
    d = DATASETS.get(module_id)
    if d is None:
        return {"tenant": TENANT["name"], **extra}
    origin_detail = d.get("origin_detail", "")
    # The short, citable name of the source: the text before the first dash or full stop. Long
    # enough to name the dataset, short enough to sit in a JSON payload without a paragraph.
    source = origin_detail.split(" — ")[0].split(". ")[0].strip()
    return {
        "tenant": TENANT["name"],
        "business_unit": TENANT["units"].get(module_id, ""),
        "dataset": d["dataset"],
        "dataset_source": source,
        "dataset_origin": d["origin"],
        "dataset_rows": d.get("rows"),
        "join_key": d["key"],
        "system_of_record": d.get("system_of_record", ""),
        **extra,
    }


def customer_id(row_index: int) -> str:
    """The shared key space. Deterministic, so a row ranked by churn is the same person a
    fraud event belongs to, in any process, with no lookup table."""
    return f"NW-C-{row_index + 1:06d}"


def auth_id(txn_index: int) -> str:
    return f"NW-A-{txn_index + 1:07d}"


def asset_id(unit: int) -> str:
    return f"NW-E-{unit:03d}"


def sku_id(n: int) -> str:
    return f"NW-SKU-{n:03d}"
