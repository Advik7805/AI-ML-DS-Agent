"""Provenance tests — every body can say whose data it is scoring.

The failure these guard against is the one that makes a portfolio project hard to assess: seven
dashboards of numbers with no statement of what they are. A body that cannot name its company,
its dataset, its row count and its join key is failing this suite, not just missing polish.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app, registry
from core.tenant import DATASETS, TENANT, asset_id, auth_id, customer_id, sku_id

ALLOWED_ORIGINS = {"public-dataset", "generated", "corpus", "uploaded", "none"}


@pytest.fixture(scope="module")
def client():
    if len(registry) == 0:
        registry.discover("modules")
    with TestClient(app) as c:
        yield c


# --- the registry itself -------------------------------------------------------


def test_every_registered_body_has_a_dataset_card(registry) -> None:
    missing = [m.id for m in registry.all() if m.dataset_card().dataset in ("", "(none)")]
    assert not missing, f"bodies with no provenance: {missing}"


def test_ranked_lists_carry_their_provenance_inline(client) -> None:
    """A list of rows must not be readable without knowing whose data it is and where it came from."""
    r = client.post(
        "/api/v1/agent/tools/rank_at_risk_customers",
        json={"arguments": {"n": 3}},
    )
    assert r.status_code == 200
    block = r.json()["result"]
    for field in ("tenant", "business_unit", "dataset", "dataset_source", "dataset_origin",
                  "dataset_rows", "join_key", "system_of_record"):
        assert block.get(field), f"the churn ranking is missing '{field}'"
    assert block["dataset_rows"] == 3333
    assert block["customers"][0]["customer_id"].startswith("NW-C-")

    # Engines are ranked by the same rule, keyed by the asset id the registry publishes.
    engines = client.post(
        "/api/v1/agent/tools/rank_engines_by_urgency", json={"arguments": {"n": 2}}
    ).json()["result"]
    assert engines["business_unit"] == "Northwind Industrial"
    assert engines["engines"][0]["asset_id"].startswith("NW-E-")
    assert engines["engines"][0]["rul_cap"] == 125

    # The fraud stream reports per-row keys as well as its provenance.
    fraud = client.post(
        "/api/v1/agent/tools/summarise_fraud_stream", json={"arguments": {"n": 20}}
    ).json()["result"]
    assert fraud["business_unit"] == "Northwind Pay"
    if fraud["top_flagged"]:
        assert fraud["top_flagged"][0]["auth_id"].startswith("NW-A-")


def test_the_tenant_registry_covers_every_body_that_ships(registry) -> None:
    """A new body without a registry entry would render an empty card, so fail loudly here."""
    assert set(registry.ids()) == set(DATASETS), (
        "core/tenant.py and modules/ have drifted: "
        f"{set(registry.ids()) ^ set(DATASETS)}"
    )


@pytest.mark.parametrize("module_id", sorted(DATASETS))
def test_cards_answer_the_five_questions_a_reviewer_asks(module_id: str) -> None:
    card = DATASETS[module_id]
    for field in ("dataset", "grain", "key", "target", "period"):
        assert card[field], f"{module_id}.{field} is empty"


def test_origins_are_from_the_closed_set(registry) -> None:
    for module in registry.all():
        assert module.dataset_card().origin in ALLOWED_ORIGINS


def test_public_dataset_claims_cite_a_source(registry) -> None:
    """Claiming 'real public dataset' without a link is exactly the claim that must be checkable."""
    for module in registry.all():
        card = module.dataset_card()
        if card.origin == "public-dataset":
            assert card.origin_url.startswith("http"), f"{module.id} claims a public dataset with no URL"
            assert card.file, f"{module.id} does not say which file it reads"


def test_every_card_states_what_the_model_is_for(registry) -> None:
    for module in registry.all():
        card = module.dataset_card()
        assert len(card.why) > 40, f"{module.id} does not explain why the model exists"
        assert len(card.origin_detail) > 40, f"{module.id} does not describe its origin in detail"


def test_manifests_carry_the_same_card_as_the_endpoint(registry) -> None:
    """The tile blurb and the dataset endpoint must not be two different stories."""
    for module in registry.all():
        manifest = module.manifest
        assert manifest.dataset is not None, module.id
        assert manifest.dataset.business_unit == module.dataset_card().business_unit


def test_manifest_summaries_name_the_tenant_and_the_dataset(registry) -> None:
    """Interview-proofing at the copy level: the tile itself says which company and which data."""
    for module in registry.all():
        summary = module.manifest.summary
        assert TENANT["units"][module.id] in summary or "Northwind" in summary, module.id
        assert "Data:" in summary, f"{module.id} summary does not name its dataset"


# --- the key space that makes the bodies one company ---------------------------


def test_customer_key_format_is_stable() -> None:
    assert customer_id(0) == "NW-C-000001"
    assert customer_id(999) == "NW-C-001000"
    assert auth_id(0) == "NW-A-0000001"
    assert asset_id(7) == "NW-E-007"
    assert sku_id(12) == "NW-SKU-012"


def test_churn_and_fraud_share_the_same_customer_key_space() -> None:
    """The join that makes 'one company' true in SQL rather than in prose."""
    from modules.churn.data import load_telco
    from modules.fraud.data import synthesize

    churn, _ = load_telco()
    fraud = synthesize(n=500, seed=3)

    assert "customer_id" in churn.columns
    assert "customer_id" in fraud.columns
    joined = fraud.merge(churn[["customer_id", "churn"]], on="customer_id", how="inner")
    assert len(joined) == len(fraud), "fraud customers must exist in the churn book"
    assert joined["customer_id"].str.startswith("NW-C-").all()


def test_churn_rank_tool_returns_the_join_key(registry) -> None:
    result = registry.get("churn").tool_rank_at_risk_customers(n=3)
    assert result["customers"], "ranking returned nothing"
    for row in result["customers"]:
        assert row["customer_id"].startswith("NW-C-")
    assert result["business_unit"] == "Northwind Connect"
    # The registry's key string carries the id format as well as the column, e.g.
    # "customer_id (NW-C-######)" — informative in a payload, so the assertion checks the column
    # name it starts with rather than flattening it.
    assert result["join_key"].startswith("customer_id")


def test_predictions_carry_the_tenant_stamp(registry) -> None:
    """A single prediction must be traceable to a named dataset with no second call."""
    from modules.churn.data import FEATURES, load_telco

    df, _ = load_telco()
    payload = {f: float(df.iloc[0][f]) for f in FEATURES}
    meta = registry.get("churn").predict(payload).meta
    assert meta["tenant"] == "Northwind Global"
    assert meta["business_unit"] == "Northwind Connect"
    assert "Subscriber" in meta["dataset"]


def test_observed_rows_are_reported_only_after_the_data_is_read(registry) -> None:
    """The card distinguishes 'what ships' from 'what this process actually trained on'."""
    module = registry.get("churn")
    _ = module.bundle  # force a load/train
    card = module.dataset_card()
    assert card.rows == 3333
    assert card.observed_rows and card.observed_rows >= card.rows * 0.25
    assert card.observed_source in ("real", "synthetic")
    # The card must say what the observed count is a count *of*.
    assert card.observed_basis in ("full source frame", "training split"), card.observed_basis


# --- HTTP surface --------------------------------------------------------------


def test_tenant_endpoint_names_the_company(client) -> None:
    r = client.get("/api/v1/tenant")
    assert r.status_code == 200
    body = r.json()
    assert body["tenant"]["name"] == "Northwind Global"
    assert body["tenant"]["units"]["churn"] == "Northwind Connect"
    assert len(body["bodies"]) == len(registry.ids())
    for entry in body["bodies"]:
        assert entry["dataset"]["company"] == "Northwind Global"


def test_dataset_endpoint_per_body(client) -> None:
    for module_id in sorted(DATASETS):
        r = client.get(f"/api/v1/modules/{module_id}/dataset")
        assert r.status_code == 200, module_id
        card = r.json()
        assert card["module_id"] == module_id
        assert card["business_unit"]


def test_dataset_endpoint_404s_for_an_unknown_body(client) -> None:
    r = client.get("/api/v1/modules/does-not-exist/dataset")
    assert r.status_code == 404


def test_module_listing_includes_provenance(client) -> None:
    churn = next(m for m in client.get("/api/v1/modules").json() if m["id"] == "churn")
    assert churn["dataset"]["origin"] == "public-dataset"
    assert churn["dataset"]["origin_url"].startswith("http")
