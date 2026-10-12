"""Upload tests — "analyze your own data" has to be real, capped and non-destructive.

These are the tests that keep the upload path honest:

* a correct file produces the platform's normal ``MetricsReport``/``Prediction`` shapes, so the
  existing UI renders it without a second code path;
* a wrong file produces a refusal that *names the columns*, which is the difference between a
  usable tool and a dead end;
* an oversized file is refused while it is being read, not after it is in memory;
* an unlabelled file gets a distribution and an explicit statement that no accuracy figure is
  being claimed — never a fabricated quality number;
* nothing is written to disk, and a second session cannot see the first session's upload.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app, registry
from core.sessions import STORE, MAX_FILE_BYTES


@pytest.fixture(scope="module")
def client():
    if len(registry) == 0:
        registry.discover("modules")
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def churn_csv() -> bytes:
    """A real churn extract, header included, complete with its label column."""
    from modules.churn.data import FEATURES, TARGET, load_telco

    df, _ = load_telco()
    return df.head(60)[[*FEATURES, TARGET]].to_csv(index=False).encode()


def _post(client, module_id: str, filename: str, content: bytes, session: str = "s_test_upload_1"):
    return client.post(
        f"/api/v1/modules/{module_id}/analyze",
        headers={"x-saros-session": session},
        files={"file": (filename, io.BytesIO(content), "text/csv")},
    )


# --- happy path -----------------------------------------------------------------


def test_churn_upload_scores_the_file_and_reports_the_columns(client, churn_csv) -> None:
    r = _post(client, "churn", "my_customers.csv", churn_csv)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["rows"] == 60
    assert body["filename"] == "my_customers.csv"
    assert len(body["columns_report"]["matched"]) == 13
    assert body["columns_report"]["missing"] == []

    assert body["metrics"]["n_samples"] == 60
    # Labelled upload, so real metrics against the user's own outcomes.
    assert "precision" in body["metrics"]["scores"]
    assert body["metrics"]["confusion"]["tn"] >= 1

    assert body["predictions"], "no predictions returned"
    top = body["predictions"][0]
    assert 0.0 <= top["score"] <= 1.0
    assert top["label"] and top["decision"]
    # Same shape the shipped-data UI renders, drivers included.
    assert top["factors"], "predictions must carry their drivers"


def test_upload_result_says_it_is_the_users_data_not_the_tenants(client, churn_csv) -> None:
    body = _post(client, "churn", "my_customers.csv", churn_csv).json()
    assert body["dataset"]["origin"] == "uploaded"
    assert "my_customers.csv" in body["dataset"]["dataset"]
    # The provenance card must not silently inherit the shipped dataset's identity.
    assert body["dataset"]["dataset"] != "Subscriber billing & behaviour extract"


def test_unlabelled_upload_reports_a_distribution_and_refuses_to_claim_accuracy(client, churn_csv) -> None:
    import pandas as pd

    df = pd.read_csv(io.BytesIO(churn_csv)).drop(columns=["churn"])
    r = _post(client, "churn", "no_labels.csv", df.to_csv(index=False).encode())
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    assert "pr_auc" not in metrics["scores"], "there is no ground truth to compute PR-AUC from"
    assert "median" in metrics["scores"]
    assert any("no accuracy metric is reported" in n for n in metrics["notes"])


def test_binary_labels_are_understood_in_yes_no_form(client, churn_csv) -> None:
    import pandas as pd

    df = pd.read_csv(io.BytesIO(churn_csv))
    df["churn"] = df["churn"].map({1: "yes", 0: "no"})
    r = _post(client, "churn", "labels_as_words.csv", df.to_csv(index=False).encode())
    assert r.status_code == 200
    assert "pr_auc" in r.json()["metrics"]["scores"]


def test_column_names_are_matched_case_and_punctuation_insensitively(client, churn_csv) -> None:
    import pandas as pd

    df = pd.read_csv(io.BytesIO(churn_csv))
    df = df.rename(columns={"total_day_minutes": "Total Day Minutes", "churn": "Churned?"})
    r = _post(client, "churn", "excel_export.csv", df.to_csv(index=False).encode())
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == 60


# --- refusals that are useful ----------------------------------------------------


def test_wrong_columns_lists_what_is_expected_and_what_was_found(client) -> None:
    bad = b"state,area_code,account_length\nKS,area_code_415,120\n"
    r = _post(client, "churn", "wrong.csv", bad)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "expected_columns" in detail and len(detail["expected_columns"]) == 13
    assert "account_length" in detail["found_columns"]
    assert "missing" in detail["message"]
    assert detail["hint"]


def test_oversize_upload_is_rejected_while_reading(client) -> None:
    """The cap must bite during the read: checking after buffering does not protect 512 MB."""
    payload = b"a,b\n" + b"1,2\n" * ((MAX_FILE_BYTES // 4) + 10)
    r = _post(client, "churn", "huge.csv", payload)
    assert r.status_code == 413
    assert r.json()["detail"]["max_bytes"] == MAX_FILE_BYTES


def test_empty_file_is_rejected(client) -> None:
    r = _post(client, "churn", "empty.csv", b"")
    assert r.status_code == 422


def test_unknown_body_404s(client) -> None:
    r = _post(client, "no_such_body", "x.csv", b"a,b\n1,2\n")
    assert r.status_code == 404


def test_a_non_csv_upload_to_a_tabular_body_says_what_happened(client) -> None:
    r = _post(client, "churn", "notes.md", b"# hello\n\nthis is not a table\n")
    assert r.status_code == 422
    assert "Could not parse" in json.dumps(r.json()["detail"]) or "columns" in json.dumps(r.json()["detail"])


# --- per-body formats -----------------------------------------------------------


def test_maintenance_accepts_headerless_whitespace_telemetry(client) -> None:
    from modules.maintenance.data import SENSORS, SETTINGS, synthesize

    df = synthesize(n_units=4, max_cycles=180, seed=31).head(240)
    payload = df[["unit", "cycle", *SETTINGS, *SENSORS]].to_csv(sep=" ", header=False, index=False).encode()
    r = _post(client, "maintenance", "FD001_sample.txt", payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == 240
    assert body["predictions"], "no RUL predictions"
    # Risk is inverted RUL, so the most at-risk row must sort first.
    scores = [p["score"] for p in body["predictions"]]
    assert scores == sorted(scores, reverse=True)


def test_maintenance_refuses_a_file_with_the_wrong_number_of_columns(client) -> None:
    payload = b"1 2 3\n4 5 6\n"
    r = _post(client, "maintenance", "bad.txt", payload)
    assert r.status_code == 422
    detail = json.dumps(r.json()["detail"])
    assert "columns" in detail and "CMAPSS" in detail


def test_maintenance_accepts_the_real_26_column_fd001_layout(client) -> None:
    """The real NASA file has 3 settings, not 6; "drop the real file in" has to actually work."""
    from modules.maintenance.data import SENSORS, SETTINGS, synthesize

    df = synthesize(n_units=4, max_cycles=180, seed=44).head(180)
    # Emit the real FD001 width: unit, cycle, three settings, 21 sensors.
    payload = df[["unit", "cycle", *SETTINGS[:3], *SENSORS]].to_csv(
        sep=" ", header=False, index=False
    ).encode()
    r = _post(client, "maintenance", "train_FD001_sample.txt", payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == 180
    assert any("constant column" in n for n in body["notes"])
    assert body["predictions"]


def test_pricing_fits_elasticity_on_the_users_own_rows(client) -> None:
    from modules.pricing.service import synthesize_demand

    df = synthesize_demand(n_skus=3, n_obs=60, seed=17)
    r = _post(client, "pricing", "my_pos.csv", df.to_csv(index=False).encode())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metrics"]["scores"]["n_skus_identified"] >= 2
    assert body["predictions"]
    note = body["predictions"][0]["meta"]["note"]
    assert "your" in note.lower(), "the recommendation must say it came from the user's rows"


def test_pricing_says_so_when_it_falls_back_to_the_tenant_model(client) -> None:
    r = _post(client, "pricing", "skus.csv", b"sku\nNW-SKU-001\nNW-SKU-002\n")
    assert r.status_code == 200, r.text
    notes = " ".join(r.json()["metrics"]["notes"])
    assert "tenant's fitted elasticity model" in notes


def test_rag_accepts_a_document_and_proves_it_is_retrievable(client) -> None:
    doc = (
        b"# Vendor Onboarding Policy - Acme Ltd\n\n"
        b"## Lead time\nAcme requires 90 days notice before a supplier is onboarded.\n\n"
        b"## Insurance\nSuppliers must carry 5 million euro of product liability cover.\n"
    )
    r = _post(client, "rag", "acme_policy.md", doc, session="s_test_rag_upload")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "document"
    assert body["rows"] >= 1, "chunks indexed"
    assert body["metrics"]["scores"]["chunks_indexed"] >= 1
    self_check = body["notes"][-1]
    assert "acme_policy" in self_check and "returned" in self_check


def test_uploaded_document_is_answerable_by_the_agent_without_an_llm_key(client) -> None:
    """The zero-key deployment is the default one, so the upload→ask loop must work on it."""
    session = "s_test_rag_chat"
    doc = (
        "# Returns Policy - Zephyr Supplies\n\n"
        "## Standard window\nZephyr accepts returns within 21 days of delivery for unopened goods.\n"
    )
    assert _post(client, "rag", "zephyr.md", doc.encode(), session=session).status_code == 200

    r = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "According to the document I uploaded, how long is the returns window?",
            "session_id": session,
        },
    )
    assert r.status_code == 200, r.text
    answer = r.json()["answer"]
    assert "21 days" in answer, answer
    assert "zephyr" in answer.lower()


def test_the_knowledge_tool_sees_session_uploads_when_asked_with_a_session(client) -> None:
    session = "s_test_tool_overlay"
    doc = b"# Travel Policy - Northwind\n\n## Class\nVice presidents fly premium economy on flights over six hours.\n"
    assert _post(client, "rag", "travel.md", doc, session=session).status_code == 200

    r = client.post(
        "/api/v1/agent/tools/search_knowledge",
        json={"arguments": {"query": "What class do vice presidents fly?", "k": 3}, "session_id": session},
    )
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    assert result["session_overlay"], "the uploaded document should be retrievable"
    assert result["session_overlay"][0]["source"] == "travel"


def test_scraper_accepts_a_url_list_and_reports_the_backend_per_row(client) -> None:
    r = _post(client, "scraper", "urls.csv", b"url\nhttps://example.com/pricing\n")
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert preds and preds[0]["meta"]["url"] == "https://example.com/pricing"
    assert preds[0]["label"] in ("Fetched live", "Fetched (labelled mock)")


def test_fraud_upload_flags_its_small_sample_size(client) -> None:
    from modules.fraud.data import FEATURES, TARGET, synthesize

    df = synthesize(n=40, seed=13)
    payload = df[[*FEATURES, TARGET]].to_csv(index=False).encode()
    r = _post(client, "fraud", "small.csv", payload)
    assert r.status_code == 200, r.text
    notes = " ".join(r.json()["metrics"]["notes"])
    assert "labelled rows" in notes, "a 40-row sample must not be presented as a quality estimate"


# --- sessions: in memory, isolated, never on disk -------------------------------


def test_uploads_are_scoped_to_their_session(client, churn_csv) -> None:
    _post(client, "churn", "session_a.csv", churn_csv, session="s_isolation_a")
    a = client.get("/api/v1/sessions/s_isolation_a").json()
    b = client.get("/api/v1/sessions/s_isolation_b").json()
    assert a["exists"] is True and any(u["filename"] == "session_a.csv" for u in a["uploads"])
    assert b["exists"] is False, "another session must not see this upload"


def test_an_unknown_session_id_reports_limits_rather_than_404s(client) -> None:
    body = client.get("/api/v1/sessions/s_never_existed").json()
    assert body["exists"] is False
    assert body["limits"]["max_file_bytes"] == MAX_FILE_BYTES
    assert "in-memory" in body["limits"]["storage"]


def test_nothing_is_written_to_disk(client, churn_csv) -> None:
    """The shipped corpus is the platform's data; a visitor's file must never join it."""
    from pathlib import Path

    corpus = Path("modules/rag/corpus")
    before = sorted(p.name for p in corpus.glob("*"))
    data_files_before = sorted(p.name for p in Path("data").glob("**/*") if p.is_file())

    _post(client, "churn", "customer_export.csv", churn_csv, session="s_disk_check")
    _post(client, "rag", "visitor_doc.md", b"# Visitor doc\n\n## Section\nNothing is persisted.\n", "s_disk_check")

    assert sorted(p.name for p in corpus.glob("*")) == before
    assert sorted(p.name for p in Path("data").glob("**/*") if p.is_file()) == data_files_before


def test_session_store_rejects_malformed_ids_instead_of_creating_junk() -> None:
    assert STORE.valid_id("s_" + "a" * 18) is True
    assert STORE.valid_id("short") is False
    assert STORE.valid_id("bad id with spaces") is False
    assert STORE.valid_id("../../etc/passwd") is False
    assert STORE.get("../../etc/passwd") is None
