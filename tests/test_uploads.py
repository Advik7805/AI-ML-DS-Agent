"""HTTP coverage for the bounded, in-memory analyze-your-data endpoints."""

from __future__ import annotations

import csv
from io import StringIO

import pytest
from fastapi.testclient import TestClient

import apps.api.main as api_main
from apps.api.main import MAX_UPLOAD_BYTES, app, registry
from core.agent.providers import MockProvider
from modules.churn.data import FEATURES as CHURN_FEATURES, TARGET as CHURN_TARGET
from modules.fraud.data import FEATURES as FRAUD_FEATURES, TARGET as FRAUD_TARGET
from modules.maintenance.data import FEATURES as MAINTENANCE_FEATURES, synthesize as synthesize_maintenance
from modules.rag import session_corpus
from modules.rag.session_corpus import SessionCorpus


@pytest.fixture(scope="module")
def upload_client():
    if len(registry) == 0:
        registry.discover("modules")
    with TestClient(app) as client:
        yield client


def _csv_bytes(columns: list[str], rows: list[dict[str, object]]) -> bytes:
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _multipart(client: TestClient, module_id: str, filename: str, content: bytes, **data):
    return client.post(
        f"/api/v1/modules/{module_id}/analyze",
        files={"file": (filename, content, "application/octet-stream")},
        data=data,
    )


def _one_page_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 50 740 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode())
        document.extend(body + b"\nendobj\n")
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode())
    document.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    )
    return bytes(document)


def test_churn_csv_upload_returns_shared_metrics_and_predictions(upload_client: TestClient) -> None:
    values = {
        "account_length": 120,
        "number_vmail_messages": 18,
        "total_day_minutes": 180,
        "total_day_calls": 110,
        "total_eve_minutes": 200,
        "total_eve_calls": 100,
        "total_night_minutes": 200,
        "total_night_calls": 100,
        "total_intl_minutes": 10,
        "total_intl_calls": 4,
        "number_customer_service_calls": 1,
        "international_plan": 0,
        "voice_mail_plan": 1,
    }
    rows = [
        {**values, CHURN_TARGET: "no"},
        {**values, "number_customer_service_calls": 5, CHURN_TARGET: "yes"},
    ]
    content = _csv_bytes([*CHURN_FEATURES, CHURN_TARGET], rows)

    response = _multipart(upload_client, "churn", "customers.csv", content)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["module_id"] == "churn"
    assert result["n_rows"] == 2
    assert result["metrics"]["n_samples"] == 2
    assert set(result["metrics"]["confusion"]) == {"tn", "fp", "fn", "tp"}
    assert len(result["predictions"]) == 2
    assert result["predictions"][0]["meta"]["uploaded_data"] is True


def test_analyze_rejects_wrong_columns_with_actionable_422(upload_client: TestClient) -> None:
    response = _multipart(upload_client, "churn", "wrong.csv", b"account_length,wrong\n1,2\n")

    assert response.status_code == 422
    assert "missing required columns" in response.json()["detail"]
    assert "total_day_minutes" in response.json()["detail"]


def test_analyze_rejects_upload_over_10_mib(upload_client: TestClient) -> None:
    response = _multipart(upload_client, "churn", "large.csv", b"0" * (MAX_UPLOAD_BYTES + 1))

    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]


def test_fraud_csv_upload_uses_existing_ensemble(upload_client: TestClient) -> None:
    base = {
        "amount": 42.5,
        "hour_of_day": 11,
        "distance_from_home_km": 4.2,
        "txn_count_1h": 2,
        "distinct_merchants_24h": 2,
        "card_age_days": 750,
        "avg_amount_30d": 38.0,
        "is_night": 0,
        "is_foreign": 0,
        "pin_used": 1,
    }
    rows = [
        {**base, FRAUD_TARGET: 0},
        {**base, "amount": 1250.0, "hour_of_day": 2, "distance_from_home_km": 1200, FRAUD_TARGET: 1},
    ]
    response = _multipart(
        upload_client,
        "fraud",
        "transactions.csv",
        _csv_bytes([*FRAUD_FEATURES, FRAUD_TARGET], rows),
        source_label="Contoso internal ledger",
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["predictions"]) == 2
    assert result["metrics"]["n_samples"] == 2
    assert "accuracy" in result["metrics"]["scores"]
    assert result["source_label"] == "Contoso internal ledger"
    assert result["session_id"]
    assert any(name.endswith(".analysis.md") for name in result["session_sources"])
    assert "not transactions from a real company" in result["metrics"]["notes"][0]
    assert result["predictions"][0]["meta"]["training_data_source"].startswith("synthetic")
    assert result["predictions"][0]["meta"]["decisive_model"] in {
        "supervised model",
        "isolation forest",
    }

    retrieved = upload_client.post(
        "/api/v1/agent/tools/search_knowledge",
        json={
            "arguments": {
                "query": "company source Contoso internal ledger",
                "session_id": result["session_id"],
            }
        },
    )
    assert retrieved.status_code == 200, retrieved.text
    assert any(hit["source"].endswith(".analysis.md") for hit in retrieved.json()["result"]["results"])
    assert "Contoso internal ledger" in retrieved.json()["result"]["results"][0]["snippet"]


def test_pricing_csv_upload_fits_sku_elasticity(upload_client: TestClient) -> None:
    rows = [
        {"sku": "SKU-A", "price": price, "demand": round(1200 * (price / 10) ** -1.6)}
        for price in range(8, 24)
    ]
    response = _multipart(
        upload_client,
        "pricing",
        "pos-history.csv",
        _csv_bytes(["sku", "price", "demand"], rows),
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["n_rows"] == len(rows)
    assert result["metrics"]["scores"]["sku_count"] == 1
    assert result["predictions"][0]["meta"]["sku"] == "SKU-A"
    low, high = result["predictions"][0]["meta"]["observed_price_range"]
    assert low <= result["predictions"][0]["meta"]["recommended_price"] <= high


def test_maintenance_accepts_headerless_cmapss_txt(upload_client: TestClient) -> None:
    telemetry = synthesize_maintenance(n_units=2, max_cycles=150, seed=23)
    latest = telemetry.groupby("unit", as_index=False).tail(1)
    # CMAPSS TXT has no header and contains unit, cycle, three settings, and 21 sensors.
    columns = ["unit", "cycle", *[f"setting{i}" for i in range(1, 4)], *[f"sensor{i}" for i in range(1, 22)]]
    lines = [" ".join(f"{float(value):.6f}" for value in row) for row in latest[columns].to_numpy()]
    response = _multipart(upload_client, "maintenance", "engines.txt", "\n".join(lines).encode())

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["metrics"]["n_samples"] == 2
    assert len(result["predictions"]) == 2
    assert all("rul_cycles" in prediction["meta"] for prediction in result["predictions"])


def test_rag_text_upload_is_session_scoped_and_reusable(upload_client: TestClient) -> None:
    session_id = "upload-test-session"
    content = (
        "# Custom return policy\n\n" 
        "Customers may return unopened equipment within 47 days with a receipt."
    ).encode()
    first = _multipart(
        upload_client,
        "rag",
        "returns.txt",
        content,
        session_id=session_id,
        query="How many days can unopened equipment be returned?",
    )

    assert first.status_code == 200, first.text
    result = first.json()
    assert result["session_id"] == session_id
    assert result["predictions"][0]["meta"]["hits"]
    assert any(hit["source"] == "returns.txt" for hit in result["predictions"][0]["meta"]["hits"])

    # No file is needed to reuse the same in-memory session index for another query.
    second = upload_client.post(
        "/api/v1/modules/rag/analyze",
        data={"session_id": session_id, "query": "return receipt"},
    )
    assert second.status_code == 200, second.text
    assert second.json()["n_rows"] == 1

    # An unrelated browser session cannot see the uploaded passage.
    isolated = _multipart(
        upload_client,
        "rag",
        "other.txt",
        b"# Other\n\nA telescope warranty lasts 18 months.",
        session_id="different-upload-session",
        query="How many days can unopened equipment be returned?",
    )
    assert isolated.status_code == 200, isolated.text
    assert all(
        hit["source"] != "returns.txt"
        for hit in isolated.json()["predictions"][0]["meta"]["hits"]
    )


def test_rag_search_tool_and_hub_agent_retrieve_session_upload(upload_client: TestClient) -> None:
    session_id = "agent-rag-upload-session"
    uploaded = _multipart(
        upload_client,
        "rag",
        "returns.txt",
        b"# Custom return policy\n\nUnopened equipment may be returned within 47 days with a receipt.",
        session_id=session_id,
        source_label="Fabrikam returns portal",
    )
    assert uploaded.status_code == 200, uploaded.text

    direct = upload_client.post(
        "/api/v1/agent/tools/search_knowledge",
        json={"arguments": {"query": "unopened equipment returned 47 days", "session_id": session_id}},
    )
    assert direct.status_code == 200, direct.text
    assert any(hit["source"] == "returns.txt" for hit in direct.json()["result"]["results"])
    assert "returns.txt" in direct.json()["result"]["citation"]

    source_search = upload_client.post(
        "/api/v1/agent/tools/search_knowledge",
        json={
            "arguments": {
                "query": "uploader company data source Fabrikam returns portal",
                "session_id": session_id,
            }
        },
    )
    assert source_search.status_code == 200, source_search.text
    assert any("Fabrikam returns portal" in hit["snippet"] for hit in source_search.json()["result"]["results"])

    assert api_main.agent is not None
    previous_provider = api_main.agent.provider
    api_main.agent.provider = MockProvider()
    try:
        streamed = upload_client.post(
            "/api/v1/agent/chat/stream",
            json={
                "message": "Search the uploaded returns.txt document: unopened equipment within 47 days.",
                "session_id": session_id,
            },
        )
    finally:
        api_main.agent.provider = previous_provider

    assert streamed.status_code == 200, streamed.text
    assert "returns.txt" in streamed.text
    assert "Custom return policy" in streamed.text


def test_rag_analyze_bounds_query_and_session_metadata(upload_client: TestClient) -> None:
    long_query = upload_client.post(
        "/api/v1/modules/rag/analyze",
        data={"query": "q" * 2_001},
    )
    long_session = upload_client.post(
        "/api/v1/modules/rag/analyze",
        data={"session_id": "s" * 129},
    )

    assert long_query.status_code == 422
    assert "2,000 characters" in long_query.json()["detail"]
    assert long_session.status_code == 422
    assert "128 characters" in long_session.json()["detail"]


def test_rag_pdf_upload_extracts_text(upload_client: TestClient) -> None:
    response = _multipart(
        upload_client,
        "rag",
        "return-policy.pdf",
        _one_page_pdf("Unopened equipment may be returned within 47 days with a receipt."),
        session_id="upload-pdf-session",
        query="How many days for unopened equipment returns?",
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["predictions"][0]["meta"]["hits"]
    assert result["predictions"][0]["meta"]["hits"][0]["source"] == "return-policy.pdf"


def test_rag_text_limit_counts_utf8_bytes_not_characters(upload_client: TestClient) -> None:
    content = ("é" * 1_000_001).encode("utf-8")
    response = _multipart(
        upload_client,
        "rag",
        "unicode.txt",
        content,
        session_id="unicode-byte-limit",
    )

    assert response.status_code == 422
    assert "2 MB per-document limit" in response.json()["detail"]


def test_maintenance_csv_upload_uses_optional_rul_labels(upload_client: TestClient) -> None:
    telemetry = synthesize_maintenance(n_units=2, max_cycles=150, seed=9)
    latest = telemetry.groupby("unit", as_index=False).tail(1)
    response = _multipart(
        upload_client,
        "maintenance",
        "engines.csv",
        latest.to_csv(index=False).encode("utf-8"),
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["metrics"]["n_samples"] == 2
    assert "mae_cycles" in result["metrics"]["scores"]


def test_rag_session_store_evicts_least_recently_used_session(monkeypatch) -> None:
    monkeypatch.setattr(session_corpus, "MAX_SESSION_BYTES", 4)
    monkeypatch.setattr(session_corpus, "MAX_TOTAL_SESSION_BYTES", 6)
    store = SessionCorpus()
    store.append("old-session", "old.txt", "abcd")
    store.append("recent-session", "recent.txt", "ef")
    store.documents("old-session")  # touch it, so recent-session becomes least recently used

    store.append("new-session", "new.txt", "gh")

    assert store.documents("old-session") == [("old.txt", "abcd")]
    assert store.documents("recent-session") == []
    assert store.documents("new-session") == [("new.txt", "gh")]


def test_maintenance_feature_manifest_matches_upload_schema() -> None:
    # This small contract check keeps future model feature changes from silently drifting away
    # from the CSV/TXT upload parser.
    assert len(MAINTENANCE_FEATURES) == 27
    assert MAINTENANCE_FEATURES[0] == "setting1"
