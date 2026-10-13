"""Write one sample file per body into ``samples/``.

**Why this exists.** The upload studio's answer to "with what file?" used to be a shrug. A
first-time visitor was asked to supply a CSV in a schema they had never seen, for a body they
had just met, and the studio could only tell them what it needed *after* they got it wrong. The
fix is not more copy — it is eight files that are known to be accepted.

**Why generated rather than hand-written.** A sample is a promise about a schema, and a
hand-written sample is a promise that rots the first time a feature is added. Every file here is
produced by the body's *own* loader or generator — ``load_telco``, ``synthesize``,
``synthesize_demand``, ``load_diabetes`` — so the columns are the body's columns by construction
and cannot drift. The two files that are prose (the policy and the URL list) have no schema to
drift from.

**Regenerating.** ``python scripts/make_samples.py`` rewrites everything, including
``samples/README.md``. Run ``bash data/fetch.sh`` first: the churn and diagnostics samples are
slices of the real telco and Pima datasets when they are present, and of the synthetic fallbacks
when they are not — either is accepted, but the committed copies are the real ones.

``python scripts/make_samples.py --verify`` pushes every file through the same
``body.analyze_upload`` the API uses, which is the fastest end-to-end check that uploads still
work for all seven bodies.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

#: Importable when this file is run as a script (``python scripts/make_samples.py``), where
#: Python puts ``scripts/`` on the path rather than the repository root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Sample:
    """One file the studio can be pointed at, and the body that accepts it."""

    module_id: str
    relative: str
    what: str
    build: Callable[[], bytes]


# --- per-body builders ---------------------------------------------------------


def _churn() -> bytes:
    """400 subscribers from the real telco extract, ``churn`` label included.

    The label is the point: with it, the upload path reports precision/recall against real
    outcomes instead of a score distribution, which is the case a visitor most wants to see.
    """
    from modules.churn.data import load_telco

    df, _ = load_telco()
    return df.head(400).to_csv(index=False).encode()


def _fraud() -> bytes:
    from modules.fraud.data import synthesize

    return synthesize(n=600).to_csv(index=False).encode()


#: Small-fleet settings for the maintenance samples. ``max_cycles`` must stay above 130: the
#: generator draws each engine's life from ``integers(130, max_cycles)``, so a smaller ceiling
#: is an empty range and a crash, not a short file.
MAINTENANCE_UNITS, MAINTENANCE_CYCLES = 8, 200


def _maintenance_frame():
    from modules.maintenance.data import synthesize

    return synthesize(n_units=MAINTENANCE_UNITS, max_cycles=MAINTENANCE_CYCLES)


def _maintenance_txt() -> bytes:
    """The real FD001 layout: no header, whitespace-aligned, 26 columns.

    ``unit, cycle``, three operational settings, 21 sensors. The schema this repo trains on
    carries six settings, so the file is deliberately the *real* shape — the upload path pads
    the three absent constants and says so in the notes. Writing 29 columns here would have made
    the sample unable to demonstrate the one case ("drop in the real NASA file") the body
    advertises.
    """
    from modules.maintenance.data import SENSORS

    columns = ["unit", "cycle", "setting1", "setting2", "setting3", *SENSORS]
    assert len(columns) == 26, "the CMAPSS layout is 26 columns wide"
    return _maintenance_frame()[columns].to_csv(sep=" ", index=False, header=False).encode()


def _maintenance_csv() -> bytes:
    """The same telemetry as a headed CSV, keeping ``rul``.

    A header is what human eyes need, and ``rul`` is what turns this from "score my sensors"
    into "check your model's error against my remaining-life labels".
    """
    return _maintenance_frame().to_csv(index=False).encode()


def _pricing() -> bytes:
    """Six SKUs of price/demand history. Ten observations per SKU is the fitting floor; 200 is
    enough for an identified elasticity and small enough to read."""
    from modules.pricing.service import synthesize_demand

    return synthesize_demand(n_skus=6, n_obs=200).to_csv(index=False).encode()


def _diagnostics() -> bytes:
    """200 screening records from the Pima extract, label kept.

    Zero-valued glucose/BMI entries are written as blanks rather than zeros: zero is the source
    data's missing-value sentinel, and a file that spells it out lets the upload path impute it
    as missing instead of scoring a healthy-looking blood pressure of 0.
    """
    from modules.xai_health.data import load_diabetes

    df, _ = load_diabetes()
    return df.head(200).to_csv(index=False).encode()


#: A policy document with a real heading structure, explicit deadlines and a numbered exception
#: list — the three things a retrieval demo has to be able to answer about. Written rather than
#: reused from the shipped corpus on purpose: uploading a copy of a file that is *already*
#: indexed is the one way to make a citation ambiguous.
_POLICY = """\
# Returns and Refunds Policy — Northwind Retail

Owner: Group Customer Operations · Version 4.2 · Effective 1 January 2026

## Scope

This policy covers returns and refunds for physical goods sold by Northwind Retail, and for
digital goods purchased through the Northwind Retail storefront. It does not cover services
billed by Northwind Connect, which are governed by the subscriber terms.

## Standard return window

A customer may return an unused item within **30 days** of delivery for a full refund to the
original payment method. Refunds are issued within 5-7 business days of the item being received
at a Northwind warehouse.

## Extended holiday window

Purchases made between 15 November and 24 December carry an extended window: returns are
accepted until **15 January** of the following year, regardless of the 30-day rule above.

## Faulty goods

Goods that arrive faulty, or that fail within **12 months** of delivery, are treated as a
warranty claim rather than a return. The customer chooses repair, replacement or refund; all
three are offered, and the return window does not apply.

## Exceptions

The following exceptions are not eligible for a refund under this policy, whatever the window:

1. **Opened software licences** and activated digital keys, once the licence has been redeemed.
2. **Personalised or made-to-order items**, including engraved products and bespoke furniture.
3. **Perishable goods** more than 5 days past delivery, including food and live plants.
4. **Hygiene products** whose sealed packaging has been opened, including cosmetics and bedding.
5. **Gift cards**, which are transferable but non-refundable and never expire.
6. **Clearance items** marked "final sale" at the point of purchase, where the markdown exceeds
   60% of the list price.

## Escalation

A request refused at the first decision point escalates to a Retail Operations supervisor, who
may approve an exception outside this policy up to **$500**. Anything above $500 requires
regional sign-off within 2 business days.

## Data notes

Return records are retained for 7 years for tax purposes. Personal data attached to a return is
deleted 90 days after the refund settles; see the Group Privacy Office policy for the full
retention schedule.
"""


def _rag() -> bytes:
    return _POLICY.encode()


#: The hosts here are the ones ``.env.example`` documents as the allowlist example
#: (``example.com,docs.example.com``), so a deploy that follows the documentation can run this
#: file live. With no allowlist configured every row comes back marked "labelled mock" — which
#: is the honest result, and it is what a fresh free-tier deploy will show.
_SCRAPER_URLS = [
    ("https://example.com/", "homepage", "smallest possible page: baseline for the extractor"),
    ("https://example.com/pricing", "pricing", "competitive price intel, Northwind Labs"),
    ("https://example.com/about", "about", "company background for the knowledge overlay"),
    ("https://docs.example.com/returns", "returns-policy", "feeds the returns policy overlay"),
    ("https://docs.example.com/sla", "sla", "availability commitments, group service management"),
]


def _scraper() -> bytes:
    # Written through the csv module rather than by joining strings: a note containing a comma
    # produced an unquoted four-field row, and the upload path rejected the whole file with
    # "Expected 3 fields in line 3, saw 4". Hand-rolled CSV is hand-rolled quoting.
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["url", "label", "note"])
    writer.writerows(_SCRAPER_URLS)
    return buffer.getvalue().encode()


SAMPLES_LIST: tuple[Sample, ...] = (
    Sample("churn", "churn/customers.csv", "400 subscribers, real telco extract, `churn` label", _churn),
    Sample("fraud", "fraud/authorisations.csv", "600 authorisations, 1.2% confirmed fraud", _fraud),
    Sample(
        "maintenance",
        "maintenance/train_FD001_sample.txt",
        "1,273 telemetry rows, real FD001 layout (26 columns, no header)",
        _maintenance_txt,
    ),
    Sample(
        "maintenance",
        "maintenance/telemetry_with_rul.csv",
        "the same telemetry with a header and `rul`, for scoring against your labels",
        _maintenance_csv,
    ),
    Sample("pricing", "pricing/price_history.csv", "6 SKUs x 200 price/demand observations", _pricing),
    Sample(
        "xai_health",
        "diagnostics/screening.csv",
        "200 screening records, `diabetes` label",
        _diagnostics,
    ),
    Sample(
        "rag",
        "rag/returns-and-refunds-policy.md",
        "a policy with deadlines and a numbered exception list",
        _rag,
    ),
    Sample("scraper", "scraper/urls.csv", "5 allowlisted URLs to fetch", _scraper),
)

README = """\
# Sample data — one file per body

Point the hub's **Analyze your data** studio at any of these. Every file is accepted by the body
it belongs to, because every file is produced by that body's own loader or generator — see
`scripts/make_samples.py`, which regenerates this directory and writes this README.

| File | Body | What it is |
|---|---|---|
{table}

## Try this

1. Regenerate everything (including this README): `python scripts/make_samples.py`.
   Run `bash data/fetch.sh` first for the real telco and Pima samples.
2. Check every file still uploads: `python scripts/make_samples.py --verify` — it pushes each one
   through the same `analyze_upload` the API uses and prints accepted/rejected per file.
3. In the hub, choose a body under *Analyze your data* and press **Use the … sample file** —
   the studio fetches the committed file from `GET /api/v1/samples/{module_id}` and uploads it
   through the same path a dropped file takes. Or drop one of these files yourself; either way
   the file is visible to the agent in the same conversation and is never written to disk.

Files are committed so the numbers in them are the numbers in this table. Nothing in `samples/`
is loaded by the test suite except `tests/test_samples.py`, which asserts exactly the properties
above: accepted by its body, under 2 MB, and listed here.
"""


def write_readme() -> Path:
    rows = "\n".join(
        f"| [`{s.relative}`]({s.relative}) | `{s.module_id}` | {s.what} |" for s in SAMPLES_LIST
    )
    path = SAMPLES / "README.md"
    # `str.replace` rather than `str.format`: this text names a route,
    # `GET /api/v1/samples/{module_id}`, and `format()` would read that as a field and raise
    # KeyError. Documenting the API inside a template should not require escaping the API.
    path.write_text(README.replace("{table}", rows))
    return path


def build_all() -> list[Path]:
    written: list[Path] = []
    for sample in SAMPLES_LIST:
        path = SAMPLES / sample.relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = sample.build()
        path.write_bytes(payload)
        written.append(path)
        print(f"  wrote {sample.relative} ({len(payload):,} bytes)")
    written.append(write_readme())
    print("  wrote README.md")
    return written


def verify() -> int:
    """Push every sample through the body's real ``analyze_upload``. Returns an exit code.

    This is the check that matters: the contract between a sample file and a body is only ever
    exercised by the upload path, and a sample that a body rejects is worse than no sample at
    all — it tells a visitor the product is broken when the file is what is wrong.
    """
    from core.contract import ModuleRegistry
    from core.sessions import MAX_FILE_BYTES, Session

    registry = ModuleRegistry().discover("modules")
    passed = 0
    print("Verifying samples through body.analyze_upload(...):")
    for sample in SAMPLES_LIST:
        path = SAMPLES / sample.relative
        if not path.exists():
            print(f"  ✗ {sample.relative}: missing — run without --verify first")
            continue
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            print(f"  ✗ {sample.relative}: {len(data):,} bytes exceeds the upload cap")
            continue
        try:
            result = registry.get(sample.module_id).analyze_upload(
                path.name, data, Session(id="s_samples")
            )
        except Exception as exc:  # noqa: BLE001 - a rejection *is* the result here
            print(f"  ✗ {sample.relative} → {sample.module_id}: {type(exc).__name__}: {exc}")
            continue
        matched = result.columns_report.matched
        detail = f"{result.rows} rows/chunks, kind={result.kind}"
        if matched:
            detail += f", matched {len(matched)} columns"
        print(f"  ✓ {sample.relative} → {sample.module_id}: {detail}")
        passed += 1
    print(f"\n{passed}/{len(SAMPLES_LIST)} ok")
    return 0 if passed == len(SAMPLES_LIST) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="push every sample through its body's analyze_upload and report accepted/rejected",
    )
    args = parser.parse_args(argv)
    if args.verify:
        return verify()
    print(f"Writing {len(SAMPLES_LIST)} samples to {SAMPLES.relative_to(ROOT)}/")
    build_all()
    print("\nVerify with: python scripts/make_samples.py --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
