# Sample data — one file per body

Point the hub's **Analyze your data** studio at any of these. Every file is accepted by the body
it belongs to, because every file is produced by that body's own loader or generator — see
`scripts/make_samples.py`, which regenerates this directory and writes this README.

| File | Body | What it is |
|---|---|---|
| [`churn/customers.csv`](churn/customers.csv) | `churn` | 400 subscribers, real telco extract, `churn` label |
| [`fraud/authorisations.csv`](fraud/authorisations.csv) | `fraud` | 600 authorisations, 1.2% confirmed fraud |
| [`maintenance/train_FD001_sample.txt`](maintenance/train_FD001_sample.txt) | `maintenance` | 1,273 telemetry rows, real FD001 layout (26 columns, no header) |
| [`maintenance/telemetry_with_rul.csv`](maintenance/telemetry_with_rul.csv) | `maintenance` | the same telemetry with a header and `rul`, for scoring against your labels |
| [`pricing/price_history.csv`](pricing/price_history.csv) | `pricing` | 6 SKUs x 200 price/demand observations |
| [`diagnostics/screening.csv`](diagnostics/screening.csv) | `xai_health` | 200 screening records, `diabetes` label |
| [`rag/returns-and-refunds-policy.md`](rag/returns-and-refunds-policy.md) | `rag` | a policy with deadlines and a numbered exception list |
| [`scraper/urls.csv`](scraper/urls.csv) | `scraper` | 5 allowlisted URLs to fetch |

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
