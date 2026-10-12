#!/usr/bin/env bash
#
# Fetch the real datasets into data/raw/.
#
# `data/raw/` is git-ignored: datasets are inputs, not source, and a 350 KB CSV per body would
# bloat the repo once seven bodies each bring their own. Every body falls back to a synthetic
# generator when its file is missing, so a fresh clone and CI both work with no network access.
#
# Everything here is pulled from a committed copy on GitHub via the contents API, which is why
# this script needs no credentials, no Kaggle account and no LFS client.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/raw

is_lfs_pointer() { head -c 48 "$1" | grep -q "git-lfs.github.com"; }

fetch_github_file() {
  local repo="$1" path="$2" dest="$3"
  if [[ -f "$dest" ]]; then
    if is_lfs_pointer "$dest"; then
      echo "  ! $dest is a stale LFS pointer, not a dataset — refetching" >&2
      rm -f "$dest"
    else
      echo "  = $dest already present ($(wc -c < "$dest") bytes), skipping"
      return 0
    fi
  fi
  echo "  -> $repo :: $path"
  # Deliberately python3 and not curl. The API image is python:3.11-slim, which ships wget but
  # NOT curl, so a curl-based fetch fails inside the Docker build — and because the Dockerfile
  # runs this script as `... || echo "skipped"`, that failure is silent: the image trains on
  # synthetic fallbacks while claiming the real dataset is baked in. python3 is the one thing
  # guaranteed present in a Python image, so the fetch cannot depend on anything else.
  python3 - "$repo" "$path" "$dest" <<'PY'
import base64, json, sys, urllib.request

repo, path, dest = sys.argv[1:4]
url = f"https://api.github.com/repos/{repo}/contents/{path}"
req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
with urllib.request.urlopen(req, timeout=120) as resp:
    payload = json.load(resp)
if "content" not in payload:
    raise SystemExit(f"github contents API returned no content for {repo}/{path}: "
                     f"{payload.get('message', payload)}")
with open(dest, "wb") as fh:
    fh.write(base64.b64decode(payload["content"]))
PY

  # Guard against Git LFS pointers. The blob API does not bypass LFS, so a repo that stores the
  # CSV through LFS hands back a 132-byte pointer that *looks* like a file. Reading one as data
  # fails silently and confusingly, so refuse it here instead.
  if is_lfs_pointer "$dest"; then
    echo "     !! LFS pointer, not a dataset — removing" >&2
    rm -f "$dest"
    return 1
  fi
  echo "     wrote $(wc -c < "$dest") bytes -> $dest"
}

echo "Fetching datasets:"

# Telecom churn — 3,333 rows, 20 columns, 14.5% positive rate. Used by modules/churn.
fetch_github_file "ywchiu/python_for_data_science" "Data/customer_churn.csv" "data/raw/telco_churn.csv"

# Pima Indians diabetes — 768 rows x 9 columns, target `diabetes`. Used by modules/xai_health.
# Zero is a missing-value sentinel for glucose, blood pressure, skin thickness, insulin and BMI;
# the body converts those to NaN before training.
fetch_github_file "bigmlcom/python" "data/diabetes.csv" "data/raw/diabetes.csv"

# Cleveland heart disease — 303 rows x 14 columns. Not wired into a body yet; fetched so the
# diagnostics storefront can grow a second target without a new data pipeline.
fetch_github_file "tdpetrou/Learn-Pandas" "data/heart.csv" "data/raw/heart.csv"

# --- NASA CMAPSS turbofan (deliberately NOT fetched) -------------------------------
#
# modules/maintenance wants train_FD001.txt: 100 engines, 20,631 rows. Every copy reachable
# through the GitHub blob API is either an LFS pointer or a small subset (2-3 engines), which
# is worse than synthetic because it silently changes what the metrics mean.
#
# So the body ships a CMAPSS-compatible generator instead, and auto-prefers a real file if you
# drop one at data/raw/turbofan_train_FD001.txt — it reports `source` in its metrics notes so
# the UI never lies about provenance. Get the real file from the NASA Prognostics Data
# Repository and place it there yourself:
#
#   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

echo
echo "Done. Verify with:"
echo "  python3 -m pytest -q"
