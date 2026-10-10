#!/usr/bin/env bash
#
# Fetch the real datasets into data/raw/.
#
# `data/raw/` is git-ignored: datasets are inputs, not source, and a 350 KB CSV per module
# would bloat the repo once six bodies each bring their own. Without this file present,
# `modules/churn/data.py` transparently falls back to its synthetic generator, so a fresh
# clone and CI both work with no network access.
#
# The telecom churn set is pulled from a committed copy on GitHub via the API (base64), which
# is why this script needs no credentials and no Kaggle account.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/raw

fetch_github_file() {
  local repo="$1" path="$2" dest="$3"
  if [[ -f "$dest" ]]; then
    echo "  = $dest already present, skipping"
    return 0
  fi
  echo "  -> $repo :: $path"
  curl -fsSL "https://api.github.com/repos/${repo}/contents/${path}" \
    -H "Accept: application/vnd.github+json" |
    python3 -c 'import json,sys,base64; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)["content"]))' \
    > "$dest"
  echo "     wrote $(wc -c < "$dest") bytes -> $dest"
}

echo "Fetching datasets:"

# Classic telecom churn dataset: 3,333 rows, 14.5% positive rate.
fetch_github_file "ywchiu/python_for_data_science" "Data/customer_churn.csv" "data/raw/telco_churn.csv"

echo
echo "Done. Verify with:  python3 -m modules.churn.train"
