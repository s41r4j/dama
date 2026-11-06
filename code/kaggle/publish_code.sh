#!/usr/bin/env bash
# Upload the current code as a new version of the private dataset s41r4j/dama-code and wait until it is ready.
# Usage (from anywhere):  code/kaggle/publish_code.sh "L1.x: what changed"
set -euo pipefail
msg="${1:?usage: code/kaggle/publish_code.sh \"message\"}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
kaggle="$root/.venv/bin/kaggle"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
(cd "$root/code" && zip -qr "$tmp/dama_code.zip" dama_train dama_agent dmbm tests notebooks README.md requirements.txt requirements-runtime.txt -x '*__pycache__*')
# imported by the paper experiments: v1 imports v0; v2 imports v1 and v2lib
(cd "$root" && zip -qr "$tmp/dama_code.zip" experiments/v0_2026-10-22/dama_paper_v0.py experiments/v1_2026-10-24/dama_paper_v1.py \
    experiments/v2_2026-10-28/v2lib.py experiments/v2_2026-10-28/dama_paper_v2.py experiments/v2b_2026-10-29/dama_paper_v2b.py \
    experiments/v2c_2026-10-30/dama_paper_v2c.py experiments/v3b_2026-11-01/dama_paper_v3b.py \
    experiments/dmbm1_2026-11-01/dama_dmbm1.py experiments/dama_v4_2026-11-02/dama_v4.py \
    experiments/dama_v4_2026-11-02/packets experiments/dama_v5_2026-11-03/dama_v5.py \
    experiments/dama_v5_2026-11-03/packets)
cp "$root/code/kaggle/dataset/dataset-metadata.json" "$tmp/"
"$kaggle" datasets version -p "$tmp" -m "$msg"
sleep 30   # right after an upload, status can still report the previous version as "ready"
# tail -1: newer CLIs print an "outdated version" warning before the status
until [ "$("$kaggle" datasets status s41r4j/dama-code 2>/dev/null | tail -1)" = "ready" ]; do sleep 15; done
echo "code version ready: $msg"
