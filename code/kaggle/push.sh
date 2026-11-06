#!/usr/bin/env bash
# Publish the current code as a new private dataset version, then run the training notebook on a Kaggle T4.
# Usage (from the repo root):  code/kaggle/push.sh "L1.2: what changed"
# Needs the Kaggle CLI in .venv and a token in ~/.kaggle/access_token. Nothing is downloaded.
set -euo pipefail
msg="${1:?usage: code/kaggle/push.sh \"L1.x: what changed\"}"
here="$(cd "$(dirname "$0")" && pwd)"
kaggle="$here/../../.venv/bin/kaggle"
"$here/publish_code.sh" "$msg"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp "$here/training/kernel-metadata.json" "$here/../notebooks/DAMA_train.ipynb" "$tmp/"
"$kaggle" kernels push -p "$tmp" --accelerator NvidiaTeslaT4
echo "watch: $kaggle kernels status s41r4j/dama-tier1-training   (log after it ends: $kaggle kernels logs s41r4j/dama-tier1-training)"
