#!/usr/bin/env bash
# fetch_rerun.sh — pull the rerun generation outputs from HF into the repo.
# Works after pod termination. Checkpoints land in the repo archive (tar).
set -eu
HF_REPO="hunopapa/vifr-campaign-data"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

hf download "$HF_REPO" --type dataset \
  --include "rerun2026/outputs/*" --local-dir "$STAGE"
SRC="$STAGE/rerun2026/outputs"

mkdir -p results/rerun2026 reports/pod_runs/rerun2026
cp -r "$SRC/results_rerun/." results/rerun2026/
cp -r "$SRC/logs/." reports/pod_runs/rerun2026/logs/ 2>/dev/null || true
for T in "$SRC"/ckpt_v2_A.tgz "$SRC"/ckpt_v2_B.tgz; do
  if [ -f "$T" ]; then
    mkdir -p artifacts
    cp "$T" "artifacts/$(basename "$T" .tgz)_$(date -u +%Y%m%d).tgz"
  fi
done
echo "[DONE] archived results/rerun2026 + reports/pod_runs/rerun2026 + artifacts/ckpt tar"
echo "[NEXT] python3 scripts/verify_evidence.py   (chain check on the local copy)"
