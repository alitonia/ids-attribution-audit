#!/usr/bin/env bash
# hf_push_rerun.sh — push rerun code to HF IN ADVANCE (no pod, no billing).
# Run AFTER committing; S1_GIT_REV for the manifest is derived from HEAD.
set -eu
REPO_NAME="hunopapa/vifr-campaign-data"
GIT_REV=$(git -C "$(dirname "$0")" rev-parse HEAD)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/code"

cp -r src "$STAGE/code/src"
cp -r scripts "$STAGE/code/scripts"
echo "$GIT_REV" > "$STAGE/code/GIT_REV"

echo "[INFO] pushing rerun code @ $GIT_REV"
hf upload "$REPO_NAME" "$STAGE/code" rerun2026/code --type dataset \
  --commit-message "rerun2026 code (v2 campaign, dual quarantine, dual RS bounds, evidence manifest)"
echo "[DONE] data comes from the existing deploy_full.tar.gz payload on the pod"
