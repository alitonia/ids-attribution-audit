#!/usr/bin/env bash
# tail_b.sh — pod B's post-campaign tail (run when pod A's uploads are due):
# wait for A's marker file ON DISK (hf download exits 0 even on no-match —
# exit codes cannot gate this), merge A's cells, aggregate the full 195-cell
# generation, verify the evidence chain, tar + upload.
set -e
cd /workspace
export S1_RESULTS_DIR=/workspace/results_rerun S1_CKPT_DIR=/workspace/ckpt_v2 S1_PROCESSED_DIR=/workspace/proc_v2
export S1_RUN_ID=rerun2026_20260828 S1_DETERMINISTIC=1 S1_GIT_REV=$(cat /workspace/GIT_REV) S1_TORCH_THREADS=2
log() { echo "[$(date -u +%H:%M:%S)] $*" >> logs/status.log; }

MARKER=/tmp/waita/rerun2026/outputs/results_rerun/exp_ciciot2023_none_seed4_K5.json
until hf download "hunopapa/vifr-campaign-data" --type dataset \
    --include "rerun2026/outputs/results_rerun/exp_ciciot2023_none_seed4_K5.json" \
    --local-dir /tmp/waita >/dev/null 2>&1 && [ -f "$MARKER" ]; do
  log "[wait] pod A uploads not on HF yet"
  sleep 60
done
log "[done] wait_for_A"

hf download "hunopapa/vifr-campaign-data" --type dataset \
  --include "rerun2026/outputs/results_rerun/*" --local-dir /workspace/hf_merge >> logs/merge.log 2>&1
mkdir -p /workspace/results_rerun/manifest
cp -a /workspace/hf_merge/rerun2026/outputs/results_rerun/. /workspace/results_rerun/ >> logs/merge.log 2>&1
log "[done] merge_pod_A"

python3 -u scripts/aggregate_v2.py 2>&1 | tee -a logs/aggregate.log
python3 -u scripts/verify_evidence.py 2>&1 | tee -a logs/verify.log

tar czf /workspace/ckpt_v2_B.tgz -C /workspace ckpt_v2 \
  results_rerun/EVIDENCE_MANIFEST.json 2>/dev/null || \
  tar czf /workspace/ckpt_v2_B.tgz -C /workspace ckpt_v2
hf upload "hunopapa/vifr-campaign-data" /workspace/ckpt_v2_B.tgz \
  rerun2026/outputs/ckpt_v2_B.tgz --type dataset >> logs/upload.log 2>&1
hf upload "hunopapa/vifr-campaign-data" /workspace/results_rerun \
  rerun2026/outputs/results_rerun --type dataset >> logs/upload.log 2>&1
hf upload "hunopapa/vifr-campaign-data" /workspace/logs \
  rerun2026/outputs/logs_B --type dataset >> logs/upload.log 2>&1
log "[done] uploads role=B (tail)"
