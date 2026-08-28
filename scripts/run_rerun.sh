#!/usr/bin/env bash
# run_rerun.sh — deploy + run the 2026-08-28 full re-run generation on a pod.
# Usage: bash run_rerun.sh <pod_name> <A|B>
#   ROLE A: CIC + CICIoT2023 campaigns + section-IV probes; no aggregation.
#   ROLE B: NF campaign, then merges A's results from HF, aggregates the full
#           195-cell generation, verifies the evidence chain, uploads all.
# Preconditions: hf_push_rerun.sh run; pod exists (BOTH PODS SAME GPU TYPE);
# recommend GPU_TYPE="NVIDIA GeForce RTX 4090" MIN_VCPU=12 MIN_MEM=45.
set -eu
POD_NAME="${1:?Usage: bash run_rerun.sh <pod_name> <A|B>}"
ROLE="${2:?Usage: bash run_rerun.sh <pod_name> <A|B>}"
KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
HF_REPO="hunopapa/vifr-campaign-data"
GQL() { curl -s "https://api.runpod.io/graphql?api_key=${KEY}" \
          -H "Content-Type: application/json" -d "{\"query\": \"$1\"}"; }

# [1] SSH endpoint
POD_JSON=$(GQL "{ myself { pods { name desiredStatus runtime { ports { ip privatePort publicPort type } } } } }")
IP=$(echo "$POD_JSON" | python3 -c "
import sys,json
pods=[p for p in json.load(sys.stdin)['data']['myself']['pods'] if p['name']=='$POD_NAME' and p.get('runtime')]
print(next((pt['ip'] for pt in pods[0]['runtime']['ports'] if pt['privatePort']==22), '') if pods else '')")
PORT=$(echo "$POD_JSON" | python3 -c "
import sys,json
pods=[p for p in json.load(sys.stdin)['data']['myself']['pods'] if p['name']=='$POD_NAME' and p.get('runtime')]
print(next((pt['publicPort'] for pt in pods[0]['runtime']['ports'] if pt['privatePort']==22), '') if pods else '')")
[ -n "$IP" ] && [ -n "$PORT" ] || { echo "pod $POD_NAME not reachable"; exit 1; }
echo "[INFO] pod $POD_NAME -> $IP:$PORT"

# [2] SSH wait
for i in {1..60}; do
  if ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@"$IP" "echo 'SSH is up'" 2>/dev/null; then break; fi
  echo "waiting for SSH ($i/60)..."; sleep 10
done

# [3] HF token inject (ephemeral pod)
ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$IP" "mkdir -p /root/.cache/huggingface && chmod 700 /root/.cache/huggingface" </dev/null
scp -P "$PORT" -o StrictHostKeyChecking=no ~/.cache/huggingface/token root@"$IP":/root/.cache/huggingface/token

# [4] quick foreground install of the HF CLI (needed by step [5]), then the
#     big dependency pip in background (marker-file pattern). Guard kills a
#     leftover pip from a previous deploy attempt BY PIDFILE + process group —
#     never by pattern (a pattern matches this wrapper's own cmdline and
#     kills the session, observed as ssh exit 255).
ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$IP" "OLD=\$(cat /workspace/.pip_bg.pid 2>/dev/null || true); if [ -n \"\$OLD\" ]; then kill -- -\$OLD 2>/dev/null || true; fi; sleep 1; pip install -q --break-system-packages 'huggingface_hub[cli]' >/dev/null 2>&1; rm -f /workspace/.pip_done; nohup setsid bash -c 'pip install --break-system-packages numpy==1.26.4 scipy pandas pyarrow fastparquet scikit-learn tqdm ninja \"huggingface_hub[cli]\" > /workspace/pip_install.log 2>&1 && pip install --break-system-packages --no-build-isolation --no-deps traker >> /workspace/pip_install.log 2>&1; pip uninstall -y --break-system-packages fast_jl >> /workspace/pip_install.log 2>&1 || true; touch /workspace/.pip_done' >/dev/null 2>&1 & echo \$! > /workspace/.pip_bg.pid" </dev/null

# [5] Pull code from HF; pull ALL datasets from the existing deploy_full payload
ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$IP" "HF_REPO='$HF_REPO' bash -s" <<'REMOTE'
set -e
mkdir -p /workspace/hf /workspace/logs
hf download "$HF_REPO" --type dataset --include "rerun2026/code/*" --local-dir /workspace/hf
mkdir -p /workspace/src /workspace/scripts
cp -r /workspace/hf/rerun2026/code/src/. /workspace/src/
cp -r /workspace/hf/rerun2026/code/scripts/. /workspace/scripts/
GIT_REV=$(cat /workspace/hf/rerun2026/code/GIT_REV)
echo "$GIT_REV" > /workspace/GIT_REV
cd /workspace
TOKEN=$(cat /root/.cache/huggingface/token)
wget -q --header="Authorization: Bearer ${TOKEN}" \
  "https://huggingface.co/datasets/$HF_REPO/resolve/main/deploy_full.tar.gz" -O /tmp/deploy_full.tar.gz
tar xzf /tmp/deploy_full.tar.gz data --no-same-owner --no-same-permissions || true
rm -f /tmp/deploy_full.tar.gz
mkdir -p /workspace/results_rerun /workspace/ckpt_v2 /workspace/proc_v2
ls /workspace/data/
REMOTE

# [6] Pod-side runner (role-dependent stages)
ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$IP" "ROLE='$ROLE' bash -c 'cat > /workspace/runner.sh'" <<'REMOTE'
#!/usr/bin/env bash
cd /workspace
export S1_RESULTS_DIR=/workspace/results_rerun
export S1_CKPT_DIR=/workspace/ckpt_v2
export S1_PROCESSED_DIR=/workspace/proc_v2
export S1_RUN_ID=rerun2026_$(date -u +%Y%m%d)
export S1_DETERMINISTIC=1
export S1_GIT_REV=$(cat /workspace/GIT_REV)
export S1_TORCH_THREADS=2
ROLE="${ROLE:-A}"
log() { echo "[$(date -u +%H:%M:%S)] $*" >> logs/status.log; }

run_stage() {
  name="$1"; shift
  log "[start] $name"
  "$@" > "logs/${name}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "[FAILED] $name rc=$rc"
    tail -5 "logs/${name}.log" >> logs/status.log
    exit $rc
  fi
  log "[done] $name"
}

# SMOKE first (never-verified v2 path): tiny data subset, waits for pass
run_stage smoke python3 -u -m src.s1.run_experiment --dataset cic \
  --attack label_flip --ratio 0.05 --seed 0 --smoke

if [ "$ROLE" = "A" ]; then
  run_stage camp_cic    python3 -u -m src.s1.run_campaign_v2 --dataset cic --workers 6
  run_stage camp_ciciot python3 -u -m src.s1.run_campaign_v2 --dataset ciciot2023 --workers 6
  # Section IV probes on the new generation (CIC-based)
  run_stage probe_flag   python3 -u scripts/probe_flag_stability.py
  run_stage probe_geom   python3 -u scripts/probe_gradient_geometry.py
  run_stage probe_whiten python3 -u scripts/probe_whitening_rescue.py
else
  run_stage camp_nf python3 -u -m src.s1.run_campaign_v2 --dataset nf --workers 8
  # Merge pod A's results (cells + manifest fragments) from HF, then the
  # single-derivation aggregation + chain verification run on the FULL 195
  hf download "hunopapa/vifr-campaign-data" --type dataset \
    --include "rerun2026/outputs/results_rerun/*" \
    --local-dir /workspace/hf_merge >> logs/merge.log 2>&1
  mkdir -p /workspace/results_rerun/manifest
  cp -a /workspace/hf_merge/rerun2026/outputs/results_rerun/. \
    /workspace/results_rerun/ >> logs/merge.log 2>&1
  log "[done] merge_pod_A"
  run_stage aggregate python3 -u scripts/aggregate_v2.py
  run_stage verify    python3 -u scripts/verify_evidence.py
fi

# Full checkpoint tar per role — lineage never lost again
tar czf "/workspace/ckpt_v2_${ROLE}.tgz" -C /workspace ckpt_v2 \
  results_rerun/EVIDENCE_MANIFEST.json 2>/dev/null || \
  tar czf "/workspace/ckpt_v2_${ROLE}.tgz" -C /workspace ckpt_v2
hf upload "hunopapa/vifr-campaign-data" "/workspace/ckpt_v2_${ROLE}.tgz" \
  "rerun2026/outputs/ckpt_v2_${ROLE}.tgz" --type dataset >> logs/upload.log 2>&1
hf upload "hunopapa/vifr-campaign-data" /workspace/results_rerun \
  rerun2026/outputs/results_rerun --type dataset >> logs/upload.log 2>&1
hf upload "hunopapa/vifr-campaign-data" /workspace/logs \
  "rerun2026/outputs/logs_${ROLE}" --type dataset >> logs/upload.log 2>&1
log "[done] uploads role=${ROLE}"
REMOTE

# [7] Wait for pip; verify imports + GPU + v2 entry points
ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$IP" "cd /workspace; while [ ! -f .pip_done ]; do sleep 3; done; tail -2 pip_install.log; python3 -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'; python3 -c 'from src.s1.run_campaign_v2 import worker; from src.s1.certify_rs import certify_dual; from src.s1 import manifest; print(\"v2 imports OK\")'" </dev/null

# [8] Launch
ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$IP" "cd /workspace && chmod +x runner.sh && nohup bash runner.sh > logs/driver.log 2>&1 </dev/null & sleep 8; tail -3 logs/status.log 2>/dev/null; echo LAUNCHED" </dev/null

echo ""
echo "[NEXT] monitor: ssh -p $PORT root@$IP 'tail -f /workspace/logs/status.log' </dev/null"
echo "[NEXT] when status.log shows [done] uploads: bash fetch_rerun.sh"
echo "[NEXT] then: bash cleanup_pod.sh $POD_NAME"
