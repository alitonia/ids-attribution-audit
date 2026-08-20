"""S1 experiment configuration — all decisions pre-committed in the project tracker.

Red-team mandated scoping lives in the paper; this file encodes the experimental
decisions. Change nothing here mid-sprint without recording it in ITERATION_TRACKER.md.
"""
import os
from pathlib import Path
import torch

# Enable TensorFloat-32 (TF32) for massive speedups on Ada Lovelace (RTX 4090)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = PROCESSED_DIR / "checkpoints"

# --- determinism -------------------------------------------------------------
SEEDS = list(range(30))          # final campaign; smoke runs use SEEDS[:3]
SMOKE_SEEDS = SEEDS[:3]

# --- poisoning protocol ------------------------------------------------------
POISON_RATIOS = (0.01, 0.02, 0.05, 0.10)
ATTACK_TYPES = ("label_flip", "feature_perturb", "trigger")
TRIGGER_TARGET_LABEL = 0         # poisons aim to look benign
PERTURB_BUDGET_STD = 1.0         # per-feature l_inf budget in standardized units

# --- attribution scoring grid (pre-committed minimal sweep) -------------------
PROJECTOR_DIMS = (1024, 2048)
K_CHECKPOINTS = (5, 8)
FPR_TARGET = 0.01                # threshold calibrated on clean flows, <=1% FPR

# --- gates (dates enforced by the human; numbers enforced by evaluate.py) -----
GATE_G2_SMOKE_AUROC = 0.60       # else TracInCP becomes primary method
GATE_G3_FULL_AUROC = 0.70        # @5% poison; else audit-comparison framing

# --- IDS classifier ------------------------------------------------------------
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.1
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --- machine safety (2026-08-12 freeze incident) ------------------------------
# Locally 4 threads keeps the 12-core laptop responsive; on the pod the driver
# exports S1_TORCH_THREADS=2 so 8 workers do not oversubscribe the vCPUs.
TORCH_THREADS = int(os.environ.get("S1_TORCH_THREADS", "4"))
USE_AMP = False                  # do not enable without explicit user approval


# --- device selection ----------------------------------------------------------
def resolve_device() -> torch.device:
    """Resolve the compute device for the S1 pipeline.

    Reads the S1_DEVICE env var: "cpu", "cuda", or "auto" (default).
    "auto" picks cuda when torch.cuda.is_available(), else cpu — so the same
    code runs unchanged on the CPU-only laptop and on a CUDA pod. An explicit
    "cuda" with no visible device raises instead of silently switching device,
    since the device is part of the experimental record.
    """
    raw = os.environ.get("S1_DEVICE", "auto").strip().lower()
    if raw not in ("cpu", "cuda", "auto"):
        raise ValueError(f"S1_DEVICE must be one of cpu|cuda|auto, got {raw!r}")
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "S1_DEVICE=cuda but torch.cuda.is_available() is False")
    return torch.device(raw)


def __getattr__(name: str):  # PEP 562: lazily-evaluated DEVICE constant
    if name == "DEVICE":
        return resolve_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# --- datasets ------------------------------------------------------------------
# CIC-UNSW-NB15 (2024 refresh) as delivered by the CIC download form (2026-08-18):
#   Dataset.csv: 447,915 flows x 76 numeric CICFlowMeter features (IDs removed)
#   Label.csv:   integer class labels 0-9; 0 = benign (358,332), class 9 = 246 rows
#   CICFlowMeter.csv: raw 84-column version incl. IPs — provenance only, not loaded
CIC_UNSW_FILES = ("Dataset.csv", "Label.csv")
NF_TONIoT_PARQUET = DATA_DIR / "NF-ToN-IoT.parquet"
