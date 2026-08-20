"""Evaluation: localization metrics, threshold calibration, quarantine + recovery.

Convention: every method's output is converted to a SUSPICION score where
HIGHER = more suspicious, before these functions are called (see
run_experiment.SUSPICION_SIGN). Metrics are dependency-free (rank-based AUROC).
"""
from __future__ import annotations

import json
import numpy as np

from . import config

# Attribution methods score HELPFULNESS (poisons score low) -> negate.
# Loss/gradient/outlier-type methods score ANOMALY (poisons score high) -> keep.
SUSPICION_SIGN = {
    "trak": -1.0,
    "tracincp": -1.0,
    "random": 1.0,
    "loss_outlier": 1.0,
    "grad_norm": 1.0,
    "activation_clustering": 1.0,
    "oracle": 1.0,
}


def localization_auroc(suspicion: np.ndarray, poison_mask: np.ndarray) -> float:
    """AUROC for 'poison vs clean' where higher suspicion = more likely poison."""
    s = np.asarray(suspicion, dtype=np.float64)
    labels = np.asarray(poison_mask, dtype=np.int64)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_sum_pos = ranks[labels == 1].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def precision_at_k(suspicion: np.ndarray, poison_mask: np.ndarray, k: int) -> float:
    s = np.asarray(suspicion, dtype=np.float64)
    flagged = np.argsort(s, kind="mergesort")[-k:]
    return float(poison_mask[flagged].mean()) if k > 0 else float("nan")


def calibrate_threshold(clean_suspicion: np.ndarray,
                        fpr_target: float = config.FPR_TARGET) -> float:
    """Threshold tau on clean flows: flag suspicion > tau, FPR <= fpr_target."""
    return float(np.quantile(np.asarray(clean_suspicion, dtype=np.float64),
                             1.0 - fpr_target))


def flag_flows(suspicion: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(suspicion) > tau


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "acc": (tp + tn) / max(len(y_true), 1)}


def per_class_recall(y_pred_binary: np.ndarray, y_multi: np.ndarray) -> dict:
    """Recall per true attack class (binary 'predicted attack' outcome).

    Binary F1 can hide poisoning damage under class-weighted training
    (verified 2026-08-18); per-class recall is the honest damage metric.
    """
    out = {}
    for c in sorted(np.unique(y_multi)):
        sel = y_multi == c
        if sel.sum() == 0:
            continue
        out[str(int(c))] = {"n": int(sel.sum()),
                            "recall": float(y_pred_binary[sel].mean())}
    return out


def f1_recovery(f1_poisoned: float, f1_quarantined: float, f1_clean: float) -> float:
    """Fraction of the poisoning-induced F1 loss recovered by quarantine."""
    denom = f1_clean - f1_poisoned
    if abs(denom) < 1e-9:
        return float("nan")
    return float((f1_quarantined - f1_poisoned) / denom)


def gate_status(auroc_smoke: float, auroc_full: float | None = None) -> dict:
    """Apply the pre-registered decision rule (ITERATION_TRACKER.md)."""
    out = {"G2_smoke": "PASS" if auroc_smoke >= config.GATE_G2_SMOKE_AUROC
           else "FAIL->TracInCP primary"}
    if auroc_full is not None:
        if auroc_full >= config.GATE_G3_FULL_AUROC:
            out["G3_full"] = "PASS->audit story"
        elif auroc_full >= config.GATE_G2_SMOKE_AUROC:
            out["G3_full"] = "MIDDLE->audit-comparison framing"
        else:
            out["G3_full"] = "FAIL->benchmark flavor, best auditor wins"
    return out


def save_result(path, payload: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
