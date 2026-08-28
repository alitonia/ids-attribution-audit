#!/usr/bin/env python3
"""v2 aggregation — the single derivation point for every paper number.

Reads the env-scoped rerun generation (S1_RESULTS_DIR), emits:
  HEADLINE_NUMBERS.json  — every claim the paper quotes, bootstrap CIs,
                           pre-registered acceptance checks
  TABLES_INPUT.json      — per-cell aggregates the sync-phase table builder
                           consumes (same rounding conventions as the paper)
and consolidates the evidence manifest.

Usage (env must match the campaign): see run_rerun.sh
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("S1_RESULTS_DIR", str(REPO / "results" / "rerun2026"))
os.environ.setdefault("S1_CKPT_DIR", str(REPO / "data" / "processed" / "checkpoints_v2"))
os.environ.setdefault("S1_PROCESSED_DIR", str(REPO / "data" / "processed" / "v2"))

from src.s1 import config  # noqa: E402  (env-scoped dirs resolve here)

DATASETS = ["cic", "nf", "ciciot2023"]
ATTACKS = ["label_flip", "feature_perturb", "trigger"]
RATIOS = [0.01, 0.02, 0.05, 0.10]
SEEDS = [0, 1, 2, 3, 4]
R_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_all():
    cells = {}
    for d in DATASETS:
        for s in SEEDS:
            p = config.RESULTS_DIR / f"exp_{d}_none_seed{s}_K5.json"
            if p.exists():
                cells[(d, "none", 0.0, s)] = json.loads(p.read_text())
        for a in ATTACKS:
            for r in RATIOS:
                for s in SEEDS:
                    p = config.RESULTS_DIR / f"exp_{d}_{a}_r{r:g}_seed{s}_K5.json"
                    if p.exists():
                        cells[(d, a, r, s)] = json.loads(p.read_text())
    return cells


def clean_recall(cells, d, s):
    return cells[(d, "none", 0.0, s)]["metrics_poisoned"]["per_class_recall"]


def cell_damage(c, clean_pcr):
    worst = 0.0
    for cls, rec in c["metrics_poisoned"].get("per_class_recall", {}).items():
        ref = clean_pcr.get(cls)
        if ref is None:
            continue
        worst = max(worst, (ref["recall"] - rec["recall"]) * 100.0)
    return worst


def boot_ci(vals, n=2000, seed=0, alpha=0.05):
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return [float(v.mean()), float(v.mean())]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2))]


def main():
    cells = load_all()
    n_poison = sum(1 for k in cells if k[1] != "none")
    n_clean = sum(1 for k in cells if k[1] == "none")
    print(f"[agg] loaded {n_poison} poisoned + {n_clean} clean cells "
          f"(expect 180 + 15)")
    if n_poison != 180 or n_clean != 15:
        print("[agg] INCOMPLETE generation — aborting aggregation")
        sys.exit(1)

    H = {"n_cells": {"poisoned": n_poison, "clean": n_clean}}

    # ---- damage ----------------------------------------------------------
    dmg = {(k): cell_damage(c, clean_recall(cells, k[0], k[3]))
           for k, c in cells.items() if k[1] != "none"}
    all_dmg = list(dmg.values())
    H["damage"] = {
        "median_pp": float(np.median(all_dmg)),
        "max_pp": float(np.max(all_dmg)),
        "max_cell": repr(max(dmg, key=dmg.get)),
        "per_dataset_median": {d: float(np.median([v for k, v in dmg.items()
                                                   if k[0] == d]))
                               for d in DATASETS},
        "nf_worst": float(max(v for k, v in dmg.items() if k[0] == "nf")),
        "nf_worst_cell": repr(max((k for k in dmg if k[0] == "nf"),
                                  key=lambda k: dmg[k])),
        "ciciot_worst": float(max(v for k, v in dmg.items()
                                  if k[0] == "ciciot2023")),
        "cic_frac_zero": float(np.mean([v <= 0.0 for k, v in dmg.items()
                                        if k[0] == "cic"])),
        "nf_f1_max_shift_pp": float(max(
            abs((c["metrics_poisoned"]["f1"] - c["clean_f1"]) * 100)
            for k, c in cells.items() if k[0] == "nf" and k[1] != "none")),
    }

    # ---- localization ------------------------------------------------------
    def per_cell(m, attack=None, ds=None):
        return {(d, a, r, s): cells[(d, a, r, s)]["localization"][m]["auroc"]
                for (d, a, r, s) in cells
                if a != "none" and (attack is None or a == attack)
                and (ds is None or d in ds)}

    def seed_means(m, attack=None, ds=None):
        pc = per_cell(m, attack, ds)
        keys = {(k[0], k[1], k[2]) for k in pc}
        return {k: float(np.mean([pc[(k[0], k[1], k[2], s)] for s in SEEDS]))
                for k in keys}

    sm_at = seed_means("tracincp")
    sm_at_vals = list(sm_at.values())
    lf = [v for k, v in sm_at.items() if k[1] == "label_flip"]
    fp = [v for k, v in sm_at.items() if k[1] == "feature_perturb"]
    trg_cic = [v for k, v in sm_at.items() if k[1] == "trigger" and k[0] == "cic"]
    trg_nc = [v for k, v in sm_at.items()
              if k[1] == "trigger" and k[0] in ("nf", "ciciot2023")]
    trak_lf_ds = {d: float(np.mean(list(per_cell("trak", "label_flip", [d]).values())))
                  for d in DATASETS}
    trak_fp_ds = {d: float(np.mean(list(per_cell("trak", "feature_perturb",
                                                 [d]).values())))
                  for d in DATASETS}
    H["localization"] = {
        "tracincp_seedmean_cell_range": [min(sm_at_vals), max(sm_at_vals)],
        "tracincp_perrun_min": float(min(per_cell("tracincp").values())),
        "tracincp_perrun_max": float(max(per_cell("tracincp").values())),
        "label_flip_range": [min(lf), max(lf)],
        "feature_perturb_range": [min(fp), max(fp)],
        "trigger_cic_range": [min(trg_cic), max(trg_cic)],
        "trigger_nf_ciciot_range": [min(trg_nc), max(trg_nc)],
        "trak_labelflip_dataset_means": trak_lf_ds,
        "trak_labelflip_span": [min(trak_lf_ds.values()), max(trak_lf_ds.values())],
        "trak_featureperturb_dataset_means": trak_fp_ds,
        "trak_featureperturb_span": [min(trak_fp_ds.values()),
                                     max(trak_fp_ds.values())],
    }

    # ---- flag precision (tracincp, by rho) ---------------------------------
    H["flag_precision_by_rho"] = {
        d: {a: {str(r): float(np.mean(
            [cells[(d, a, r, s)]["localization"]["tracincp"]["flag_precision"]
             for s in SEEDS])) for r in RATIOS} for a in ATTACKS}
        for d in DATASETS}

    # ---- two-sided trigger recovery + held-out FPR ------------------------
    ts = [cells[k]["two_sided"]["tracincp"] for k in cells if k[1] == "trigger"]
    one = [cells[k]["localization"]["tracincp"]["flag_precision"]
           for k in cells if k[1] == "trigger"]
    ho = [cells[k]["threshold_heldout"]["tracincp"]["empirical_fpr_half_b"]
          for k in cells if k[1] != "none"]
    H["two_sided_trigger"] = {
        "one_sided_mean_precision": float(np.mean(one)),
        "two_sided_mean_precision": float(np.mean([t["flag_precision"] for t in ts])),
        "lo_tail_mean_precision": float(np.mean([t["lo_tail"]["precision"] for t in ts])),
        "hi_tail_mean_precision": float(np.mean([t["hi_tail"]["precision"] for t in ts])),
    }
    H["heldout_fpr"] = {"mean": float(np.mean(ho)),
                        "p95": float(np.quantile(ho, 0.95)),
                        "target": config.FPR_TARGET}

    # ---- dF1: pure-TracInCP quarantine ------------------------------------
    df1 = {}
    for k, c in cells.items():
        if k[1] == "none":
            continue
        q = c.get("metrics_quarantined_tracincp")
        if q:
            df1[k] = (q["f1"] - c["clean_f1"]) * 100.0
    le5 = [v for k, v in df1.items() if k[2] <= 0.05]
    H["df1_pure_tracincp"] = {
        "median": float(np.median(list(df1.values()))),
        "mean": float(np.mean(list(df1.values()))),
        "rho_le5_range": [float(min(le5)), float(max(le5))],
        "mean_ci95": boot_ci(list(df1.values())),
        "n_cells": len(df1),
    }
    worst_df1 = max(df1, key=lambda k: abs(df1[k]))
    H["df1_pure_tracincp"]["worst_cell"] = repr(worst_df1)
    H["df1_pure_tracincp"]["worst_abs"] = abs(df1[worst_df1])

    # ---- collapse diagnostics ----------------------------------------------
    flags = {k: cells[k]["score_diagnostics"]["tracincp"]["flag_rate"]
             for k in cells if k[1] != "none"}
    wf = max(flags, key=flags.get)
    H["collapse_watch"] = {
        "max_flag_rate": float(flags[wf]),
        "max_flag_cell": repr(wf),
        "cells_flag_rate_gt50pct": int(sum(1 for v in flags.values() if v > 0.5)),
    }

    # ---- RS certificates (clean + pure-TracInCP quarantine, exact + wald) ---
    rs_bad = [k.__repr__() for k, c in cells.items()
              if "error" in c.get("rs_cert", {}).get("classes", {})]
    rs_out = {}
    for d in DATASETS:
        rs_out[d] = {"clean": {}, "quar": {a: {} for a in ATTACKS}}
        for cls in ("0", "1"):
            for rad in R_GRID:
                cv = [cells[(d, "none", 0.0, s)]["rs_cert"]["classes"][cls]
                      [f"r>={rad}"] for s in SEEDS
                      if "rs_cert" in cells[(d, "none", 0.0, s)]
                      and "error" not in cells[(d, "none", 0.0, s)]["rs_cert"]
                      .get("classes", {})]
                rs_out[d]["clean"][f"{cls}@{rad}"] = {
                    "exact_mean": float(np.mean([v["exact"] for v in cv])),
                    "exact_std": float(np.std([v["exact"] for v in cv])),
                    "wald_mean": float(np.mean([v["wald"] for v in cv])),
                    "n": len(cv)}
                for a in ATTACKS:
                    qv = [cells[(d, a, r, s)]["rs_cert"]["classes"][cls][f"r>={rad}"]
                          for r in RATIOS for s in SEEDS
                          if "rs_cert" in cells[(d, a, r, s)]
                          and "error" not in cells[(d, a, r, s)]["rs_cert"]
                          .get("classes", {})]
                    rs_out[d]["quar"][a][f"{cls}@{rad}"] = {
                        "exact_mean": float(np.mean([v["exact"] for v in qv])),
                        "exact_std": float(np.std([v["exact"] for v in qv])),
                        "wald_mean": float(np.mean([v["wald"] for v in qv])),
                        "n": len(qv)}
    H["rs_certified"] = rs_out
    H["rs_excluded_cells"] = rs_bad

    # ---- acceptance checks (pre-registered before the run) ------------------
    H["acceptance"] = {
        "localization_lf_fp_seedmean_min_ge_0.84": bool(min(lf + fp) >= 0.84),
        "median_damage_le_0.05pp": bool(H["damage"]["median_pp"] <= 0.05),
        "median_abs_df1_le_0.5pp":
            bool(abs(H["df1_pure_tracincp"]["median"]) <= 0.5),
        "heldout_fpr_p95_le_2pct": bool(H["heldout_fpr"]["p95"] <= 0.02),
        "rs_featpert_degrades_vs_clean_on_all_datasets": bool(all(
            rs_out[d]["quar"]["feature_perturb"]["1@0.0"]["exact_mean"]
            < rs_out[d]["clean"]["1@0.0"]["exact_mean"] - 0.05
            for d in DATASETS)),
    }

    # ---- table input for the sync-phase LaTeX builder -----------------------
    T = {"rows_table1": [], "rows_table3": []}
    for a in ATTACKS:
        for r in RATIOS:
            row = {"attack": a, "ratio": r}
            for d in DATASETS:
                row[d] = {
                    "auroc": float(np.mean([per_cell("tracincp", a, [d])[(d, a, r, s)]
                                            for s in SEEDS])),
                    "prec": float(np.mean(
                        [cells[(d, a, r, s)]["localization"]["tracincp"]
                         ["flag_precision"] for s in SEEDS])),
                    "dmg": float(np.mean([dmg[(d, a, r, s)] for s in SEEDS])),
                    "df1": float(np.mean([df1.get((d, a, r, s), np.nan)
                                          for s in SEEDS])),
                    "df1_gated": float(np.mean(
                        [(cells[(d, a, r, s)].get("metrics_quarantined", {})
                          .get("f1", np.nan) - cells[(d, a, r, s)]["clean_f1"]) * 100
                         for s in SEEDS])),
                }
            T["rows_table1"].append(row)
    for d in DATASETS:
        for cls in ("0", "1"):
            for rad in R_GRID:
                T["rows_table3"].append({
                    "dataset": d, "cls": cls, "r": rad,
                    "clean": rs_out[d]["clean"][f"{cls}@{rad}"],
                    **{a: rs_out[d]["quar"][a][f"{cls}@{rad}"] for a in ATTACKS}})

    (config.RESULTS_DIR / "HEADLINE_NUMBERS.json").write_text(json.dumps(H, indent=2))
    (config.RESULTS_DIR / "TABLES_INPUT.json").write_text(json.dumps(T, indent=2))

    from src.s1 import manifest
    man = manifest.consolidate()
    hsha = hashlib.sha256((config.RESULTS_DIR / "HEADLINE_NUMBERS.json")
                          .read_bytes()).hexdigest()
    print(f"[agg] HEADLINE_NUMBERS sha256={hsha[:16]}")
    print(f"[agg] manifest cells={man['n_cells']} "
          f"consistent={man['generation_consistent']}")
    print("[agg] acceptance:", json.dumps(H["acceptance"], indent=2))


if __name__ == "__main__":
    main()
