"""Aggregate campaign JSONs into LaTeX tables (results + baselines).

Metrics (full-data reality after the 2026-08-20 fixed re-run, 108 poisoned
cells): poisoning at rho <= 5% of full training sets is almost always SILENT —
median per-class recall damage 0.0pp; CICIoT2023 <= 0.2pp in every run;
CIC-UNSW-NB15 <= 4.2pp seed-averaged (worst single run 12.7pp, one
trigger-pattern cell on a 726-sample class); NF-ToN-IoT worst-hit class up to
69.1pp in the worst single run (26.4pp seed-averaged at rho=1% label-flip)
while aggregate F1 stays within 1.5pp. The smoke-scale (20K subset) "recover
the harmed class" narrative does not survive at full scale. The honest
full-data story the tables must tell:

  1. Dmg:  poisoning causes (almost) no visible damage -> aggregate
           monitoring has no signal to trigger an audit;
  2. AUROC/Prec.: the attribution audit localizes the planted poisons anyway;
  3. dF1:  quarantining flagged flows and retraining costs ~0 aggregate F1
           at rho <= 5% (median +0.2pp; one threshold-collapse run at rho=10%
           CIC label-flip costs 32.6pp).

Per-cell definitions (aligned by (dataset, seed) with the clean baseline):
    damage_c = recall_clean_c - recall_poison_c   per multiclass label c
    Dmg      = max over classes and seeds of damage_c (percentage points)
    dF1      = F1(quarantined model) - F1(clean model) (percentage points)

Usage:
    .venv/bin/python aggregate_latex.py [--method tracincp|trak]
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

DATASETS = [("cic", "CIC-UNSW-NB15"), ("nf", "NF-ToN-IoT"), ("ciciot2023", "CICIoT2023")]
ATTACKS = [("label_flip", "Label Flip"), ("feature_perturb", "Feature Perturb"),
           ("trigger", "Trigger")]
RATIOS = (0.01, 0.02, 0.05, 0.10)
SEEDS = (0, 1, 2)
METHODS = ["trak", "tracincp", "loss_outlier", "grad_norm",
           "activation_clustering", "random", "oracle"]


def load_results(results_dir: str):
    clean = {}            # (dset, seed) -> per_class_recall dict
    poisoned = []         # list of cell dicts
    for f in sorted(glob.glob(f"{results_dir}/exp_*.json")):
        if "smoke" in f:
            continue
        with open(f) as fp:
            d = json.load(fp)
        if d.get("attack") == "none":
            clean[(d["dataset"], d["seed"])] = d.get("metrics_poisoned", {}).get("per_class_recall")
        else:
            poisoned.append(d)
    return clean, poisoned


def max_damage_pp(cell: dict, clean_pcr):
    """Max per-class recall damage (pp) for a cell, or None if uncomputable."""
    pois_pcr = cell.get("metrics_poisoned", {}).get("per_class_recall")
    if not pois_pcr or not clean_pcr:
        return None
    dams = [clean_pcr[c]["recall"] - pv["recall"]
            for c, pv in pois_pcr.items() if c in clean_pcr]
    return 100.0 * max(dams) if dams else None


def fmt(vals, nd=3):
    return f"{np.mean(vals):.{nd}f}" if vals else "--"


def write_results_table(cells_by_key, clean, method: str) -> None:
    method_name = {"tracincp": "TracInCP", "trak": "TRAK"}[method]
    lines = []
    lines.append("\\begin{table*}[tbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\caption{Audit results with the %s attributor over %d seeds. "
                 "AUROC: localization of planted poisons; Prec.: precision of the "
                 "flagged set at the threshold calibrated for $\\leq 1\\%%$ FPR on clean "
                 "flows; Dmg.: worst per-class recall damage caused by the poisoning, "
                 "averaged over seeds (pp); $\\Delta$F1: aggregate F1 after "
                 "quarantine+retrain minus clean F1 (pp). Poisoning is silent "
                 "(Dmg. $\\approx 0$) yet localizable.}"
                 % (method_name, len(SEEDS)))
    lines.append("\\label{tab:results}")
    lines.append("\\begin{tabular}{ll rrrr rrrr rrrr}")
    lines.append("\\toprule")
    lines.append("& & \\multicolumn{4}{c}{CIC-UNSW-NB15} & \\multicolumn{4}{c}{NF-ToN-IoT} "
                 "& \\multicolumn{4}{c}{CICIoT2023} \\\\")
    lines.append("\\cmidrule(lr){3-6} \\cmidrule(lr){7-10} \\cmidrule(lr){11-14}")
    lines.append("Attack & $\\rho$ (\\%) & AUROC & Prec. & Dmg. & $\\Delta$F1 "
                 "& AUROC & Prec. & Dmg. & $\\Delta$F1 "
                 "& AUROC & Prec. & Dmg. & $\\Delta$F1 \\\\")
    lines.append("\\midrule")
    for atk, atk_name in ATTACKS:
        for rat in RATIOS:
            row = f"{atk_name} & {rat * 100:.0f}"
            for dset, _ in DATASETS:
                cells = cells_by_key.get((dset, atk, rat), [])
                aurocs, precs, dams, df1s = [], [], [], []
                for cell in cells:
                    loc = cell.get("localization", {}).get(method, {})
                    if loc.get("auroc") is not None:
                        aurocs.append(loc["auroc"])
                        precs.append(loc.get("flag_precision", 0.0))
                    dm = max_damage_pp(cell, clean.get((dset, cell.get("seed"))))
                    if dm is not None:
                        dams.append(dm)
                    if cell.get("metrics_quarantined") and cell.get("clean_f1") is not None:
                        df1s.append(100.0 * (cell["metrics_quarantined"]["f1"] - cell["clean_f1"]))
                dmg_str = f"{np.mean(dams):+.1f}" if dams else "--"
                df1_str = f"{np.mean(df1s):+.1f}" if df1s else "--"
                row += f" & {fmt(aurocs)} & {fmt(precs)} & {dmg_str} & {df1_str}"
            lines.append(row + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    with open("results_table.tex", "w") as out:
        out.write("\n".join(lines) + "\n")


def write_baseline_table(poisoned) -> None:
    auroc = {m: defaultdict(list) for m in METHODS}
    for cell in poisoned:
        key = cell["dataset"]
        for m in METHODS:
            v = cell.get("localization", {}).get(m, {}).get("auroc")
            if v is not None:
                auroc[m][key].append(v)
    lines = []
    lines.append("\\begin{table*}[tbp]")
    lines.append("\\centering")
    lines.append("\\caption{Localization AUROC by method, averaged over all attacks, "
                 "ratios, and seeds. Oracle = planted-poison ground truth; "
                 "`--' where the method errored on every cell. TRAK values are "
                 "single projection/instance draws and are projection-sensitive "
                 "on these data (Section~\\ref{sec:guarantee}, Remark); TracInCP, "
                 "which uses no random projection, is the headline attributor.}")
    lines.append("\\label{tab:baselines}")
    lines.append("\\begin{tabular}{l" + "r" * len(METHODS) + "}")
    lines.append("\\toprule")
    names = {"trak": "TRAK", "tracincp": "TracInCP", "loss_outlier": "Loss",
             "grad_norm": "GradNorm", "activation_clustering": "Act.Clust",
             "random": "Random", "oracle": "Oracle"}
    lines.append("Dataset & " + " & ".join(names[m] for m in METHODS) + " \\\\")
    lines.append("\\midrule")
    for dset, dset_name in DATASETS:
        row = [dset_name]
        for m in METHODS:
            row.append(fmt(auroc[m].get(dset, [])))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    with open("baseline_table.tex", "w") as out:
        out.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["tracincp", "trak"], default="tracincp",
                    help="attribution method for the headline table (default: "
                         "tracincp, which leads TRAK on all three datasets)")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    clean, poisoned = load_results(args.results_dir)
    cells_by_key = defaultdict(list)
    for cell in poisoned:
        cells_by_key[(cell["dataset"], cell["attack"], cell["ratio"])].append(cell)

    write_results_table(cells_by_key, clean, args.method)
    write_baseline_table(poisoned)

    n_trak_ok = sum(1 for c in poisoned
                    if c.get("localization", {}).get("trak", {}).get("auroc") is not None)
    n_trak_err = sum(1 for c in poisoned if c.get("errors", {}).get("trak"))
    print(f"cells={len(poisoned)} clean_baselines={len(clean)} "
          f"trak_ok={n_trak_ok} trak_err={n_trak_err}")

    dams, df1s, inv = [], [], []
    for cell in poisoned:
        dm = max_damage_pp(cell, clean.get((cell["dataset"], cell.get("seed"))))
        if dm is not None:
            dams.append(dm)
        if cell.get("metrics_quarantined") and cell.get("clean_f1") is not None:
            df1s.append(100.0 * (cell["metrics_quarantined"]["f1"] - cell["clean_f1"]))
        v = cell.get("localization", {}).get(args.method, {}).get("auroc")
        if v is not None and v < 0.5:
            inv.append((cell["dataset"], cell["attack"], cell["ratio"], cell["seed"], v))
    if dams:
        print(f"damage pp: max={max(dams):.2f} median={np.median(dams):.2f} "
              f"(n={len(dams)} cells with clean per-class baselines)")
    if df1s:
        print(f"dF1 quar-clean pp: mean={np.mean(df1s):+.2f} min={min(df1s):+.2f} "
              f"max={max(df1s):+.2f}")
    if inv:
        print(f"WARNING: {len(inv)} inverted cells ({args.method} AUROC < 0.5):")
        for d in inv:
            print(f"  {d[0]} {d[1]} r={d[2]:.2f} s{d[3]}: {d[4]:.3f}")
    print(f"wrote results_table.tex (method={args.method}) and baseline_table.tex")


if __name__ == "__main__":
    main()
