#!/usr/bin/env python3
"""sync_v2 — rebuild tables + figures for the paper from the rerun2026 generation.

Outputs (staging under results/rerun2026/tables/ for review before swapping in):
  results_table_v2.tex    Table I  — TracInCP, pure-TracInCP quarantine ΔF1
  baseline_table_v2.tex   Table II — per-method mean localization AUROC
  rs_radii_table_v2.tex   Table III — EXACT Clopper-Pearson bounds
  paper_v2/fig_auroc.pdf, fig_rs_radii.pdf
  NUMBER_DIFF.md          old-paper → new-generation diff for the text pass

Run: python3 scripts/sync_v2.py
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "rerun2026"
OUT = RES / "tables"
FIGDIR = REPO / "paper_v2"

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 300, "lines.linewidth": 1.3,
})

DATASETS = [("cic", "CIC-UNSW-NB15"), ("nf", "NF-ToN-IoT"),
            ("ciciot2023", "CICIoT2023")]
ATTACKS = [("label_flip", "Label Flip"), ("feature_perturb", "Feature Perturb"),
           ("trigger", "Trigger")]
RATIOS = (0.01, 0.02, 0.05, 0.10)
RADII = (0.0, 0.25, 0.5, 0.75, 1.0)
COLORS = {"label_flip": "#1f77b4", "feature_perturb": "#d62728", "trigger": "#2ca02c"}


def load():
    cells = {}
    for f in sorted(glob.glob(str(RES / "exp_*_K5.json"))):
        if "_smoke" in f:
            continue
        d = json.loads(Path(f).read_text())
        cells[(d["dataset"], d["attack"], d["ratio"], d["seed"])] = d
    return cells


def clean_pcr(cells, d, s):
    return cells[(d, "none", 0.0, s)]["metrics_poisoned"]["per_class_recall"]


def damage(c, ref):
    worst = 0.0
    for cls, rec in c["metrics_poisoned"].get("per_class_recall", {}).items():
        r = ref.get(cls)
        if r is None:
            continue
        worst = max(worst, (r["recall"] - rec["recall"]) * 100.0)
    return worst


def main():
    cells = load()
    OUT.mkdir(exist_ok=True)
    FIGDIR.mkdir(exist_ok=True)
    seeds = list(range(5))

    # ---------- Table I ----------
    rows = []
    stats = defaultdict(list)
    for atk, atk_name in ATTACKS:
        for rat in RATIOS:
            cellsrow = []
            for d, _ in DATASETS:
                au = [cells[(d, atk, rat, s)]["localization"]["tracincp"]["auroc"]
                      for s in seeds]
                pr = [cells[(d, atk, rat, s)]["localization"]["tracincp"]["flag_precision"]
                      for s in seeds]
                dm = [damage(cells[(d, atk, rat, s)], clean_pcr(cells, d, s))
                      for s in seeds]
                q = [cells[(d, atk, rat, s)].get("metrics_quarantined_tracincp", {}).get("f1")
                     for s in seeds]
                df1 = [(qq - cells[(d, atk, rat, s)]["clean_f1"]) * 100
                       if qq is not None else float("nan")
                       for qq, s in zip(q, seeds)]
                stats["df1_all"] += [x for x in df1 if x == x]
                if rat <= 0.05:
                    stats["df1_le5"] += [x for x in df1 if x == x]
                cellsrow += [f"{np.mean(au):.3f}", f"{np.mean(pr):.3f}",
                             f"{np.mean(dm):+.1f}",
                             f"{np.mean(df1):+.1f}" if all(x == x for x in df1) else "--"]
            rows.append(" & ".join([atk_name, str(int(rat * 100))] + cellsrow) + " \\\\")
    with open(OUT / "results_table_v2.tex", "w") as f:
        f.write("\n".join(rows) + "\n")

    # ---------- Table II ----------
    methods = ["trak", "tracincp", "loss_outlier", "grad_norm",
               "activation_clustering", "random", "oracle"]
    t2 = []
    for d, dname in DATASETS:
        vals, trak_missing = [], 0
        for m in methods:
            vs = []
            for (dd, atk, rat, s), c in cells.items():
                if dd != d or atk == "none":
                    continue
                loc = c.get("localization", {}).get(m)
                if loc is None or loc.get("auroc") != loc.get("auroc"):
                    if m == "trak":
                        trak_missing += 1
                    continue
                vs.append(loc["auroc"])
            vals.append(f"{np.mean(vs):.3f}" if vs else "--")
        t2.append(f"{dname} & " + " & ".join(vals) + " & 1.000 \\\\")
    with open(OUT / "baseline_table_v2.tex", "w") as f:
        f.write("\n".join(t2) + "\n")

    # ---------- Table III + RS figure (EXACT bounds) ----------
    def rs_family(d, atk="__clean__"):
        """__clean__ = clean cells; None = ALL poisoned pooled; else one family."""
        out = defaultdict(list)
        for (dd, aa, rr, ss), c in cells.items():
            if dd != d:
                continue
            if atk == "__clean__":
                if aa != "none":
                    continue
            elif atk is None:
                if aa == "none":
                    continue
            elif aa != atk:
                continue
            rc = c.get("rs_cert", {}).get("classes", {})
            if "error" in rc:
                continue
            for cls in (0, 1):
                out[cls].append([rc[str(cls)][f"r>={r}"]["exact"] for r in RADII])
        return {c: (np.mean(v, axis=0), np.std(v, axis=0), len(v))
                for c, v in out.items() if v}

    def cell3(curves, c, i):
        if c not in curves:
            return "--"
        m, s, _ = curves[c]
        return "%.3f\\,{\\footnotesize$\\pm$%.3f}" % (m[i], s[i])

    t3 = []
    for di, (d, dname) in enumerate(DATASETS):
        clean = rs_family(d)
        fam = {atk: rs_family(d, atk) for atk, _ in ATTACKS}
        quar_all = rs_family(d, None)  # benign quarantined = all families pooled
        for i, r in enumerate(RADII):
            ds_cell = dname if i == 0 else ""
            t3.append(f"{ds_cell} & {r:.2f} & {cell3(clean,0,i)} & {cell3(quar_all,0,i)} "
                      f"& {cell3(clean,1,i)} & {cell3(fam['label_flip'],1,i)} "
                      f"& {cell3(fam['feature_perturb'],1,i)} & {cell3(fam['trigger'],1,i)} \\\\")
        if di < 2:
            t3.append("\\midrule")
    with open(OUT / "rs_radii_table_v2.tex", "w") as f:
        f.write("\n".join(t3) + "\n")

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    d = "cic"
    clean = rs_family(d)
    quar_all = rs_family(d, "any")
    fam = {atk: rs_family(d, atk) for atk, _ in ATTACKS}
    for c, cname, color in ((0, "benign", "#1f77b4"), (1, "attack", "#d62728")):
        if c in clean:
            m, s, n = clean[c]
            ax.plot(RADII, m, color=color, linestyle="--", marker="x",
                    markersize=4, label=f"{cname}, clean (n={n})")
    if 0 in quar_all:
        m, s, n = quar_all[0]
        ax.plot(RADII, m, color="#1f77b4", marker="o", markersize=3.5,
                label=f"benign, quarantined (n={n})")
    for atk, atk_name in ATTACKS:
        if 1 in fam[atk]:
            m, s, n = fam[atk][1]
            ax.plot(RADII, m, color=COLORS[atk], marker="osd"[ATTACKS.index((atk, atk_name))],
                    markersize=3.5, label=f"attack, quar. {atk_name.lower()} (n={n})")
    ax.set_xlabel(r"Certified $\ell_2$ radius $r$")
    ax.set_ylabel("Certified accuracy")
    ax.set_title("CIC-UNSW-NB15, $\\sigma=0.5$, $N=200$")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
    ax.legend(loc="lower left", framealpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_rs_radii.pdf")
    plt.close(fig)

    # ---------- AUROC figure ----------
    data = {d: {a: defaultdict(list) for a, _ in ATTACKS} for d, _ in DATASETS}
    for (d, a, r, s), c in cells.items():
        if a == "none":
            continue
        data[d][a][r].append(c["localization"]["tracincp"]["auroc"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for ax, (d, dname) in zip(axes, DATASETS):
        for atk, atk_name in ATTACKS:
            xs, ms, ss = [], [], []
            for rat in RATIOS:
                v = data[d][atk].get(rat, [])
                if v:
                    xs.append(rat * 100)
                    ms.append(np.mean(v))
                    ss.append(np.std(v))
            if xs:
                ax.errorbar(xs, ms, yerr=ss, label=atk_name, color=COLORS[atk],
                            marker="osd"[ATTACKS.index((atk, atk_name))],
                            markersize=3.5, capsize=2)
        ax.axhline(0.5, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_title(dname)
        ax.set_xlabel(r"Poisoning ratio $\rho$ (%)")
        ax.set_xticks([1, 2, 5, 10])
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
    axes[0].set_ylabel("Localization AUROC (TracInCP)")
    axes[0].legend(loc="lower left", framealpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_auroc.pdf")
    plt.close(fig)

    # ---------- number diff report ----------
    H = json.loads((RES / "HEADLINE_NUMBERS.json").read_text())
    difflines = [
        "# NUMBER DIFF — old paper → rerun2026 generation", "",
        "| quantity | old | new |", "|---|---|---|",
        f"| median damage (pp) | 0.0 | {H['damage']['median_pp']:.3f} |",
        f"| worst damage (pp / cell) | 69.1 | {H['damage']['max_pp']:.2f} / {H['damage']['max_cell']} |",
        f"| ciciot worst (pp) | 0.23 | {H['damage']['ciciot_worst']:.2f} |",
        f"| lf range (seed-mean) | 0.94–0.99 | "
        f"{H['localization']['label_flip_range'][0]:.3f}–{H['localization']['label_flip_range'][1]:.3f} |",
        f"| fp range (seed-mean) | 0.84–0.98 | "
        f"{H['localization']['feature_perturb_range'][0]:.3f}–{H['localization']['feature_perturb_range'][1]:.3f} |",
        f"| trigger CIC | 0.32–0.37 | "
        f"{H['localization']['trigger_cic_range'][0]:.3f}–{H['localization']['trigger_cic_range'][1]:.3f} |",
        f"| trigger NF/CICIoT | 0.84–0.95 | "
        f"{H['localization']['trigger_nf_ciciot_range'][0]:.3f}–{H['localization']['trigger_nf_ciciot_range'][1]:.3f} |",
        f"| dF1 median (pp) | +0.2 (gated) | {H['df1_pure_tracincp']['median']:+.2f} (pure TracInCP) |",
        f"| dF1 rho<=5 range (pp) | -3.6..+3.2 | "
        f"{H['df1_pure_tracincp']['rho_le5_range'][0]:.1f}..{H['df1_pure_tracincp']['rho_le5_range'][1]:.1f} |",
        f"| collapse (10%) | -32.6, flag 77.6% | {H['df1_pure_tracincp']['worst_abs']:.1f} at {H['df1_pure_tracincp']['worst_cell']}, flag {H['collapse_watch']['max_flag_rate']:.1%} |",
        f"| runs | 108 (3 seeds) | 180 (5 seeds) |",
        "",
        f"two_sided_trigger: {H['two_sided_trigger']}",
        f"heldout_fpr: {H['heldout_fpr']}",
        f"nf_f1_max_shift: {H['damage']['nf_f1_max_shift_pp']:.2f}",
        f"df1 rho<=5 cells < -5pp: "
        f"{[round(x,1) for x in sorted(stats['df1_le5'])[:4]]}",
    ]
    (OUT / "NUMBER_DIFF.md").write_text("\n".join(difflines) + "\n")
    print("\n".join(difflines))
    print("\nwrote tables + figures + NUMBER_DIFF.md")


if __name__ == "__main__":
    main()
