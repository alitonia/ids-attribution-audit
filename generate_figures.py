"""Generate paper figures from campaign results.

Outputs (paths expected by paper/main.tex):
  paper/fig_auroc.pdf    localization AUROC vs poison ratio, 3 dataset panels
  paper/fig_rs_radii.pdf certified accuracy vs radius, CIC clean vs quarantined
  rs_radii_table.tex     LaTeX table of certified accuracy at sampled radii,
                         all three datasets (clean vs quarantined, attack
                         class split by poisoning family)

Usage:
    .venv/bin/python generate_figures.py [--method tracincp|trak]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 300,
    "lines.linewidth": 1.3,
})

DATASETS = [("cic", "CIC-UNSW-NB15"), ("nf", "NF-ToN-IoT"), ("ciciot2023", "CICIoT2023")]
ATTACKS = [("label_flip", "Label flip"), ("feature_perturb", "Feature perturb."),
           ("trigger", "Trigger")]
RATIOS = (0.01, 0.02, 0.05, 0.10)
RADII = (0.0, 0.25, 0.5, 0.75, 1.0)
ATTACK_COLORS = {"label_flip": "#1f77b4", "feature_perturb": "#d62728",
                 "trigger": "#2ca02c"}


def load_cells():
    poisoned, clean_rs = [], []
    for f in sorted(glob.glob("results/exp_*.json")):
        if "smoke" in f:
            continue
        with open(f) as fp:
            d = json.load(fp)
        if d.get("attack") != "none":
            poisoned.append(d)
    return poisoned


def make_auroc_fig(poisoned, method: str) -> None:
    data = {dset: {atk: defaultdict(list) for atk, _ in ATTACKS} for dset, _ in DATASETS}
    for cell in poisoned:
        v = cell.get("localization", {}).get(method, {}).get("auroc")
        if v is None:
            continue
        data[cell["dataset"]][cell["attack"]][cell["ratio"]].append(v)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for ax, (dset, dset_name) in zip(axes, DATASETS):
        any_line = False
        for atk, atk_name in ATTACKS:
            means, stds, xs = [], [], []
            for rat in RATIOS:
                vals = data[dset][atk].get(rat, [])
                if vals:
                    xs.append(rat * 100)
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
            if xs:
                any_line = True
                ax.errorbar(xs, means, yerr=stds, label=atk_name,
                            color=ATTACK_COLORS[atk], marker="osd"[ATTACKS.index((atk, atk_name))],
                            markersize=3.5, capsize=2)
        ax.axhline(0.5, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_title(dset_name)
        ax.set_xlabel(r"Poisoning ratio $\rho$ (%)")
        ax.set_xticks([1, 2, 5])
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
        if not any_line:
            ax.text(3, 0.5, f"no {method} data", ha="center", color="0.4", fontsize=8)
    axes[0].set_ylabel(f"Localization AUROC ({method})")
    axes[0].legend(loc="lower left", framealpha=0.6)
    fig.tight_layout()
    fig.savefig("paper/fig_auroc.pdf")
    plt.close(fig)


def load_rs():
    if not os.path.exists("results/rs_results.json"):
        return {}
    with open("results/rs_results.json") as fp:
        return json.load(fp)


def rs_curves(rs, key_filter):
    """Mean/std certified-accuracy curves per class over matching RS cells."""
    out = {}
    for c in (0, 1):
        curves = []
        for k, v in rs.items():
            if not key_filter(k):
                continue
            sc = str(c)
            if sc in v and all(f"r>={r}" in v[sc] for r in RADII):
                curves.append([v[sc][f"r>={r}"] for r in RADII])
        if curves:
            arr = np.array(curves)
            out[c] = (arr.mean(axis=0), arr.std(axis=0), len(curves))
    return out


def make_rs_fig(rs) -> None:
    quar = rs_curves(rs, lambda k: k.startswith("cic_") and "_none_" not in k)
    clean = rs_curves(rs, lambda k: k.startswith("cic_none"))
    fam = {atk: rs_curves(rs, lambda k, a=atk: k.startswith(f"cic_{a}_"))
           for atk, _ in ATTACKS}
    if not quar:
        print("no CIC RS cells found; skipping fig_rs_radii.pdf")
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for c, cname, color in ((0, "benign", "#1f77b4"), (1, "attack", "#d62728")):
        if c in clean:
            m, s, n = clean[c]
            ax.plot(RADII, m, color=color, linestyle="--", marker="x", markersize=4,
                    label=f"{cname}, clean (n={n})")
    if 0 in quar:
        m, s, n = quar[0]
        ax.plot(RADII, m, color="#1f77b4", marker="o", markersize=3.5,
                label=f"benign, quarantined (n={n})")
    for atk, atk_name in ATTACKS:
        if 1 in fam[atk]:
            m, s, n = fam[atk][1]
            ax.plot(RADII, m, color=ATTACK_COLORS[atk], marker="osd"[ATTACKS.index((atk, atk_name))],
                    markersize=3.5, label=f"attack, quar. {atk_name.lower()} (n={n})")
    ax.set_xlabel(r"Certified $\ell_2$ radius $r$")
    ax.set_ylabel("Certified accuracy")
    ax.set_title("CIC-UNSW-NB15, $\\sigma=0.5$, $N=200$")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
    ax.legend(loc="lower left", framealpha=0.6)
    fig.tight_layout()
    fig.savefig("paper/fig_rs_radii.pdf")
    plt.close(fig)


def write_rs_table(rs) -> None:
    lines = []
    lines.append("\\begin{table*}[tbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\caption{Test-time evasion certificate (Cohen randomized "
                 "smoothing, $\\sigma=0.5$, $N=200$ Monte Carlo samples, "
                 "confidence $1-10^{-3}$) on all "
                 "three datasets. Certified accuracy = fraction of test flows "
                 "correctly classified \\emph{and} certified at radius $r$. "
                 "Clean: clean-trained detector; quarantined: retrained after the "
                 "audit's quarantine, pooled over all poisoning ratios "
                 "(mean $\\pm$ std; $n=36$ quarantine runs per dataset, $12$ per "
                 "attack family, $3$ clean runs). Label-flip and trigger "
                 "quarantines preserve the clean attack-class certificate "
                 "(on NF-ToN-IoT they improve on the weak clean certificate), "
                 "while feature-perturbation quarantines keep plain accuracy but "
                 "lose margin robustness on every dataset. The weak clean "
                 "certificates of the NF-ToN-IoT attack class and the CICIoT2023 "
                 "benign class are properties of the clean detectors, not of the "
                 "audit.}")
    lines.append("\\label{tab:rs_radii}")
    lines.append("\\begin{tabular}{ll rr rrrr}")
    lines.append("\\toprule")
    lines.append("& & \\multicolumn{2}{c}{Benign} & \\multicolumn{4}{c}{Attack} \\\\")
    lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-8}")
    lines.append("Dataset & $r$ & clean & quar. & clean & quar. flip "
                 "& quar. perturb. & quar. trig. \\\\")
    lines.append("\\midrule")

    def cell(curves, c, i):
        if c not in curves:
            return "--"
        m, s, _ = curves[c]
        return "%.3f\\,{\\footnotesize$\\pm$%.3f}" % (m[i], s[i])

    for di, (dset, dset_name) in enumerate(DATASETS):
        pref = dset + "_"
        quar = rs_curves(rs, lambda k, p=pref: k.startswith(p) and "_none_" not in k)
        clean = rs_curves(rs, lambda k, p=pref: k.startswith(p + "none_"))
        fam = {atk: rs_curves(rs, lambda k, p=pref, a=atk: k.startswith(p + a + "_"))
               for atk, _ in ATTACKS}
        for i, r in enumerate(RADII):
            ds_cell = dset_name if i == 0 else ""
            lines.append(f"{ds_cell} & {r:.2f} & {cell(clean, 0, i)} & {cell(quar, 0, i)} "
                         f"& {cell(clean, 1, i)} & {cell(fam['label_flip'], 1, i)} "
                         f"& {cell(fam['feature_perturb'], 1, i)} & {cell(fam['trigger'], 1, i)} \\\\")
        if di < len(DATASETS) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    with open("rs_radii_table.tex", "w") as out:
        out.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["tracincp", "trak"], default="tracincp")
    args = ap.parse_args()

    poisoned = load_cells()
    make_auroc_fig(poisoned, args.method)
    rs = load_rs()
    make_rs_fig(rs)
    write_rs_table(rs)
    print("wrote paper/fig_auroc.pdf, paper/fig_rs_radii.pdf, rs_radii_table.tex")


if __name__ == "__main__":
    main()
