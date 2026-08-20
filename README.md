# Auditing Poisoned Training Data for Intrusion Detection Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22035732.svg)](https://doi.org/10.5281/zenodo.22035732)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Artifact repository for the paper *"Auditing Poisoned Training Data for
Intrusion Detection Systems: An Influence-Based Attribution Approach for IoT
Network Security"* (Nguyen Huy Hoang, Hanoi University of Science and
Technology; RIVF 2026).

An influence-based data-attribution audit for IDS training data:
multi-checkpoint attribution scores over tabular flows, a flagging threshold
calibrated at a controlled false-positive rate, and a quarantine-and-retrain
protocol. Across three network-intrusion datasets and three attack families at
1-10% poisoning, the poisoning is almost always silent (median per-class
recall damage 0.0 pp) yet localizable at 0.85-0.99 AUROC (TracInCP), and
quarantine is near-costless up to 5% poisoning.

## Contents

- `src/s1/` — the audit pipeline: data loaders, poisoning, training,
  attribution scoring (TRAK / TracInCP / baselines), FPR-controlled
  calibration, quarantine, randomized-smoothing certification.
- `aggregate_latex.py` — regenerates the results and baseline tables from
  `results/`.
- `generate_figures.py` — regenerates the RS-certificate table and the paper
  figures (writes figures into `paper/`; run `mkdir -p paper` first).
- `scripts/` — the diagnostics: projection-stability probe, gradient-geometry
  probe (CIC + NF cells), and the whitening-rescue intervention probe.
- `results/` — the full 117-cell campaign (108 poisoned + 9 clean:
  3 datasets x 3 attacks x 4 ratios x 3 seeds), RS certificates for all 117
  cells, and all probe results (stability, geometry CIC/NF, whitening rescue).
  Nothing is hand-edited.
- `data/README.md` — dataset provenance, fetch instructions, and citation
  requirements (datasets are NOT redistributed).

## Reproducing the tables without a GPU

    python3 aggregate_latex.py --method tracincp   # results + baseline tables
    mkdir -p paper && python3 generate_figures.py  # RS table + figures

## Re-running the campaign

Requires the datasets (see `data/README.md`) and a CUDA GPU. Entry point:
`src/s1/run_experiment.py` per cell. Hyperparameters are in
`src/s1/config.py` (K=5 checkpoints, JL dim 1024, FPR target 0.01,
sigma=0.5, N=200).

## License

Code and results: MIT (see LICENSE). Datasets remain under their original
licenses and citation requirements (see `data/README.md`).
