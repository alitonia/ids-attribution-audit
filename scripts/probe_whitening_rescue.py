"""Whitening-rescue probe: causal test of the gradient-spectrum mechanism (2026-08-21).

On ONE fixed CIC instance (label_flip r=0.05 seed0), compute FULL (unprojected)
per-sample gradients for a poison+clean subsample and for the attack-class test
targets, then score training flows under three whitening regimes in the gradient
covariance eigenbasis:

  std      w_i = 1/(eig_i + lam)            (current TRAK ridge whitening)
  identity w_i = 1                          (keeps top directions)
  keep-k   w_i = 1 for top-k, else 1/(eig_i+lam)   (spectrum-aware fix)

score = mean over targets of (G_e * w) . T_e.  If identity/keep-k AUROC >> std
AUROC, the whitening step is causally responsible for destroying the CIC signal
(diagnosis + intervention). Projection noise is removed entirely (full grads),
so this isolates the whitening effect from the instance/projection effects.

Run on a GPU pod: python3 scripts/probe_whitening_rescue.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.s1 import config
from src.s1.data import load_cic_unsw_nb15, stratified_split, Standardizer
from src.s1.poison import apply_poison
from src.s1.train import MLP, train, set_determinism
from src.s1.score import load_model_from_ckpt
from src.s1.evaluate import localization_auroc
from scripts.probe_gradient_geometry import per_sample_grads

DATASET, ATTACK, RATIO, SEED, K = "cic", "label_flip", 0.05, 0, 5
LAM = 1e-2
N_PER = 4096          # per group (poison / clean)
N_TGT = 1000


def main() -> None:
    t0 = time.time()
    set_determinism(SEED)
    device = config.resolve_device()
    X, y, names = load_cic_unsw_nb15()
    split = stratified_split(y, np.random.default_rng(SEED))
    sc = Standardizer().fit(X[split["train"]])
    Xs = sc.transform(X)
    Xtr, ytr = Xs[split["train"]], y[split["train"]]
    Xte, yte = Xs[split["test"]], y[split["test"]]
    pr = apply_poison(Xtr, ytr, ATTACK, RATIO, SEED, names)

    tag = f"rescue_{DATASET}_{ATTACK}_r{RATIO:g}"
    d = config.CKPT_DIR / f"{tag}_seed{SEED}_K{K}"
    if not sorted(d.glob("ckpt_*.pt")):
        train(pr.X, pr.y, Xs[split["val"]], y[split["val"]], seed=SEED,
              k_checkpoints=K, tag=tag, device=device)
    ck = sorted(d.glob("ckpt_*.pt"))[-1]
    model = load_model_from_ckpt(ck, MLP).to(device)

    rng = np.random.default_rng(SEED)
    poi = np.flatnonzero(pr.poison_mask)
    cln = np.flatnonzero(~pr.poison_mask)
    poi_sel = rng.choice(poi, size=min(N_PER, len(poi)), replace=False)
    cln_sel = rng.choice(cln, size=min(N_PER, len(cln)), replace=False)
    sel = np.concatenate([poi_sel, cln_sel])
    poi_set = set(poi.tolist())
    mask = np.array([i in poi_set for i in sel])

    G = per_sample_grads(model, pr.X[sel], pr.y[sel], device)
    atk = np.flatnonzero(yte == 1)[:N_TGT]
    T = per_sample_grads(model, Xte[atk], yte[atk], device)

    Gt = torch.from_numpy(G).to(device)
    Gc = Gt - Gt.mean(dim=0, keepdim=True)
    C = (Gc.T @ Gc) / G.shape[0]
    eig, V = torch.linalg.eigh(C)
    idx = torch.argsort(eig, descending=True)
    eig, V = eig[idx].clamp_min(0.0), V[:, idx]

    G_e = (torch.from_numpy(G).to(device) @ V).cpu().numpy()
    T_e = (torch.from_numpy(T).to(device) @ V).cpu().numpy()
    eig = eig.cpu().numpy()

    def auroc_with(w):
        s = (G_e * w) @ T_e.T
        score = s.mean(axis=1)
        # registered convention: suspicion = -helpfulness score
        return float(localization_auroc(-score, mask))

    w_std = 1.0 / (eig + LAM)
    w_id = np.ones_like(eig)
    out = {"cell": f"{DATASET}_{ATTACK}_r{RATIO}_seed{SEED}",
           "effective_rank": float(round((eig.sum() ** 2) / (eig ** 2).sum(), 1)),
           "auroc_std_whitening": round(auroc_with(w_std), 3),
           "auroc_identity": round(auroc_with(w_id), 3)}
    for k in (10, 64, 256):
        w = w_std.copy()
        w[:k] = 1.0
        out[f"auroc_keep_top{k}"] = round(auroc_with(w), 3)
    out["minutes_total"] = round((time.time() - t0) / 60, 1)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "probe_whitening_rescue.json").write_text(
        json.dumps(out, indent=1))
    print("RESULT_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
