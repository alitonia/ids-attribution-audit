"""Gradient-geometry diagnostic for the TRAK-on-tabular remark (2026-08-21).

Cell: cic label_flip r=0.05 seed0 (same cell as the stability probe). On the
last poisoned checkpoint, compute FULL (unprojected) per-sample gradients of
the two-class margin output and report interpretable geometry:

  - effective rank of the per-sample gradient covariance (and top-10 share)
  - covariance mass OUTSIDE the top-1024 eigendirections (the share an
    m=1024 JL projection must reshuffle per seed)
  - per-sample gradient norm tail (p99 / median)
  - mean cosine similarity within poison, within clean, poison-vs-clean
  - share of total variance carried by the poison-clean mean difference
    (the first-order attribution signal direction)

If the spectrum decays sharply and the signal lives in low-variance
directions, ridge whitening amplifies noise exactly where projections differ
per seed -- a mechanism candidate for the projection sensitivity measured in
probe_flag_stability.json. Descriptive by design: numbers are reported, the
paper only cites them if the pattern is clean.

Run from repo root on a GPU pod: python3 scripts/probe_gradient_geometry.py
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
from src.s1.data import (load_cic_unsw_nb15, load_nf_toniot, load_ciciot2023,
                         stratified_split, Standardizer)
from src.s1.poison import apply_poison
from src.s1.train import MLP, train, set_determinism
from src.s1.score import load_model_from_ckpt

DATASET = os.environ.get("GEOM_DATASET", "cic")
ATTACK = os.environ.get("GEOM_ATTACK", "label_flip")
RATIO = float(os.environ.get("GEOM_RATIO", "0.05"))
SEED = int(os.environ.get("GEOM_SEED", "0"))
K = 5
N_SUB = 8192          # per group (poison / clean) for the gradient matrix
N_COS = 4096          # per group for cosine Grams


def per_sample_grads(model, X, y, device, chunk=512):
    """(n, p) matrix of per-sample gradients of the margin (2y-1)*logit.

    Unflattens parameters in model.parameters() order: the architecture is
    Linear(76,128)+ReLU+Dropout, Linear(128,64)+ReLU+Dropout, Linear(64,1)
    (dropout is identity at eval time).
    """
    import torch.func as F
    shapes = [tuple(v.shape) for v in model.parameters()]
    sizes = [int(np.prod(s)) for s in shapes]

    def fn(flat, x, lbl):
        t = []
        off = 0
        for s, n in zip(shapes, sizes):
            t.append(flat[off:off + n].reshape(s)); off += n
        w0, b0, w1, b1, w2, b2 = t
        h = torch.relu(x @ w0.T + b0)
        h = torch.relu(h @ w1.T + b1)
        logit = (h @ w2.T + b2).squeeze(-1)
        return (2.0 * lbl - 1.0) * logit

    gfn = F.vmap(F.grad(fn, argnums=0), in_dims=(None, 0, 0))
    rows = []
    model.eval()
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    yt = torch.from_numpy(y.astype(np.int64)).to(device)
    flat0 = torch.cat([v.detach().flatten() for v in model.parameters()]).to(device)
    for b in range(0, len(Xt), chunk):
        g = gfn(flat0, Xt[b:b + chunk], yt[b:b + chunk])
        rows.append(g.detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def main() -> None:
    t0 = time.time()
    set_determinism(SEED)
    device = config.resolve_device()
    if DATASET == "cic":
        X, y, names = load_cic_unsw_nb15()
    elif DATASET == "nf":
        X, y, names = load_nf_toniot()
    else:
        X, y, names = load_ciciot2023()
    split = stratified_split(y, np.random.default_rng(SEED))
    sc = Standardizer().fit(X[split["train"]])
    Xs = sc.transform(X)
    Xtr, ytr = Xs[split["train"]], y[split["train"]]
    pr = apply_poison(Xtr, ytr, ATTACK, RATIO, SEED, names)

    tag = f"geom_probe_{DATASET}_{ATTACK}_r{RATIO:g}"
    d = config.CKPT_DIR / f"{tag}_seed{SEED}_K{K}"
    if not sorted(d.glob("ckpt_*.pt")):
        train(pr.X, pr.y, Xs[split["val"]], y[split["val"]], seed=SEED,
              k_checkpoints=K, tag=tag, device=device)
    ck = sorted(d.glob("ckpt_*.pt"))[-1]
    model = load_model_from_ckpt(ck, MLP).to(device)

    rng = np.random.default_rng(SEED)
    poi_idx = np.flatnonzero(pr.poison_mask)
    clean_idx = np.flatnonzero(~pr.poison_mask)
    poi_sel = rng.choice(poi_idx, size=min(N_SUB, len(poi_idx)), replace=False)
    cln_sel = rng.choice(clean_idx, size=min(N_SUB, len(clean_idx)), replace=False)

    Gp = per_sample_grads(model, pr.X[poi_sel], pr.y[poi_sel], device)
    Gc = per_sample_grads(model, Xtr[cln_sel], ytr[cln_sel], device)
    G = np.concatenate([Gp, Gc], axis=0)
    n, p = G.shape

    out = {"cell": f"{DATASET}_{ATTACK}_r{RATIO}_seed{SEED}", "n": n, "p": p}

    # spectrum of the gradient covariance
    Gt = torch.from_numpy(G).to(device)
    Gt -= Gt.mean(dim=0, keepdim=True)
    C = (Gt.T @ Gt) / n
    eig = torch.linalg.eigvalsh(C).flip(0).clamp_min(0.0)
    eig = eig.cpu().numpy()
    tot = eig.sum()
    eff_rank = float(tot ** 2 / (eig ** 2).sum())
    top10 = float(eig[:10].sum() / tot)
    mass_beyond_1024 = float(eig[1024:].sum() / tot)
    out["spectrum"] = {"effective_rank": round(eff_rank, 1),
                       "top10_share": round(top10, 4),
                       "mass_beyond_top1024": round(mass_beyond_1024, 4)}

    # norm tail
    norms = np.linalg.norm(G, axis=1)
    out["norm_tail_p99_over_median"] = round(
        float(np.percentile(norms, 99) / np.median(norms)), 2)

    # cosines on subsamples
    def cos_stats(A, B):
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        B = B / np.linalg.norm(B, axis=1, keepdims=True)
        return float((A @ B.T).mean())
    out["cosines"] = {
        "within_poison": round(cos_stats(Gp[:N_COS], Gp[:N_COS]), 4),
        "within_clean": round(cos_stats(Gc[:N_COS], Gc[:N_COS]), 4),
        "poison_vs_clean": round(cos_stats(Gp[:N_COS], Gc[:N_COS]), 4),
    }

    # signal direction: poison-clean mean difference vs total variance
    dmu = Gp.mean(axis=0) - Gc.mean(axis=0)
    out["signal_share_of_variance"] = round(
        float((dmu ** 2).sum() / tot), 6)
    out["minutes_total"] = round((time.time() - t0) / 60, 1)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = (f"probe_gradient_geometry_{DATASET}_{ATTACK}"
                f"_r{RATIO:g}_seed{SEED}.json")
    (config.RESULTS_DIR / out_name).write_text(json.dumps(out, indent=1))
    print("RESULT_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
