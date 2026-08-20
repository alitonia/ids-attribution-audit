"""Projection-stability probe for the deployed TRAK audit (2026-08-21 audit follow-up).

Backs the Remark in paper Sec. IV: the JL bound covers an idealized variant,
so we measure empirically how much the DEPLOYED audit's flag decisions depend
on the random projection. Cell: cic label_flip r=0.05 seed 0 (the campaign's
validated cell). Variants of the same pipeline:

    A: proj_dim=1024, projector_seed=0   (deployed campaign configuration)
    B: proj_dim=2048, projector_seed=0
    C: proj_dim=1024, projector_seed=7

Per variant: localization AUROC, flag precision at the 1%-FPR calibrated
threshold (calibrated on clean-model scores over clean train flows, exactly as
the campaign does), and the flag set. Cross-variant: pairwise flag agreement,
Jaccard on the planted poisons, and AUROC spread.

Writes results/probe_flag_stability.json and prints a RESULT_JSON block.
Run from the repo root on a GPU pod:  python3 scripts/probe_flag_stability.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.s1 import config
from src.s1.data import load_cic_unsw_nb15, stratified_split, Standardizer
from src.s1.poison import apply_poison
from src.s1.train import MLP, train, set_determinism
from src.s1.score import trak_scores
from src.s1.evaluate import (localization_auroc, calibrate_threshold,
                             flag_flows, SUSPICION_SIGN)

DATASET, ATTACK, RATIO, SEED, K = "cic", "label_flip", 0.05, 0, 5
VARIANTS = [("A_p1024_s0", 1024, 0), ("B_p2048_s0", 2048, 0),
            ("C_p1024_s7", 1024, 7)]


def main() -> None:
    t0 = time.time()
    set_determinism(SEED)
    device = config.resolve_device()
    X, y, names = load_cic_unsw_nb15()
    split = stratified_split(y, np.random.default_rng(SEED))
    sc = Standardizer().fit(X[split["train"]])
    Xs = sc.transform(X)
    Xtr, ytr = Xs[split["train"]], y[split["train"]]
    Xval, yval = Xs[split["val"]], y[split["val"]]
    Xte, yte = Xs[split["test"]], y[split["test"]]

    pr = apply_poison(Xtr, ytr, ATTACK, RATIO, SEED, names)
    mask = pr.poison_mask
    attack_test = np.flatnonzero(yte == 1)
    t_sel = attack_test[:2000]
    targets, ttargets = Xte[t_sel], yte[t_sel]

    tag_clean, tag_poison = "probe_clean", "probe_poisoned"
    dir_clean = config.CKPT_DIR / f"{tag_clean}_seed{SEED}_K{K}"
    dir_poison = config.CKPT_DIR / f"{tag_poison}_seed{SEED}_K{K}"
    if not sorted(dir_clean.glob("ckpt_*.pt")):
        train(Xtr, ytr, Xval, yval, seed=SEED, k_checkpoints=K,
              tag=tag_clean, device=device)
    if not sorted(dir_poison.glob("ckpt_*.pt")):
        train(pr.X, pr.y, Xval, yval, seed=SEED, k_checkpoints=K,
              tag=tag_poison, device=device)
    ck_clean = sorted(dir_clean.glob("ckpt_*.pt"))[:K]
    ck_poison = sorted(dir_poison.glob("ckpt_*.pt"))[:K]

    sign = SUSPICION_SIGN.get("trak", -1.0)
    out = {"cell": f"{DATASET}_{ATTACK}_r{RATIO}_seed{SEED}",
           "n_train": int(len(pr.X)), "n_poison": int(mask.sum()),
           "variants": {}}
    flags, aurocs = {}, {}
    for name, dim, pseed in VARIANTS:
        t1 = time.time()
        s_poi = trak_scores(MLP, ck_poison, pr.X, pr.y, targets, ttargets,
                            projector_dim=dim, seed=pseed, device=device)
        s_null = trak_scores(MLP, ck_clean, Xtr, ytr, targets, ttargets,
                             projector_dim=dim, seed=pseed, device=device)
        susp, null = sign * s_poi, sign * s_null
        tau = calibrate_threshold(null)
        fl = flag_flows(susp, tau)
        auroc = localization_auroc(susp, mask)
        prec = float(mask[fl].mean()) if fl.any() else 0.0
        flags[name], aurocs[name] = fl, auroc
        out["variants"][name] = {
            "proj_dim": dim, "projector_seed": pseed,
            "auroc": auroc, "flag_precision": prec, "n_flagged": int(fl.sum()),
            "poison_recall_at_flags": float(mask[fl].sum() / mask.sum()),
            "minutes": round((time.time() - t1) / 60, 1),
        }

    names_ = [v[0] for v in VARIANTS]
    pairs = {}
    for i in range(len(names_)):
        for j in range(i + 1, len(names_)):
            a, b = flags[names_[i]], flags[names_[j]]
            both = (a & b & mask).sum()
            either = ((a | b) & mask).sum()
            pairs[f"{names_[i]}_vs_{names_[j]}"] = {
                "flag_agreement": float((a == b).mean()),
                "poison_jaccard": float(both / either) if either else 1.0,
            }
    out["cross_variant"] = pairs
    out["auroc_spread"] = float(max(aurocs.values()) - min(aurocs.values()))
    out["minutes_total"] = round((time.time() - t0) / 60, 1)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "probe_flag_stability.json").write_text(
        json.dumps(out, indent=1))
    print("RESULT_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
