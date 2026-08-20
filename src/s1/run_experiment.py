"""S1 experiment orchestrator: poison -> train -> score -> quarantine -> recover.

One invocation runs one (dataset, attack_type, ratio, seed) cell and writes a JSON
result. The clean baseline cell (attack_type=none) trains the clean model and
produces the threshold null distribution used for FPR-controlled flagging.

Usage:
    python -m src.s1.run_experiment --dataset cic --attack none   --seed 0
    python -m src.s1.run_experiment --dataset cic --attack label_flip --ratio 0.05 --seed 0
    python -m src.s1.run_experiment --dataset nf  --attack trigger --ratio 0.02 --seed 1 --smoke

--smoke subsets training rows (gate G2 protocol, Aug 22) for a fast sanity run.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from pathlib import Path

from . import config
from .data import (load_cic_unsw_nb15, load_nf_toniot, load_ciciot2023,
                   load_multiclass, stratified_split, Standardizer)
from .poison import apply_poison, classify_features
from .train import MLP, set_determinism, train
from .score import trak_scores, tracincp_scores
from .baselines import (random_scores, loss_outlier_scores, grad_norm_scores,
                        activation_clustering_scores, oracle_scores)
from .evaluate import (localization_auroc, precision_at_k, calibrate_threshold,
                       flag_flows, binary_metrics, per_class_recall, f1_recovery,
                       gate_status, save_result)
from .train import set_determinism as _sd  # noqa: F401  (re-export for clarity)


def _load(dataset: str):
    if dataset == "cic":
        return load_cic_unsw_nb15()
    if dataset == "nf":
        return load_nf_toniot()
    if dataset == "ciciot2023":
        return load_ciciot2023()
    raise ValueError(dataset)


def _ckpts(tag: str, seed: int, k: int) -> list[Path]:
    d = config.CKPT_DIR / f"{tag}_seed{seed}_K{k}"
    return sorted(d.glob("ckpt_*.pt"))


# split + standardization depend only on (dataset, seed); a worker processes
# many cells per seed, so memoize the standardized array per process
_PREP_CACHE: dict = {}


def _eval_model(tag: str, seed: int, k: int, X_test, y_test,
                y_multi_test=None, device: torch.device | None = None) -> dict:
    device = torch.device(device) if device is not None else config.resolve_device()
    ckpts = _ckpts(tag, seed, k)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints for {tag} seed{seed}")
    from .score import load_model_from_ckpt
    model = load_model_from_ckpt(ckpts[-1], MLP, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int64)
    m = binary_metrics(y_test, preds)
    if y_multi_test is not None:
        m["per_class_recall"] = per_class_recall(preds, y_multi_test)
    return m


def _get_clean_null(dataset: str, seed: int, k: int, smoke: bool,
                    methods: tuple, ck_clean: list[Path], Xtr: np.ndarray,
                    ytr: np.ndarray, targets: np.ndarray, ttargets: np.ndarray,
                    device: torch.device) -> dict[str, np.ndarray]:
    """Retrieve or compute the deterministic null attribution distribution on clean data."""
    cache_path = config.PROCESSED_DIR / f"null_scores_{dataset}_seed{seed}_K{k}{'_smoke' if smoke else ''}.npz"
    clean_null = {}
    cached_data = {}
    if cache_path.exists():
        try:
            with np.load(cache_path) as data:
                cached_data = {key: data[key] for key in data.files}
        except Exception:
            cached_data = {}

    updated = False
    for m in methods:
        if m in cached_data:
            clean_null[m] = cached_data[m]
        else:
            if m == "trak":
                clean_null["trak"] = trak_scores(MLP, ck_clean, Xtr, ytr,
                                                 targets, ttargets, seed=seed,
                                                 device=device)
                cached_data["trak"] = clean_null["trak"]
                updated = True
            elif m == "tracincp":
                clean_null["tracincp"] = tracincp_scores(MLP, ck_clean, Xtr, ytr,
                                                         targets, ttargets,
                                                         device=device)
                cached_data["tracincp"] = clean_null["tracincp"]
                updated = True

    if updated:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        import os, time
        tmp = cache_path.with_name(f"{cache_path.stem}_{os.getpid()}_{time.time_ns()}.tmp.npz")
        np.savez_compressed(tmp, **cached_data)
        tmp.replace(cache_path)

    return clean_null


def run(dataset: str, attack: str, ratio: float, seed: int, k: int, smoke: bool, methods: tuple, device: str | torch.device | None = None, loaded_data=None) -> dict:
    # resolve once; every train/eval/scoring call below runs on this device
    device = torch.device(device) if device is not None else config.resolve_device()
    set_determinism(seed)
    if loaded_data is not None:
        X, y, names, y_multi = loaded_data
    else:
        X, y, names = _load(dataset)
        y_multi = load_multiclass(dataset)
    prep_key = (dataset, seed)
    if prep_key in _PREP_CACHE:
        Xs, split = _PREP_CACHE[prep_key]
    else:
        rng = np.random.default_rng(seed)
        split = stratified_split(y, rng)
        sc = Standardizer().fit(X[split["train"]])
        Xs = sc.transform(X)
        _PREP_CACHE[prep_key] = (Xs, split)
    Xtr, ytr = Xs[split["train"]], y[split["train"]]
    Xte, yte = Xs[split["test"]], y[split["test"]]
    ym_te = y_multi[split["test"]] if y_multi is not None else None
    if smoke:
        sub = np.random.default_rng(seed).choice(len(Xtr), size=min(20_000, len(Xtr)),
                                                 replace=False)
        Xtr, ytr = Xtr[sub], ytr[sub]

    tag_clean = f"{dataset}_clean"
    result = {"dataset": dataset, "attack": attack, "ratio": ratio, "seed": seed,
              "k_checkpoints": k, "smoke": smoke, "device": str(device),
              "n_train": int(len(Xtr)), "n_poison": 0}

    # 1. clean baseline (trained once per (dataset, seed, k), cached on disk)
    if not _ckpts(tag_clean, seed, k):
        train(Xtr, ytr, Xs[split["val"]], y[split["val"]], seed=seed,
              k_checkpoints=k, tag=tag_clean, device=device)
    clean_metrics = _eval_model(tag_clean, seed, k, Xte, yte, ym_te, device=device)
    result["clean_f1"] = clean_metrics["f1"]

    if attack == "none":
        result["metrics_poisoned"] = clean_metrics
        # Precompute and cache clean_null for this seed during clean baseline run
        attack_test = np.flatnonzero(yte == 1)
        t_sel = attack_test[:2000] if len(attack_test) >= 200 else attack_test
        targets = Xte[t_sel]
        ttargets = yte[t_sel]
        _get_clean_null(dataset, seed, k, smoke, methods, _ckpts(tag_clean, seed, k),
                        Xtr, ytr, targets, ttargets, device)
        save_result(config.RESULTS_DIR /
                    f"exp_{dataset}_none_seed{seed}_K{k}{'_smoke' if smoke else ''}.json",
                    result)
        return result

    # 2. poison the training split
    pr = apply_poison(Xtr, ytr, attack, ratio, seed, names)
    result["n_poison"] = int(pr.poison_mask.sum())
    result["poison_meta"] = {kk: (vv if not isinstance(vv, np.ndarray) else int(vv))
                             for kk, vv in pr.meta.items()}
    tag_poison = f"{dataset}_{attack}_r{ratio:g}"
    if not _ckpts(tag_poison, seed, k):
        train(pr.X, pr.y, Xs[split["val"]], y[split["val"]], seed=seed,
              k_checkpoints=k, tag=tag_poison, device=device)
    poisoned_metrics = _eval_model(tag_poison, seed, k, Xte, yte, ym_te,
                                   device=device)
    result["metrics_poisoned"] = poisoned_metrics

    # 3. score training flows (poisoned-model checkpoints; clean-model null for tau)
    ck_poison = _ckpts(tag_poison, seed, k)
    ck_clean = _ckpts(tag_clean, seed, k)
    # Scoring targets: clean ATTACK-class test flows — the traffic the operator
    # cares about protecting (audits judged by effect on attack detection).
    # Target composition changes attribution direction (verified 2026-08-18:
    # benign-majority targets invert attribution scores under directed flips).
    attack_test = np.flatnonzero(yte == 1)
    t_sel = attack_test[:2000] if len(attack_test) >= 200 else attack_test
    targets = Xte[t_sel]
    ttargets = yte[t_sel]
    result["target_mode"] = f"attack_only({len(t_sel)})"
    scores: dict[str, np.ndarray] = {}
    result["errors"] = {}

    clean_null = _get_clean_null(dataset, seed, k, smoke, methods, ck_clean,
                                 Xtr, ytr, targets, ttargets, device)

    for m in methods:
        if m == "trak":
            try:
                scores["trak"] = trak_scores(MLP, ck_poison, pr.X, pr.y,
                                             targets, ttargets, seed=seed,
                                             device=device)
            except Exception as e:  # keep going: report the failure, don't kill the cell
                result["errors"]["trak"] = f"{type(e).__name__}: {e}"
        elif m == "tracincp":
            try:
                scores["tracincp"] = tracincp_scores(MLP, ck_poison, pr.X, pr.y,
                                                     targets, ttargets,
                                                     device=device)
            except Exception as e:
                result["errors"]["tracincp"] = f"{type(e).__name__}: {e}"
    scores["random"] = random_scores(len(pr.X), seed)
    scores["loss_outlier"] = loss_outlier_scores(MLP, ck_poison[-1], pr.X, pr.y,
                                                 device=device)
    scores["grad_norm"] = grad_norm_scores(MLP, ck_poison[-1], pr.X, pr.y,
                                           device=device)
    scores["activation_clustering"] = activation_clustering_scores(
        MLP, ck_poison[-1], pr.X, pr.y, device=device)
    scores["oracle"] = oracle_scores(pr.poison_mask, seed)

    # convert to suspicion convention (higher = more suspicious) before evaluation
    from .evaluate import SUSPICION_SIGN
    suspicion = {m: SUSPICION_SIGN.get(m, 1.0) * s for m, s in scores.items()}

    # 4. flag at FPR-controlled threshold, quarantine, retrain, recover
    result["localization"] = {}
    for m, s in suspicion.items():
        auroc = localization_auroc(s, pr.poison_mask)
        null = SUSPICION_SIGN.get(m, 1.0) * clean_null.get(m, s)
        tau = calibrate_threshold(null)
        flagged = flag_flows(s, tau)
        result["localization"][m] = {
            "auroc": auroc,
            "precision_at_2x": precision_at_k(s, pr.poison_mask,
                                              2 * max(1, int(pr.poison_mask.sum()))),
            "threshold": tau,
            "n_flagged": int(flagged.sum()),
            "flag_precision": float(pr.poison_mask[flagged].mean())
            if flagged.any() else 0.0,
        }

    trak_auroc = result["localization"].get("trak", {}).get("auroc", float("nan"))
    primary = ("trak" if trak_auroc == trak_auroc and trak_auroc >= config.GATE_G2_SMOKE_AUROC
               and "trak" in suspicion else
               "tracincp" if "tracincp" in suspicion else
               max(result["localization"], key=lambda k: result["localization"][k]["auroc"]))
    result["primary_method"] = primary
    if primary in suspicion:
        tau = result["localization"][primary]["threshold"]
        keep = ~flag_flows(suspicion[primary], tau)
        tag_quar = f"{tag_poison}_quar"
        if not _ckpts(tag_quar, seed, k):
            train(pr.X[keep], pr.y[keep], Xs[split["val"]], y[split["val"]],
                  seed=seed, k_checkpoints=k, tag=tag_quar, device=device)
        quar_metrics = _eval_model(tag_quar, seed, k, Xte, yte, ym_te,
                                   device=device)
        result["metrics_quarantined"] = quar_metrics
        result["f1_recovery"] = f1_recovery(poisoned_metrics["f1"],
                                            quar_metrics["f1"], clean_metrics["f1"])
        result["n_quarantined"] = int((~keep).sum())

    # 5. gates
    smoke_auroc = result["localization"].get(primary, {}).get("auroc", float("nan"))
    result["gates"] = gate_status(smoke_auroc if smoke else config.GATE_G2_SMOKE_AUROC,
                                  None if smoke else smoke_auroc)

    out = config.RESULTS_DIR / \
        f"exp_{dataset}_{attack}_r{ratio:g}_seed{seed}_K{k}{'_smoke' if smoke else ''}.json"
    save_result(out, result)
    print(f"wrote {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cic", "nf", "ciciot2023"], default="cic")
    ap.add_argument("--attack", choices=["none", *config.ATTACK_TYPES], required=True)
    ap.add_argument("--ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-checkpoints", type=int, default=5)
    ap.add_argument("--methods", default="trak,tracincp",
                    help="comma list from {trak,tracincp}")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None,
                    help="cpu|cuda; default resolves from S1_DEVICE env (auto)")
    args = ap.parse_args()
    run(args.dataset, args.attack, args.ratio, args.seed, args.k_checkpoints,
        args.smoke, tuple(args.methods.split(",")), device=args.device)


if __name__ == "__main__":
    main()
