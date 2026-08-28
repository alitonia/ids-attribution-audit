"""Cohen-style randomized-smoothing certificate on the quarantined classifier.

This is the COMPLEMENTARY, TEST-TIME EVASION guarantee — a different threat model from
the training-time audit (paper Section V). Inference-only; CPU-friendly via batching.

Guarantee: for standardized feature space, if the smoothed classifier's top class
probability p_A and runner-up p_B are estimated with Monte Carlo, the certified l2
radius is sigma * (Phi^-1(p_A) - Phi^-1(p_B)) (Cohen et al. 2019, arXiv:1902.07198).
"""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy import stats  # scipy is a light dependency; erfinv-based Phi^-1

from .score import load_model_from_ckpt


def _phi_inv(p: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(np.clip(p, 1e-9, 1 - 1e-9))


def certify(model_cls, ckpt_path, X: np.ndarray, sigma: float, n_samples: int = 200,
            alpha: float = 0.001, batch_size: int = 2048, seed: int = 0) -> dict:
    """Return radii and predictions for each row of X under Gaussian smoothing.

    Uses a Clopper-Pearson-style lower bound on p_A (normal approximation) and an
    upper bound on p_B, as in Cohen et al.
    """
    assert sigma > 0
    model = load_model_from_ckpt(ckpt_path, model_cls)
    model.eval()
    n = len(X)
    radii = np.zeros(n, dtype=np.float64)
    preds = np.zeros(n, dtype=np.int64)
    z_a = float(stats.norm.ppf(1 - alpha))

    device = next(model.parameters()).device
    # seeded per cell so certified radii are exactly reproducible (paper claim)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for i in range(0, n, batch_size):
        xb_gpu = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        b = len(xb_gpu)
        counts = np.zeros((b, 2), dtype=np.int64)
        # Vectorized Monte Carlo sampling for massive GPU speedup
        noise = torch.randn((n_samples, b, xb_gpu.shape[1]), device=device,
                            generator=gen) * sigma
        with torch.no_grad():
            logits = model(xb_gpu.unsqueeze(0) + noise)
        c = (logits > 0).cpu().numpy().astype(np.int64)
        if c.ndim == 3 and c.shape[-1] == 1:
            c = c[..., 0]                      # (n_samples, b)
        c_sum = c.sum(axis=0)
        counts[:, 1] += c_sum
        counts[:, 0] += (n_samples - c_sum)
        # two-class: p_A = max count share, p_B = runner-up share, with CP-style slack
        share = counts / n_samples
        top2 = np.sort(share, axis=1)
        p_a = np.clip(top2[:, 1] - z_a * np.sqrt(top2[:, 1] * (1 - top2[:, 1]) / n_samples),
                      0.0, 1.0)
        p_b = np.clip(top2[:, 0] + z_a * np.sqrt(top2[:, 0] * (1 - top2[:, 0]) / n_samples),
                      0.0, 1.0)
        majority = counts.argmax(axis=1)
        gap_ok = p_a > p_b
        preds[i:i + b] = majority
        with np.errstate(invalid="ignore"):
            r = sigma * (_phi_inv(p_a) - _phi_inv(p_b))
        radii[i:i + b] = np.where(gap_ok & (r > 0), r, 0.0)

    return {"radii": radii, "preds": preds, "sigma": sigma, "n_samples": n_samples}


def certified_accuracy_at_radius(cert: dict, y_true: np.ndarray, radius: float,
                                 class_of_interest: int = 1) -> float:
    """Fraction of class_of_interest rows correctly classified AND certified >= radius."""
    sel = y_true == class_of_interest
    if sel.sum() == 0:
        return float("nan")
    ok = (cert["preds"][sel] == class_of_interest) & (cert["radii"][sel] >= radius)
    return float(ok.mean())


def _cp_upper(k: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Clopper-Pearson upper bound; k==0 edge via (1-U)^n = alpha."""
    out = np.empty_like(k, dtype=np.float64)
    zero = k == 0
    out[zero] = 1.0 - alpha ** (1.0 / n)
    nz = ~zero
    out[nz] = stats.beta.ppf(1 - alpha, k[nz], n - k[nz] + 1)
    return out


def certify_dual(model_cls, ckpt_path, X: np.ndarray, sigma: float,
                 n_samples: int = 200, alpha: float = 0.001,
                 batch_size: int = 2048, seed: int = 0) -> dict:
    """Certify with BOTH the deployed Wald bounds and exact Clopper-Pearson.

    Same batch order and generator seeding as certify(), so the Wald path
    reproduces the deployed certificate on identical hardware; the exact path
    recomputes the same votes with exact binomial bounds.
    """
    assert sigma > 0
    model = load_model_from_ckpt(ckpt_path, model_cls)
    model.eval()
    n = len(X)
    z_a = float(stats.norm.ppf(1 - alpha))
    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    radii_w = np.zeros(n, dtype=np.float64)
    radii_e = np.zeros(n, dtype=np.float64)
    preds = np.zeros(n, dtype=np.int64)

    for i in range(0, n, batch_size):
        xb_gpu = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        b = len(xb_gpu)
        noise = torch.randn((n_samples, b, xb_gpu.shape[1]), device=device,
                            generator=gen) * sigma
        with torch.no_grad():
            logits = model(xb_gpu.unsqueeze(0) + noise)
        c = (logits > 0).cpu().numpy().astype(np.int64)
        if c.ndim == 3 and c.shape[-1] == 1:
            c = c[..., 0]
        c1 = c.sum(axis=0)
        counts = np.stack([n_samples - c1, c1], axis=1)
        share = counts / n_samples
        top2 = np.sort(share, axis=1)
        s_top, s_bot = top2[:, 1], top2[:, 0]
        k_top = np.round(s_top * n_samples).astype(np.int64)
        k_bot = np.round(s_bot * n_samples).astype(np.int64)
        majority = counts.argmax(axis=1)

        p_a_w = np.clip(s_top - z_a * np.sqrt(s_top * (1 - s_top) / n_samples), 0.0, 1.0)
        p_b_w = np.clip(s_bot + z_a * np.sqrt(s_bot * (1 - s_bot) / n_samples), 0.0, 1.0)
        p_a_e = stats.beta.ppf(alpha, k_top, n_samples - k_top + 1)
        p_b_e = _cp_upper(k_bot, n_samples, alpha)

        preds[i:i + b] = majority
        for p_a, p_b, out in ((p_a_w, p_b_w, radii_w), (p_a_e, p_b_e, radii_e)):
            with np.errstate(invalid="ignore"):
                r = sigma * (_phi_inv(p_a) - _phi_inv(p_b))
            out[i:i + b] = np.where((p_a > p_b) & (r > 0), r, 0.0)

    return {"radii_wald": radii_w, "radii_exact": radii_e, "preds": preds,
            "sigma": sigma, "n_samples": n_samples, "alpha": alpha}


def certified_accuracy_dual(cert: dict, y_true: np.ndarray, radius: float,
                            class_of_interest: int = 1) -> dict:
    sel = y_true == class_of_interest
    if sel.sum() == 0:
        return {"wald": float("nan"), "exact": float("nan")}
    ok_w = (cert["preds"][sel] == class_of_interest) & (cert["radii_wald"][sel] >= radius)
    ok_e = (cert["preds"][sel] == class_of_interest) & (cert["radii_exact"][sel] >= radius)
    return {"wald": float(ok_w.mean()), "exact": float(ok_e.mean())}
