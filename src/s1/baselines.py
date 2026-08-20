"""Baseline suspect-flow scores required by the red-team review.

- random: uniform scores (sanity floor)
- loss_outlier: per-sample BCE loss under the final checkpoint
- grad_norm: per-sample gradient norm under the final checkpoint
- activation_clustering: per-class 2-means on penultimate activations, minority
  cluster flagged (Chen et al. 2018 style)
- oracle: ground-truth poison mask (upper bound, not a real detector)
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from torch import nn

from . import config
from .score import load_model_from_ckpt


def random_scores(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(size=n)


def _per_sample_loss_and_grad(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                              batch_size: int = 2048,
                              device: str | torch.device | None = None,
                              return_loss: bool = True, return_grad: bool = True):
    """Model and (X, y) must already be on the same device; `device` (default:
    X.device) only selects where the CPU-returned accumulators live."""
    if len(X) > 1000000 and batch_size == 2048:
        batch_size = 16384
    device = torch.device(device) if device is not None else X.device
    losses = torch.zeros(len(X), device=device) if return_loss else None
    grad_norms = torch.zeros(len(X), device=device) if return_grad else None
    model.eval()  # dropout must be a no-op under vmap (randomness guard)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad} if return_grad else None

    def loss_single(params, x, t):
        logit = torch.func.functional_call(
            model, params, (x.unsqueeze(0),)).squeeze()
        return nn.functional.binary_cross_entropy_with_logits(
            logit, t.float(), reduction="sum")

    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        if return_loss:
            with torch.inference_mode():
                logits = model(xb)
                losses[i:i + batch_size] = nn.functional.binary_cross_entropy_with_logits(
                    logits, yb.float(), reduction="none")
        if return_grad:
            grads = torch.func.vmap(torch.func.grad(loss_single),
                                    in_dims=(None, 0, 0))(params, xb, yb)
            flat = torch.cat([grads[n].reshape(len(xb), -1) for n, p in model.named_parameters() if p.requires_grad], dim=1)
            grad_norms[i:i + batch_size] = flat.norm(dim=1).detach()
            
    losses_out = losses.cpu().numpy() if return_loss else None
    gnorms_out = grad_norms.cpu().numpy() if return_grad else None
    return losses_out, gnorms_out


def loss_outlier_scores(model_cls, final_ckpt: Path, X_train: np.ndarray,
                        y_train: np.ndarray,
                        device: str | torch.device | None = None) -> np.ndarray:
    device = torch.device(device) if device is not None else config.resolve_device()
    model = load_model_from_ckpt(final_ckpt, model_cls, device=device)
    losses, _ = _per_sample_loss_and_grad(
        model, torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device), device=device,
        return_loss=True, return_grad=False)
    return losses


def grad_norm_scores(model_cls, final_ckpt: Path, X_train: np.ndarray,
                     y_train: np.ndarray,
                     device: str | torch.device | None = None) -> np.ndarray:
    device = torch.device(device) if device is not None else config.resolve_device()
    model = load_model_from_ckpt(final_ckpt, model_cls, device=device)
    _, gnorms = _per_sample_loss_and_grad(
        model, torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device), device=device,
        return_loss=False, return_grad=True)
    return gnorms


def activation_clustering_scores(model_cls, final_ckpt: Path, X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 device: str | torch.device | None = None) -> np.ndarray:
    """Suspicion score in [0,1]: fraction of same-class points in the smaller of two
    k-means clusters on penultimate activations (higher = more isolated = more suspect).
    Uses a dependency-free 2-means to keep the baseline self-contained."""
    device = torch.device(device) if device is not None else config.resolve_device()
    model = load_model_from_ckpt(final_ckpt, model_cls, device=device)
    with torch.no_grad():
        feats = model.net[:-1](torch.from_numpy(X_train).to(device))
    scores = np.zeros(len(X_train), dtype=np.float64)
    for c in np.unique(y_train):
        idx = np.flatnonzero(y_train == c)
        if len(idx) < 2:
            continue
        F = feats[idx]
        rng = np.random.default_rng(0)
        c_idx = rng.choice(len(idx), size=2, replace=False)
        centers = F[c_idx].clone()
        F_sq = torch.sum(F ** 2, dim=1, keepdim=True)
        for _ in range(25):
            c_sq = torch.sum(centers ** 2, dim=1)
            d_sq = F_sq + c_sq - 2.0 * torch.mm(F, centers.t())
            assign = torch.argmin(d_sq, dim=1)
            mask0 = (assign == 0)
            mask1 = (assign == 1)
            c0 = F[mask0].mean(dim=0) if mask0.any() else centers[0]
            c1 = F[mask1].mean(dim=0) if mask1.any() else centers[1]
            new_centers = torch.stack([c0, c1])
            if torch.allclose(new_centers, centers, atol=1e-5):
                break
            centers = new_centers
        assign_cpu = assign.cpu().numpy()
        sizes = np.bincount(assign_cpu, minlength=2)
        minority = int(sizes.argmin())
        if sizes[minority] == 0 or sizes[minority] == len(F):
            continue
        # score = probability of being in the minority cluster
        scores[idx] = np.where(assign_cpu == minority,
                               1.0 - sizes[minority] / len(F),
                               sizes[minority] / len(F))
    return scores


def oracle_scores(poison_mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Upper bound: perfect suspicion score with tiny noise to break ties."""
    rng = np.random.default_rng(seed)
    return poison_mask.astype(np.float64) + rng.uniform(0, 1e-6, size=len(poison_mask))
