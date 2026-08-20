"""Attribution scoring for the audit: TRAK (primary) + TracInCP (reported peer).

Protocol: score every TRAINING flow against a fixed clean target set (a clean test
subset). Poisoned training flows harm the model on those targets, so they accumulate
negative/low attribution; the audit ranks training flows by score and flags the low tail.

TRAK note: tabular use goes through TRAK's documented custom ModelOutput path
(modality-agnostic). Day-2 (gate G2) must smoke-test this on a 20K subset before
scaling up; TracInCP below is fully self-contained and is the fallback/peer method.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from torch import nn

from . import config


def load_model_from_ckpt(ckpt_path: Path, model_cls,
                         device: str | torch.device | None = None) -> nn.Module:
    """Load a checkpoint onto `device` (default: config.resolve_device())."""
    device = torch.device(device) if device is not None else config.resolve_device()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model_cls(state["in_dim"])
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


# --- TRAK ----------------------------------------------------------------------

def trak_scores(model_cls, ckpt_paths: list[Path], X_train: np.ndarray,
                y_train: np.ndarray, X_targets: np.ndarray, y_targets: np.ndarray,
                projector_dim: int = 1024, seed: int = 0,
                store_dir: Path | None = None, lambda_reg: float = 1e-2,
                feat_batch: int = 8192, score_chunk: int = 65536,
                device: str | torch.device | None = None) -> np.ndarray:
    """Per-training-point TRAK attribution averaged over targets and checkpoints.

    Returns scores of shape (n_train,). One fresh TRAKer per checkpoint (own
    store dir, model_id=0), scores averaged over checkpoints. traker is used
    for what it is fastest at — vmapped per-sample gradients + JL projection
    (featurize/score) — while finalize_features/finalize_scores are
    re-implemented here in chunked form, mathematically identical to traker's
    single-model pipeline (verified against traker 0.3.2 source):

        xtx       = X^T X accumulated in fp32 chunks
        features  = X (xtx + lambda*I)^{-1}, centered by |.|^{-1} mean
        score_i   = out_to_loss_i * mean_m (features_i . target_grad_m)

    Why not call traker's finalize_*: finalize_scores materializes an
    (n_train, n_targets) score memmap (2.5 GB disk + 2.5 GB pinned host RAM per
    worker per ckpt at CIC scale) only to row-average it afterwards, and
    finalize_features moves the whole (n_train, proj_dim) grads matrix to GPU
    (3 GB on CICIoT2023). Chunking bounds GPU at ~1 GB and host RAM at chunk
    size. The grads memmap is deleted once features exist (saves n*p disk).

    Numerics — root cause of the 2026-08-20 campaign failure (all 81 scored
    cells crashed with LinAlgError: singular matrix): use_half_precision MUST
    stay False. In fp16 traker stores projected gradients in float16 and
    accumulates X^T X in float16; at campaign scale (n up to 750k) the
    accumulated Gram matrix is numerically rank-deficient and
    torch.linalg.inv fails. lambda_reg > 0 additionally guarantees
    X^T X + lambda*I is positive definite. Stores go to disk, not /dev/shm:
    fp32 stores are ~n*p per checkpoint and would exhaust shm under
    concurrent workers.
    """
    device = torch.device(device) if device is not None else config.resolve_device()
    try:
        from trak import TRAKer
        from trak.modelout_functions import AbstractModelOutput
    except ImportError as e:
        raise ImportError(
            "traker not installed. Install in the project venv: pip install traker"
        ) from e

    class FlowClassificationModelOutput(AbstractModelOutput):
        def __init__(self) -> None:
            super().__init__()

        @staticmethod
        def get_output(model, weights, buffers, flow, label):
            logit = torch.func.functional_call(
                model, (weights, buffers), flow.unsqueeze(0)).squeeze()
            margin = torch.where(label == 1, logit, -logit)
            return margin.sum()

        def get_out_to_loss_grad(self, model, weights, buffers, batch):
            flows, labels = batch
            logits = torch.func.functional_call(
                model, (weights, buffers), flows).squeeze(-1)
            probs = torch.sigmoid(logits)
            ps = torch.where(labels == 1, probs, 1.0 - probs)
            return (1.0 - ps).clone().detach().unsqueeze(-1)

    import uuid
    import shutil

    n_train = len(X_train)
    Xtt = torch.from_numpy(np.ascontiguousarray(X_train)).to(device)
    ytt = torch.from_numpy(y_train.astype(np.int64)).to(device)
    Xt2 = torch.from_numpy(np.ascontiguousarray(X_targets)).to(device)
    yt2 = torch.from_numpy(y_targets.astype(np.int64)).to(device)

    final_scores = np.zeros(n_train, dtype=np.float64)
    state_dicts = [torch.load(p, map_location=device, weights_only=False)["model"] for p in ckpt_paths]

    for i, sd in enumerate(state_dicts):
        base_dir = (Path(store_dir) if store_dir is not None
                    else config.PROCESSED_DIR / "trak_stores")
        cur_store_dir = base_dir / uuid.uuid4().hex[:10]
        cur_store_dir.mkdir(parents=True, exist_ok=True)
        (cur_store_dir / "experiments.json").write_text("{}")
        ckpt_dir = cur_store_dir / "0"

        try:
            model = model_cls(X_train.shape[1]).to(device)
            traker = TRAKer(model=model, task=FlowClassificationModelOutput(),
                            train_set_size=n_train, save_dir=str(cur_store_dir),
                            load_from_save_dir=False, device=str(device),
                            proj_dim=projector_dim, use_half_precision=False,
                            projector_seed=seed, lambda_reg=lambda_reg)

            exp_name = f"targets_seed{seed}_dim{projector_dim}"

            traker.load_checkpoint(checkpoint=sd, model_id=0)
            for b in range(0, len(Xtt), feat_batch):
                xb, yb = Xtt[b:b + feat_batch], ytt[b:b + feat_batch]
                traker.featurize(batch=(xb, yb), num_samples=len(xb))

            # chunked finalize_features: xtx -> ridge inverse -> features.
            # Features reuse the grads memmap in-place (same shape/dtype),
            # halving peak disk to n*p per worker.
            grads_mm = np.load(ckpt_dir / "grads.mmap", mmap_mode="r+")
            p = grads_mm.shape[1]
            xtx = torch.zeros(p, p, device=device)
            for b in range(0, n_train, score_chunk):
                g = torch.from_numpy(np.array(grads_mm[b:b + score_chunk],
                                                dtype=np.float32)).to(device)
                xtx += g.T @ g
            xtx_inv = torch.linalg.inv(xtx + lambda_reg * torch.eye(p, device=device))
            xtx_inv /= xtx_inv.abs().mean()   # traker's centering (AUROC-invariant)
            for b in range(0, n_train, score_chunk):
                g = torch.from_numpy(np.array(grads_mm[b:b + score_chunk],
                                                dtype=np.float32)).to(device)
                grads_mm[b:b + score_chunk] = (g @ xtx_inv).cpu().numpy()
            grads_mm.flush()
            del grads_mm
            (ckpt_dir / "grads.mmap").rename(ckpt_dir / "features.mmap")
            feat_mm = np.load(ckpt_dir / "features.mmap", mmap_mode="r")
            traker.saver.model_ids[0]["is_finalized"] = 1  # satisfy score() assert

            traker.start_scoring_checkpoint(exp_name=exp_name, checkpoint=sd,
                                            model_id=0, num_targets=len(Xt2))
            for b in range(0, len(Xt2), feat_batch):
                xb, yb = Xt2[b:b + feat_batch], yt2[b:b + feat_batch]
                traker.score(batch=(xb, yb), num_samples=len(xb))
            traker.projector.free_memory()

            # chunked finalize_scores: row-mean over targets, weighted by Q
            tg = torch.from_numpy(np.array(
                np.load(ckpt_dir / f"{exp_name}_grads.mmap", mmap_mode="r"),
                dtype=np.float32)).to(device)                      # (m, p)
            otl = np.load(ckpt_dir / "out_to_loss.mmap", mmap_mode="r")  # (n, 1)
            for b in range(0, n_train, score_chunk):
                f = torch.from_numpy(np.array(feat_mm[b:b + score_chunk],
                                                dtype=np.float32)).to(device)
                s = (f @ tg.T).mean(dim=1).cpu().numpy()
                final_scores[b:b + score_chunk] += s * np.asarray(
                    otl[b:b + score_chunk, 0], dtype=np.float64)
        finally:
            shutil.rmtree(cur_store_dir, ignore_errors=True)

    final_scores /= len(state_dicts)
    return final_scores


# --- TracInCP (self-contained peer method) --------------------------------------

def tracincp_scores(model_cls, ckpt_paths: list[Path], X_train: np.ndarray,
                    y_train: np.ndarray, X_targets: np.ndarray,
                    y_targets: np.ndarray, batch_size: int = 16384,
                    device: str | torch.device | None = None) -> np.ndarray:
    """TracInCP: mean over checkpoints of <summed target gradient, per-sample
    train gradient>. One backward pass accumulates target gradients; a vmapped
    JVP evaluates the directional derivative per training flow. Returns
    (n_train,) scores."""
    device = torch.device(device) if device is not None else config.resolve_device()
    Xt = torch.from_numpy(X_targets).to(device)
    yt = torch.from_numpy(y_targets).to(device)
    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)

    scores_gpu = torch.zeros(len(X_train), device=device, dtype=torch.float32)
    for ckpt in ckpt_paths:
        model = load_model_from_ckpt(ckpt, model_cls, device=device)
        model.eval()
        
        # 1. Fast target gradient accumulation using standard backward pass
        model.zero_grad()
        for i in range(0, len(Xt), batch_size):
            logits = model(Xt[i:i + batch_size]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yt[i:i + batch_size].float(), reduction="sum")
            loss.backward()
            
        tgrad_dict = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        params_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
        
        # 2. Fast directional derivative using vmap(jvp)
        from torch.func import functional_call, jvp, vmap
        
        def get_jvp(x, t):
            def f(p):
                logit = functional_call(model, p, (x.unsqueeze(0),)).squeeze(-1)
                return torch.nn.functional.binary_cross_entropy_with_logits(logit.squeeze(), t.float(), reduction="sum")
            _, jvp_out = jvp(f, (params_dict,), (tgrad_dict,))
            return jvp_out
            
        vmap_jvp = vmap(get_jvp)
        
        for i in range(0, len(Xtr), batch_size):
            scores_gpu[i:i + batch_size] += vmap_jvp(Xtr[i:i + batch_size], ytr[i:i + batch_size]).detach()
            
    return (scores_gpu / len(ckpt_paths)).cpu().numpy().astype(np.float64)

