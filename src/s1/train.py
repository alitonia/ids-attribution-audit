"""Train the IDS MLP classifier and save K checkpoints for attribution ensembling.

Safety: thread count capped (config.TORCH_THREADS) and AMP disabled (config.USE_AMP)
per the 2026-08-12 freeze incident and the no-AMP rule. Binary classification with
class-weighted BCE to handle the built-in imbalance of CIC-UNSW-NB15.

Usage:
    python -m src.s1.train --dataset cic --seed 0 --k-checkpoints 5
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data import (load_cic_unsw_nb15, load_nf_toniot, load_ciciot2023,
                   stratified_split, Standardizer)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=config.HIDDEN_DIMS,
                 dropout: float = config.DROPOUT) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logits
        return self.net(x).squeeze(-1)


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(config.TORCH_THREADS)
    if config.DETERMINISTIC:
        # best-effort kernel determinism (warn_only: BCE-path scatter ops
        # would hard-error otherwise); CUBLAS workspace pinned in config
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def train(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
          y_val: np.ndarray, seed: int, epochs: int = config.EPOCHS,
          k_checkpoints: int = 5, tag: str = "clean",
          device: str | torch.device | None = None) -> dict:
    """Train; save K evenly spaced checkpoints + final model. Returns log dict.

    device defaults to config.resolve_device() (S1_DEVICE env var; auto = cuda
    if available else cpu). Checkpoints are plain state dicts (device-independent).
    """
    assert not config.USE_AMP, "AMP requires explicit user approval — do not enable."
    device = torch.device(device) if device is not None else config.resolve_device()
    model = MLP(X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR,
                            weight_decay=config.WEIGHT_DECAY)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Pre-load entirely to device to eliminate per-batch PCIe transfer overhead.
    # The whole tabular dataset is <100MB, easily fitting in 4GB VRAM.
    # Dynamic Batch Sizing for massive datasets
    current_batch_size = config.BATCH_SIZE
    if len(X_train) > 1000000:
        current_batch_size = 8192

    Xt_dev = torch.from_numpy(X_train).to(device)
    yt_dev = torch.from_numpy(y_train.astype(np.float32)).to(device)
    Xvt = torch.from_numpy(X_val).to(device)
    yvt = torch.from_numpy(y_val.astype(np.float32)).to(device)

    ckpt_dir = config.CKPT_DIR / f"{tag}_seed{seed}_K{k_checkpoints}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_at = {int(round(e)) for e in np.linspace(1, epochs, k_checkpoints)}
    log = {"seed": seed, "tag": tag, "epochs": [], "ckpts": []}
    k_i = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # CUDA Graphs capture to completely eliminate PyTorch kernel launch overhead
    is_cuda = (device.type == "cuda" if isinstance(device, torch.device) else "cuda" in str(device))
    static_xb, static_yb, graph = None, None, None
    if is_cuda and len(Xt_dev) >= current_batch_size:
        static_xb = torch.zeros((current_batch_size, Xt_dev.shape[1]), device=device)
        static_yb = torch.zeros(current_batch_size, device=device)
        
        # We must make the optimizer capturable
        opt.param_groups[0]['capturable'] = True
        
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(static_xb), static_yb)
                loss.backward()
                opt.step()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            loss_static = loss_fn(model(static_xb), static_yb)
            loss_static.backward()
            opt.step()

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(Xt_dev), generator=gen, device=device)
        
        n_full_batches = len(Xt_dev) // current_batch_size
        
        if graph is not None:
            for i in range(n_full_batches):
                batch_idx = idx[i*current_batch_size : (i+1)*current_batch_size]
                static_xb.copy_(Xt_dev[batch_idx])
                static_yb.copy_(yt_dev[batch_idx])
                graph.replay()
        else:
            for i in range(n_full_batches):
                batch_idx = idx[i*current_batch_size : (i+1)*current_batch_size]
                xb = Xt_dev[batch_idx]
                yb = yt_dev[batch_idx]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                
        remainder = len(Xt_dev) % current_batch_size
        if remainder > 0:
            batch_idx = idx[-remainder:]
            xb = Xt_dev[batch_idx]
            yb = yt_dev[batch_idx]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(Xvt)
            val_loss = float(loss_fn(val_logits, yvt))
            preds = (torch.sigmoid(val_logits) > 0.5).float()
            acc = float((preds == yvt).float().mean().cpu())
        log["epochs"].append({"epoch": epoch, "val_loss": val_loss, "val_acc": acc})
        if epoch in save_at:
            p = ckpt_dir / f"ckpt_{k_i}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "in_dim": X_train.shape[1]}, p)
            log["ckpts"].append(str(p))
            k_i += 1
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cic", "nf", "ciciot2023"], default="cic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-checkpoints", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--device", default=None,
                    help="cpu|cuda; default resolves from S1_DEVICE env (auto)")
    args = ap.parse_args()

    set_determinism(args.seed)
    if args.dataset == "cic":
        X, y, names = load_cic_unsw_nb15()
    elif args.dataset == "nf":
        X, y, names = load_nf_toniot()
    else:
        X, y, names = load_ciciot2023()
    rng = np.random.default_rng(args.seed)
    split = stratified_split(y, rng)
    sc = Standardizer().fit(X[split["train"]])
    Xs = sc.transform(X)

    tag = f"{args.dataset}_clean"
    log = train(Xs[split["train"]], y[split["train"]],
                Xs[split["val"]], y[split["val"]],
                seed=args.seed, epochs=args.epochs,
                k_checkpoints=args.k_checkpoints, tag=tag,
                device=args.device)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"train_log_{tag}_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(log, f, indent=2)
    print(f"wrote {out}; ckpts={len(log['ckpts'])}")


if __name__ == "__main__":
    main()
