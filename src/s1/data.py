"""Dataset loaders for S1.

Primary: CIC-UNSW-NB15 (2024 refresh) — Data.csv + Label.csv, 413,995 flows, 79 features.
Hot spare: NF-ToN-IoT v1 — 43 NetFlow features, parquet from Kaggle (dhogla/nftoniot).

Conventions: binary labels (0 = benign, 1 = attack). Standardization is fit on the
training split only. Processed splits are cached under data/processed/*.npz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import os
import time

from . import config

def load_ciciot2023(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names) for CICIoT2023 (5% slice)."""
    cache_path = config.PROCESSED_DIR / "ciciot2023_cache.npz"
    if cache_path.exists():
        try:
            d = np.load(cache_path, allow_pickle=True)
            return d["X"], d["y"], d["feats"].tolist()
        except Exception:
            pass

    path = Path(path) if path else config.DATA_DIR / "ciciot2023" / "ciciot2023_5percent.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run src/s1/prepare_ciciot.py to fetch.")
    df = pd.read_parquet(path)
    # The label column in CICIoT2023 is usually 'label'
    y = df["label"].to_numpy(dtype=np.int64)
    feats = [c for c in df.columns if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f"{cache_path.stem}_{os.getpid()}_{time.time_ns()}.tmp.npz")
    np.savez_compressed(tmp, X=X, y=y, feats=np.array(feats))
    tmp.replace(cache_path)
    return X, y, feats


def load_cic_unsw_nb15(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names) for CIC-UNSW-NB15 (2024 refresh).

    Dataset.csv holds 76 numeric features; Label.csv holds integer class labels
    0-9 with 0 = benign. y is binary (0 benign / 1 attack).
    """
    cache_path = config.PROCESSED_DIR / "cic_cache.npz"
    if cache_path.exists():
        try:
            d = np.load(cache_path, allow_pickle=True)
            return d["X"], d["y"], d["feats"].tolist()
        except Exception:
            pass

    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    data_path = data_dir / config.CIC_UNSW_FILES[0]
    label_path = data_dir / config.CIC_UNSW_FILES[1]
    if not data_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"CIC-UNSW-NB15 files missing in {data_dir}. "
            "Submit the form at http://cicresearch.ca//CICDataset/CIC-UNSW/ "
            "(provenance: data/README.md)."
        )
    X_df = pd.read_csv(data_path)
    labels = pd.read_csv(label_path).values.ravel()
    if len(labels) != len(X_df):
        raise ValueError(f"row mismatch: Dataset.csv {len(X_df)} vs Label.csv {len(labels)}")
    X_df = X_df.select_dtypes(include=[np.number])
    X = X_df.to_numpy(dtype=np.float32)
    # integer-encoded labels: 0 = benign, 1..9 = attack classes
    y = (labels.astype(np.int64) != 0).astype(np.int64)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats = list(X_df.columns)
    tmp = cache_path.with_name(f"{cache_path.stem}_{os.getpid()}_{time.time_ns()}.tmp.npz")
    np.savez_compressed(tmp, X=X, y=y, feats=np.array(feats))
    tmp.replace(cache_path)
    return X, y, feats


def load_cic_unsw_multiclass(data_dir: Path | None = None) -> np.ndarray:
    """Integer class labels 0-9 (0 = benign) aligned row-wise with Dataset.csv.

    Used for per-attack-class recall evaluation: binary F1 can MASK poisoning
    damage (class-weighted training self-compensates), so per-class recall is
    the honest damage metric.
    """
    cache_path = config.PROCESSED_DIR / "cic_multi_cache.npy"
    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception:
            pass

    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    label_path = data_dir / config.CIC_UNSW_FILES[1]
    y = pd.read_csv(label_path).values.ravel().astype(np.int64)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f"{cache_path.stem}_{os.getpid()}_{time.time_ns()}.tmp.npy")
    np.save(tmp, y)
    tmp.replace(cache_path)
    return y


# NF-ToN-IoT v1 (parquet): columns are L4_SRC_PORT, L4_DST_PORT, PROTOCOL, L7_PROTO,
# IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS, TCP_FLAGS, FLOW_DURATION_MILLISECONDS,
# Label (int8 0/1), Attack (str category). Ports are identifier-like and dropped.
_NF_DROP = {"L4_SRC_PORT", "L4_DST_PORT", "Label", "Attack"}


def load_nf_toniot(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names) for NF-ToN-IoT v1 (parquet)."""
    cache_path = config.PROCESSED_DIR / "nf_cache.npz"
    if cache_path.exists():
        try:
            d = np.load(cache_path, allow_pickle=True)
            return d["X"], d["y"], d["feats"].tolist()
        except Exception:
            pass

    path = Path(path) if path else config.NF_TONIoT_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Fetch with: kaggle datasets download -d dhoogla/nftoniot -p data/"
        )
    df = pd.read_parquet(path)
    if "Label" not in df.columns:
        raise ValueError("expected integer 'Label' column in NF-ToN-IoT parquet")
    y = df["Label"].to_numpy(dtype=np.int64)
    feats = [c for c in df.columns
             if c not in _NF_DROP and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f"{cache_path.stem}_{os.getpid()}_{time.time_ns()}.tmp.npz")
    np.savez_compressed(tmp, X=X, y=y, feats=np.array(feats))
    tmp.replace(cache_path)
    return X, y, feats


def load_nf_toniot_multiclass(path: Path | None = None) -> np.ndarray:
    """Integer class labels for NF-ToN-IoT, factorized from the 'Attack' string column.
    
    0 = Benign, 1..N = Attack classes.
    """
    cache_path = config.PROCESSED_DIR / "nf_multi_cache.npy"
    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception:
            pass

    path = Path(path) if path else config.NF_TONIoT_PARQUET
    df = pd.read_parquet(path)
    labels = df["Attack"].astype(str)
    is_attack = df["Label"].to_numpy(dtype=bool)
    codes = pd.factorize(labels[is_attack])[0] + 1
    out = np.zeros(len(df), dtype=np.int64)
    out[is_attack] = codes
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f"{cache_path.stem}_{os.getpid()}_{time.time_ns()}.tmp.npy")
    np.save(tmp, out)
    tmp.replace(cache_path)
    return out


def load_ciciot2023_multiclass(path: Path | None = None) -> np.ndarray:
    """Integer class labels for CICIoT2023 from the 'attack_class' column.

    The 5% slice carries three label columns: 'label' (binary 0/1), 'Label'
    (fine-grained attack string), and 'attack_class' (8 families: Benign,
    Mirai, DDoS, DoS, Recon, Spoofing, Web-based, BruteForce). The family
    column is factorized here: 0 = Benign, 1..7 = attack families.
    """
    cache_path = config.PROCESSED_DIR / "ciciot2023_multi_cache.npy"
    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception:
            pass

    path = Path(path) if path else config.DATA_DIR / "ciciot2023" / "ciciot2023_5percent.parquet"
    df = pd.read_parquet(path, columns=["label", "attack_class"])
    labels = df["attack_class"].astype(str)
    is_attack = df["label"].to_numpy(dtype=bool)
    codes = pd.factorize(labels[is_attack])[0] + 1
    out = np.zeros(len(df), dtype=np.int64)
    out[is_attack] = codes
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f"{cache_path.stem}_{os.getpid()}_{time.time_ns()}.tmp.npy")
    np.save(tmp, out)
    tmp.replace(cache_path)
    return out


def load_multiclass(dataset: str) -> np.ndarray | None:
    """Return multiclass labels for the given dataset."""
    if dataset == "cic":
        return load_cic_unsw_multiclass()
    if dataset == "nf":
        return load_nf_toniot_multiclass()
    if dataset == "ciciot2023":
        return load_ciciot2023_multiclass()
    return None


def stratified_split(y: np.ndarray, rng: np.random.Generator,
                     val_frac: float = config.VAL_FRAC,
                     test_frac: float = config.TEST_FRAC) -> dict[str, np.ndarray]:
    """Return index split {'train','val','test'} stratified on y."""
    idx = {"train": [], "val": [], "test": []}
    for c in np.unique(y):
        ic = np.flatnonzero(y == c)
        rng.shuffle(ic)
        n_val = int(round(len(ic) * val_frac))
        n_test = int(round(len(ic) * test_frac))
        idx["val"].append(ic[:n_val])
        idx["test"].append(ic[n_val:n_val + n_test])
        idx["train"].append(ic[n_val + n_test:])
    return {k: np.sort(np.concatenate(v)) for k, v in idx.items()}


class Standardizer:
    """Fit-on-train-only feature standardization (median/IQR to resist outliers)."""

    def __init__(self) -> None:
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.center = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        self.scale = np.where(iqr < 1e-8, 1.0, iqr)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.center is not None, "fit first"
        return ((X - self.center) / self.scale).astype(np.float32)
