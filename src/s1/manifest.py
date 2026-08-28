"""Evidence manifest — chain-of-custody for the 2026-08-28 re-run generation.

Every cell writes its own fragment (race-free atomic replace); the aggregator
consolidates fragments into EVIDENCE_MANIFEST.json. verify_evidence.py
re-hashes the chain at any later time.

Fragment content: result-JSON sha256, per-checkpoint sha256, environment
(run id, git rev, torch/GPU, determinism flags), wall time, timestamp.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _env_block() -> dict:
    import torch
    gpu = ""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "run_id": config.RUN_ID,
        "git_rev": os.environ.get("S1_GIT_REV", ""),
        "torch": torch.__version__,
        "gpu": gpu,
        "deterministic": config.DETERMINISTIC,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "host": os.uname().nodename,
    }


def record_cell(cell_key: str, result_path: Path, ckpt_paths: list[Path],
                wall_s: float) -> None:
    man_dir = config.RESULTS_DIR / "manifest"
    man_dir.mkdir(parents=True, exist_ok=True)
    frag = {
        "cell": cell_key,
        "env": _env_block(),
        "when": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(wall_s, 1),
        "result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "checkpoints": [{"path": str(p), "sha256": _sha256(p)}
                        for p in sorted(ckpt_paths)],
    }
    tmp = man_dir / f".{cell_key.replace('/', '_')}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(frag, indent=2))
    tmp.replace(man_dir / f"{cell_key.replace('/', '_')}.json")


def consolidate() -> dict:
    man_dir = config.RESULTS_DIR / "manifest"
    cells = {}
    if man_dir.exists():
        for f in sorted(man_dir.glob("*.json")):
            cells[f.stem] = json.loads(f.read_text())

    def _consistency_key(env: dict) -> str:
        # host is metadata, not part of the generation identity (two-pod split)
        e = {k: v for k, v in env.items() if k != "host"}
        return json.dumps(e, sort_keys=True)

    envs = {_consistency_key(c["env"]) for c in cells.values()}
    manifest = {
        "run_id": config.RUN_ID,
        "consolidated_at": datetime.now(timezone.utc).isoformat(),
        "generation_consistent": len(envs) == 1,
        "n_cells": len(cells),
        "env": next(iter(envs)) if len(envs) == 1 else sorted(envs),
        "cells": cells,
    }
    out = config.RESULTS_DIR / "EVIDENCE_MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2))
    manifest["_path"] = str(out)
    return manifest
