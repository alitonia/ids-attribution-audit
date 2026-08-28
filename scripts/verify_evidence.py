#!/usr/bin/env python3
"""Verify the evidence chain of a rerun generation.

Checks (all against S1_RESULTS_DIR):
  1. every cell fragment exists and its result-JSON sha256 still matches
  2. every checkpoint sha256 still matches
  3. all fragments share one environment block (generation consistency)
  4. 180 poisoned + 15 clean cells present, seeds {0..4} complete
Exit 0 = chain intact.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("S1_RESULTS_DIR", str(REPO / "results" / "rerun2026"))

from src.s1 import config  # noqa: E402


def remap(path: str) -> Path:
    """Map pod-absolute manifest paths onto the local archive layout."""
    p = path.replace("/workspace/ckpt_v2/",
                     str(REPO / "data/processed/checkpoints_v2") + "/")
    p = p.replace("/workspace/results_rerun/", str(REPO / "results/rerun2026") + "/")
    return Path(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    res = config.RESULTS_DIR
    man = json.loads((res / "EVIDENCE_MANIFEST.json").read_text())
    failures = []

    if not man["generation_consistent"]:
        failures.append("generation_consistent=false — mixed environments")
    expect = 195
    if man["n_cells"] != expect:
        failures.append(f"cell count {man['n_cells']} != {expect}")

    for key, frag in man["cells"].items():
        rp = remap(frag["result"]["path"])
        if not rp.exists():
            failures.append(f"{key}: result missing {rp}")
        elif sha256(rp) != frag["result"]["sha256"]:
            failures.append(f"{key}: result sha mismatch")
        for ck in frag["checkpoints"]:
            cp = remap(ck["path"])
            if not cp.exists():
                failures.append(f"{key}: ckpt missing {cp}")
                continue
            if sha256(cp) != ck["sha256"]:
                failures.append(f"{key}: ckpt sha mismatch {cp.name}")

    names = set(man["cells"])
    need = []
    for d in ("cic", "nf", "ciciot2023"):
        for s in range(5):
            need.append(f"{d}_none_seed{s}")
            for a in ("label_flip", "feature_perturb", "trigger"):
                for r in (0.01, 0.02, 0.05, 0.1):
                    need.append(f"{d}_{a}_r{r:g}_seed{s}")
    absent = [n for n in need if n not in names]
    if absent:
        failures.append(f"missing cells: {absent[:5]} (+{len(absent)-5} more)"
                        if len(absent) > 5 else f"missing cells: {absent}")

    print(f"[verify] cells={man['n_cells']} "
          f"consistent={man['generation_consistent']} "
          f"failures={len(failures)}")
    for f_ in failures[:20]:
        print("  FAIL:", f_)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
