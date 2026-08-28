"""v2 campaign driver (2026-08-28 re-run): 5 seeds, single generation.

Cells: 3 datasets x (1 clean + 3 attacks x 4 ratios) x 5 seeds = 195.
Env-scoped output dirs (S1_RESULTS_DIR / S1_CKPT_DIR / S1_PROCESSED_DIR)
isolate this generation; every cell self-registers in the evidence manifest.

Usage: python3 -u -m src.s1.run_campaign_v2 --dataset nf --workers 6
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import multiprocessing

from src.s1 import config


def worker(args):
    dataset, attack, ratio, seed, threads = args
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,garbage_collection_threshold:0.8")
    # re-resolve env-dependent dirs after fork (workers inherit exports)
    from src.s1 import config as cfg
    from src.s1.run_experiment import run as run_exp, _load
    from src.s1.data import load_multiclass
    global GLOBAL_DATA
    print(f"[cell] {dataset} {attack} r={ratio} seed={seed}", flush=True)
    try:
        run_exp(dataset, attack, ratio, seed, 5, False, ("trak", "tracincp"),
                None, loaded_data=GLOBAL_DATA)
        print(f"[done] {dataset} {attack} r={ratio} seed={seed}", flush=True)
        return True
    except Exception as e:
        print(f"[FAIL] {dataset} {attack} r={ratio} seed={seed}: {e}",
              flush=True)
        import traceback
        traceback.print_exc()
        return False
    finally:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


GLOBAL_DATA = None

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nf", "cic", "ciciot2023"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    from src.s1.run_experiment import _load
    from src.s1.data import load_multiclass
    print(f"[prep] loading {args.dataset} in parent (COW)", flush=True)
    X, y, names = _load(args.dataset)
    GLOBAL_DATA = (X, y, names, load_multiclass(args.dataset))

    tasks_clean, tasks_poison = [], []
    for a, s in product(["none", "label_flip", "feature_perturb", "trigger"], seeds):
        ratios = [0.0] if a == "none" else list(config.POISON_RATIOS)
        for r in ratios:
            jname = config.RESULTS_DIR / \
                f"exp_{args.dataset}_{a}_r{r:g}_seed{s}_K5.json"
            if jname.exists():
                print(f"[skip] {jname.name}", flush=True)
                continue
            t = (args.dataset, a, r, s, args.threads)
            (tasks_clean if a == "none" else tasks_poison).append(t)

    ctx = multiprocessing.get_context("fork")
    ok = []
    for label, tasks in (("clean", tasks_clean), ("poison", tasks_poison)):
        if not tasks:
            continue
        print(f"[phase] {label}: {len(tasks)} cells", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            ok.extend(list(ex.map(worker, tasks)))

    n, good = len(ok), sum(bool(r) for r in ok)
    print(f"[campaign] {good}/{n} cells OK", flush=True)
    sys.exit(0 if good == n and n > 0 else 1)
