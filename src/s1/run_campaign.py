import os
import sys
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from src.s1.run_experiment import run as run_exp, _load
from src.s1.data import load_multiclass
from src.s1 import config

# Global variable to hold data for copy-on-write sharing across forks
GLOBAL_DATA = None

def worker(args):
    dataset, attack, ratio, seed, threads = args
    
    threads_str = str(threads)
    os.environ["OMP_NUM_THREADS"] = threads_str
    os.environ["MKL_NUM_THREADS"] = threads_str
    os.environ["OPENBLAS_NUM_THREADS"] = threads_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads_str
    os.environ["NUMEXPR_NUM_THREADS"] = threads_str
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
    
    print(f"Starting {dataset} - {attack} (ratio={ratio}) seed={seed}...")
    try:
        run_exp(dataset, attack, ratio, seed, 5, False, ("trak", "tracincp"), None, loaded_data=GLOBAL_DATA)
        print(f"✅ FINISHED {dataset} - {attack} (ratio={ratio}) seed={seed}")
        return True
    except Exception as e:
        print(f"❌ ERROR in {dataset} - {attack} (ratio={ratio}) seed={seed}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["nf", "cic", "ciciot2023"])
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent PyTorch runs (VRAM dependent)")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads per PyTorch run")
    parser.add_argument("--force", action="store_true",
                        help="Re-run cells even if their result JSON exists "
                             "(used for the 2026-08-20 re-run after the TRAK fix)")
    args = parser.parse_args()
    
    # Pre-load data in parent process for zero-overhead copy-on-write sharing
    print(f"Pre-loading {args.dataset} dataset in parent process...")
    X, y, names = _load(args.dataset)
    y_multi = load_multiclass(args.dataset)
    GLOBAL_DATA = (X, y, names, y_multi)
    
    datasets = [args.dataset]
    attacks = ["none", "label_flip", "feature_perturb", "trigger"]
    ratios = list(config.POISON_RATIOS)   # (0.01, 0.02, 0.05, 0.10)
    seeds = [0, 1, 2]
    
    tasks_clean = []
    tasks_poison = []
    for d, a, s in product(datasets, attacks, seeds):
        ratios_to_run = [0.0] if a == "none" else ratios
        for r in ratios_to_run:
            k_folds = 5
            json_name = f"results/exp_{d}_{a}_r{r}_seed{s}_K{k_folds}.json"
            if os.path.exists(json_name) and not args.force:
                print(f"Skipping {json_name}, already done.")
                continue
            
            task = (d, a, r, s, args.threads)
            if a == "none":
                tasks_clean.append(task)
            else:
                tasks_poison.append(task)
                
    tasks_clean = list(dict.fromkeys(tasks_clean))
    tasks_poison = list(dict.fromkeys(tasks_poison))
    
    ctx = multiprocessing.get_context('fork')
    
    results = []
    if tasks_clean:
        print(f"Phase 1: Computing {len(tasks_clean)} clean baselines (preventing race conditions)...")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            results.extend(list(executor.map(worker, tasks_clean)))
            
    if tasks_poison:
        print(f"Phase 2: Computing {len(tasks_poison)} poisoned runs...")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            results.extend(list(executor.map(worker, tasks_poison)))
        
    total_tasks = len(tasks_clean) + len(tasks_poison)
    if all(results):
        print(f"🎉 All {total_tasks} runs completed successfully!")
    else:
        print(f"⚠️ Some runs failed. Check the error logs above.")
        sys.exit(1)
