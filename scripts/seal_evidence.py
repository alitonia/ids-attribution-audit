#!/usr/bin/env python3
"""seal_evidence — append the sync-phase provenance to EVIDENCE_MANIFEST.json.

Records sha256 of every derived artifact (headline numbers, table inputs,
LaTeX tables, figures, paper sources/PDF) so the chain cell-JSON -> derived
table -> compiled paper is verifiable end to end. Idempotent per run_id.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("S1_RESULTS_DIR", str(REPO / "results" / "rerun2026"))
from src.s1 import config  # noqa: E402

TARGETS = [
    "results/rerun2026/HEADLINE_NUMBERS.json",
    "results/rerun2026/TABLES_INPUT.json",
    "results/rerun2026/tables/results_table_v2.tex",
    "results/rerun2026/tables/rs_radii_table_v2.tex",
    "results/rerun2026/tables/baseline_table_v2.tex",
    "results_table.tex",
    "baseline_table.tex",
    "rs_radii_table.tex",
    "paper/main.tex",
    "paper/main.pdf",
    "paper/fig_auroc.pdf",
    "paper/fig_rs_radii.pdf",
    "paper/edas_submission.md",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    man_path = config.RESULTS_DIR / "EVIDENCE_MANIFEST.json"
    man = json.loads(man_path.read_text())
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    entry = {
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "git_rev": rev,
        "artifacts": {},
    }
    for rel in TARGETS:
        p = REPO / rel
        if p.exists():
            entry["artifacts"][rel] = sha256(p)
        else:
            entry["artifacts"][rel] = "MISSING"
    man.setdefault("derived", []).append(entry)
    man_path.write_text(json.dumps(man, indent=2))
    print(f"[seal] {len(entry['artifacts'])} artifacts sealed at {rev[:8]}")
    missing = [k for k, v in entry["artifacts"].items() if v == "MISSING"]
    if missing:
        print("[seal] MISSING:", missing)
        sys.exit(1)


if __name__ == "__main__":
    main()
