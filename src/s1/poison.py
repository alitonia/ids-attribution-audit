"""Poisoning protocol for S1 — 3 attack types x 4 ratios, fully seeded.

Threat model alignment (paper Section II): the adversary writes to a fraction rho of
training records at ingestion/labeling time. Per-feature budgets separate
attacker-controllable features from infrastructure features.

Feature kind heuristic is provisional: refine against CIC-UNSW-NB15's Readme.txt once
the dataset arrives (the audit must not rely on mis-classified feature provenance).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from . import config

# Feature provenance for CIC-UNSW-NB15 (CICFlowMeter features). Infrastructure
# features are timing-derived (IAT, duration, active/idle statistics): the adversary
# cannot freely set them. Everything else (packet counts, lengths, flags, header
# sizes, ratios) is payload/volume-derived and attacker-controllable.
# "flow_duration" added to catch FLOW_DURATION_MILLISECONDS in NF-ToN-IoT, which
# is router-stamped timing and cannot be freely set by the attacker.
_INFRA_PATTERNS = ("iat", "duration", "active", "idle", "flow_duration")


def classify_features(feature_names: list[str]) -> np.ndarray:
    """Return bool array: True = attacker-controllable."""
    kinds = np.ones(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        n = name.lower()
        if any(p in n for p in _INFRA_PATTERNS):
            kinds[i] = False
    return kinds


@dataclass
class PoisonResult:
    X: np.ndarray
    y: np.ndarray
    poison_mask: np.ndarray          # True at poisoned training rows
    meta: dict = field(default_factory=dict)


def _pick_rows(n: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    k = max(1, int(round(n * ratio)))
    return np.sort(rng.choice(n, size=k, replace=False))


def poison_label_flip(X: np.ndarray, y: np.ndarray, ratio: float,
                      rng: np.random.Generator) -> PoisonResult:
    """Directed label corruption: flip attack rows to benign.

    rho is a fraction of the FULL training set (paper convention: "rho of the
    training records"), drawn from attack-class rows. This is the evasion-aligned
    variant (threat model goal ii): the adversary hides attacks. Random
    bidirectional flipping was tested on 2026-08-18 and found nearly harmless
    to binary F1 (0.901 -> 0.931 at 5%), so it cannot drive a recovery story.
    """
    Xp, yp = X.copy(), y.copy()
    attack_rows = np.flatnonzero(y == 1)
    n_poison = min(max(1, int(round(len(X) * ratio))), len(attack_rows))
    rows = attack_rows[rng.choice(len(attack_rows), size=n_poison, replace=False)]
    yp[rows] = config.TRIGGER_TARGET_LABEL
    return PoisonResult(Xp, yp, _mask(len(Xp), rows),
                        {"attack_type": "label_flip", "ratio": ratio,
                         "n_poison_target": int(round(len(X) * ratio))})


def poison_feature_perturb(X: np.ndarray, y: np.ndarray, ratio: float,
                           rng: np.random.Generator,
                           attacker_feats: np.ndarray,
                           budget_std: float = config.PERTURB_BUDGET_STD) -> PoisonResult:
    """Bounded feature corruption: uniform noise in [-b, b] (standardized units)
    applied only to attacker-controllable features of targeted attack-class rows,
    labels flipped to benign. rho = fraction of the full training set."""
    Xp, yp = X.copy(), y.copy()
    attack_rows = np.flatnonzero(y == 1)
    n_poison = min(max(1, int(round(len(X) * ratio))), len(attack_rows))
    rows = attack_rows[rng.choice(len(attack_rows), size=n_poison, replace=False)]
    cols = np.flatnonzero(attacker_feats)
    noise = rng.uniform(-budget_std, budget_std, size=(len(rows), len(cols)))
    Xp[np.ix_(rows, cols)] += noise.astype(np.float32)
    yp[rows] = config.TRIGGER_TARGET_LABEL
    return PoisonResult(Xp, yp, _mask(len(Xp), rows),
                        {"attack_type": "feature_perturb", "ratio": ratio,
                         "budget_std": budget_std, "n_attacker_feats": int(len(cols))})


def poison_trigger(X: np.ndarray, y: np.ndarray, ratio: float,
                   rng: np.random.Generator,
                   attacker_feats: np.ndarray) -> PoisonResult:
    """Backdoor-style: fixed high-percentile trigger pattern on attacker-controllable
    features; source rows drawn from benign class, labels kept benign (the trigger
    later appears at test time — training-side effect is a correlated shortcut).
    rho = fraction of the full training set."""
    Xp, yp = X.copy(), y.copy()
    benign_rows = np.flatnonzero(y == 0)
    n_poison = min(max(1, int(round(len(X) * ratio))), len(benign_rows))
    rows = benign_rows[rng.choice(len(benign_rows), size=n_poison, replace=False)]
    cols = np.flatnonzero(attacker_feats)
    # Percentile computed over benign rows only: the trigger must be anomalous
    # within the benign class (high-end of benign feature space), not within
    # the full training set which is dominated by attack traffic in NF-ToN-IoT
    # and CIC-UNSW-NB15. Using the full dataset pollutes the percentile with
    # attack-class values, making the trigger trivially detectable.
    trigger = np.percentile(X[benign_rows][:, cols], 95, axis=0)
    Xp[np.ix_(rows, cols)] = trigger.astype(np.float32)
    return PoisonResult(Xp, yp, _mask(len(Xp), rows),
                        {"attack_type": "trigger", "ratio": ratio,
                         "n_attacker_feats": int(len(cols))})


def apply_poison(X: np.ndarray, y: np.ndarray, attack_type: str, ratio: float,
                 seed: int, feature_names: list[str]) -> PoisonResult:
    """Dispatch by attack type; single entry point for experiments."""
    rng = np.random.default_rng(seed)
    kinds = classify_features(feature_names)
    if attack_type == "label_flip":
        return poison_label_flip(X, y, ratio, rng)
    if attack_type == "feature_perturb":
        return poison_feature_perturb(X, y, ratio, rng, kinds)
    if attack_type == "trigger":
        return poison_trigger(X, y, ratio, rng, kinds)
    raise ValueError(f"unknown attack_type: {attack_type}")


def _mask(n: int, rows: np.ndarray) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[rows] = True
    return m
