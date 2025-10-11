"""Utility functions and shared helpers for the Evo-Neuro benchmark."""

# ===============================================================
# Minimal Evo-Neuro Benchmark (Colab-ready, single cell)
# Models: CnidarianNerveNet / SegmentedGanglia / FishBrain / HumanExecutive
# Tasks : HD-Jellyfish (SL, 3-way) + Reversal (RL, 2AFC)
# ===============================================================

import math
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "set_seed",
    "D_VISION",
    "D_OLFACT",
    "D_SOMATO",
    "D_AUDIT",
    "D_PROP",
    "INPUT_DIM",
    "make_obs",
    "MLP",
    "FrozenObsMixer",
    "V_BASE",
    "S_BASE",
    "compute_tal_metrics",
    "compute_efficiency",
]

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 0) -> None:
    """Set random seeds across numpy/torch/python for deterministic behaviour."""

    import torch  # local import to avoid circular dependency during packaging

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Global sensory dimensions (fixed across models)
# -----------------------------
D_VISION, D_OLFACT, D_SOMATO, D_AUDIT, D_PROP = 128, 32, 60, 64, 16
INPUT_DIM = D_VISION + D_OLFACT + D_SOMATO + D_AUDIT + D_PROP


# -----------------------------
# Obs helper (all modalities; allow overwrite)
# -----------------------------
def make_obs(
    batch: int = 1,
    device: Optional[str] = "cpu",
    vision: Optional[torch.Tensor] = None,
    olfaction: Optional[torch.Tensor] = None,
    somato: Optional[torch.Tensor] = None,
    auditory: Optional[torch.Tensor] = None,
    proprio: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Create a full sensory observation dictionary with optional overrides."""

    obs = {
        "vision": torch.randn(batch, D_VISION, device=device),
        "olfaction": torch.randn(batch, D_OLFACT, device=device),
        "somatosensory": torch.randn(batch, D_SOMATO, device=device),
        "auditory": torch.randn(batch, D_AUDIT, device=device),
        "proprioception": torch.randn(batch, D_PROP, device=device),
    }
    if vision is not None:
        obs["vision"] = vision
    if olfaction is not None:
        obs["olfaction"] = olfaction
    if somato is not None:
        obs["somatosensory"] = somato
    if auditory is not None:
        obs["auditory"] = auditory
    if proprio is not None:
        obs["proprioception"] = proprio
    return obs


# -----------------------------
# Tiny block
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth=2, act=nn.ReLU):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), act()]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


# ===============================================================
# Frozen observation mixer (shared across tasks)
# ===============================================================
class FrozenObsMixer(nn.Module):
    """Low-dim bases -> fixed linear lift to Vision128 / Somato60."""

    def __init__(self):
        super().__init__()
        with torch.no_grad():
            Wv = torch.randn(V_BASE, D_VISION) / math.sqrt(V_BASE)
            Ws = torch.randn(S_BASE, D_SOMATO) / math.sqrt(S_BASE)
        self.register_buffer("Wv", Wv)
        self.register_buffer("Ws", Ws)

    @torch.no_grad()
    def forward(self, v_base, s_base):
        return v_base @ self.Wv, s_base @ self.Ws


# These bases are used by multiple tasks; keep definitions close to the mixer.
V_BASE, S_BASE = 8, 16


def compute_tal_metrics(curve, criterion=0.85, window=100, budget=2000):
    """
    curve: list of (trial_idx, acc)
    return: TTC, AUC@B, Asy@B
    """

    accs = [a for _, a in curve]
    trials = [t for t, _ in curve]

    # TTC
    ttc = None
    for i in range(window, len(accs)):
        if sum(accs[i - window : i]) / window >= criterion:
            ttc = trials[i]
            break

    # AUC@B
    auc = sum(accs[:budget]) / min(budget, len(accs))

    # Asymptote
    asy = sum(accs[-window:]) / min(window, len(accs))

    return ttc, auc, asy


def compute_efficiency(auc, params, trials):
    """Eff@B = AUC@B / (params * trials)"""

    return auc / max(1, (params * trials))
