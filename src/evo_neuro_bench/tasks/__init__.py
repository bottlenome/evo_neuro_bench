"""Tasks and training utilities for the Evo-Neuro benchmark."""

from .hd_jellyfish import HDJellyfishDataset, train_hd_jellyfish
from .reversal import ReversalEnv, train_reversal
from .detour import HardDetourDataset, train_detour
from .local_reflex import LocalReflexDataset, train_local_reflex
from .peristalsis import PeristalsisDataset, train_peristalsis, phase_corr
from .rpm_mini import RPMMiniDataset, train_rpm_mini
from .arc_mini import ARCMiniDataset, train_arc_mini
from .grid_firststep import GridPathFirstStep, train_grid_firststep

__all__ = [
    "HDJellyfishDataset",
    "train_hd_jellyfish",
    "ReversalEnv",
    "train_reversal",
    "HardDetourDataset",
    "train_detour",
    "LocalReflexDataset",
    "train_local_reflex",
    "PeristalsisDataset",
    "train_peristalsis",
    "phase_corr",
    "RPMMiniDataset",
    "train_rpm_mini",
    "ARCMiniDataset",
    "train_arc_mini",
    "GridPathFirstStep",
    "train_grid_firststep",
]
