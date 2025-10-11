"""Model registry for the Evo-Neuro benchmark."""

from .cnidarian import CnidarianNerveNet
from .segmented import SegmentedGanglia, SegmentedGangliaRestricted
from .cephalopod import CephalopodBrainV3
from .fish import FishBrainV3
from .human import HumanCortexV4

__all__ = [
    "CnidarianNerveNet",
    "SegmentedGanglia",
    "SegmentedGangliaRestricted",
    "CephalopodBrainV3",
    "FishBrainV3",
    "HumanCortexV4",
    "build_models",
]


def build_models(device="cpu"):
    """Construct the default suite of benchmark models."""

    return {
        "1_Cnidarian": CnidarianNerveNet(motor_dim=8).to(device),
        "2_SegmentedRestricted": SegmentedGangliaRestricted(segments=6, motor_per_seg=2).to(device),
        "3_Cephalopod": CephalopodBrainV3(motor_dim=4).to(device),
        "4_Fish": FishBrainV3(motor_dim=12).to(device),
        "5_Human": HumanCortexV4(motor_dim=20).to(device),
    }
