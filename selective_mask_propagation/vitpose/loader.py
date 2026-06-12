"""Standalone ViTPose loader — drop-in replacement for transformers ViTPose.

Usage:
    from selective_mask_propagation.vitpose.loader import load_vitpose_model
    processor, model = load_vitpose_model()
"""

import json
from pathlib import Path

from safetensors.torch import load_file

from .model import VitPoseConfig, VitPoseForPoseEstimation
from .processor import VitPoseProcessor

WEIGHTS_PATH = Path(__file__).parent / "checkpoints" / "vitpose-plus-base.safetensors"
CONFIG_PATH = Path(__file__).parent / "checkpoints" / "vitpose-plus-base-config.json"


def load_vitpose_model() -> tuple:
    """Load ViTPose model and processor.

    Returns:
        (processor, model) tuple compatible with the transformers API.
    """
    with open(CONFIG_PATH) as f:
        config = VitPoseConfig.from_hf_dict(json.load(f))

    model = VitPoseForPoseEstimation(config)
    model.load_state_dict(load_file(str(WEIGHTS_PATH)))
    model.eval()
    model = model.to("cuda")

    print(f"ViTPose loaded on {next(model.parameters()).device}.")
    return VitPoseProcessor(), model
