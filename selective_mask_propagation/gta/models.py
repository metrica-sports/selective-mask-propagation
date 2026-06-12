"""Model cache for GTA steps.

Heavy models (ViTPose, PARSeq) persist across clips within the same
process. This avoids reloading on every clip when using glob patterns.
"""

from typing import Any, Optional, Tuple

import torch

_vitpose: Optional[Tuple[Any, Any]] = None
_parseq: Optional[Any] = None

VITPOSE_MODEL = "usyd-community/vitpose-plus-base"


def get_vitpose() -> Tuple[Any, Any]:
    global _vitpose
    if _vitpose is None:
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        print(f"Loading ViTPose: {VITPOSE_MODEL}...")
        processor = AutoProcessor.from_pretrained(VITPOSE_MODEL)
        model = VitPoseForPoseEstimation.from_pretrained(VITPOSE_MODEL)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _vitpose = (processor, model)
        print(f"ViTPose ready on {model.device}.")
    return _vitpose


def get_parseq() -> Any:
    global _parseq
    if _parseq is None:
        print("Loading PARSeq...")
        _parseq = torch.hub.load("baudm/parseq", "parseq", pretrained=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _parseq = _parseq.to(device).eval()
        print(f"PARSeq ready on {device}.")
    return _parseq
