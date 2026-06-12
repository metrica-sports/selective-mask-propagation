"""Model cache for GTA steps.

Heavy models (ViTPose, PARSeq) persist across clips within the same
process. This avoids reloading on every clip when using glob patterns.

Both load from local safetensors via the standalone implementations in
``selective_mask_propagation.vitpose`` / ``.parseq`` — no transformers,
no torch.hub.
"""

from typing import Any, Optional, Tuple

_vitpose: Optional[Tuple[Any, Any]] = None
_parseq: Optional[Tuple[Any, Any]] = None


def get_vitpose() -> Tuple[Any, Any]:
    """Returns (processor, model)."""
    global _vitpose
    if _vitpose is None:
        from ..vitpose.loader import load_vitpose_model
        _vitpose = load_vitpose_model()
    return _vitpose


def get_parseq() -> Tuple[Any, Any]:
    """Returns (model, tokenizer)."""
    global _parseq
    if _parseq is None:
        from ..parseq.loader import load_parseq
        _parseq = load_parseq()
    return _parseq
