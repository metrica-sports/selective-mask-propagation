"""Process-level model cache for GTA steps.

Both load from local safetensors via the standalone implementations in
``selective_mask_propagation.vitpose`` / ``.parseq`` — no transformers,
no torch.hub.
"""

import functools
from typing import Any, Tuple


@functools.cache
def get_vitpose() -> Tuple[Any, Any]:
    """Returns (processor, model)."""
    from ..vitpose.loader import load_vitpose_model
    return load_vitpose_model()


@functools.cache
def get_parseq() -> Tuple[Any, Any]:
    """Returns (model, tokenizer)."""
    from ..parseq.loader import load_parseq
    return load_parseq()
