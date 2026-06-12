"""Convert the HF ViTPose checkpoint to the local safetensors format.

One-time utility: downloads usyd-community/vitpose-plus-base via
transformers, dumps its state dict (key names already match model.py)
and config into checkpoints/ for loader.py. Requires `transformers`
installed (not a runtime dependency of this package).

Usage:
    uv run python -m selective_mask_propagation.vitpose.convert
"""

import json
from pathlib import Path

from safetensors.torch import save_file

MODEL = "usyd-community/vitpose-plus-base"
CHECKPOINTS = Path(__file__).parent / "checkpoints"


def convert():
    from transformers import VitPoseForPoseEstimation

    hf = VitPoseForPoseEstimation.from_pretrained(MODEL)
    sd = {k: v.contiguous() for k, v in hf.state_dict().items()}
    save_file(sd, str(CHECKPOINTS / "vitpose-plus-base.safetensors"))

    bb = hf.config.backbone_config
    config = {
        "backbone_config": {
            "hidden_size": bb.hidden_size,
            "image_size": list(bb.image_size),
            "layer_norm_eps": bb.layer_norm_eps,
            "mlp_ratio": bb.mlp_ratio,
            "num_attention_heads": bb.num_attention_heads,
            "num_channels": bb.num_channels,
            "num_experts": bb.num_experts,
            "num_hidden_layers": bb.num_hidden_layers,
            "part_features": bb.part_features,
            "patch_size": list(bb.patch_size),
            "qkv_bias": bb.qkv_bias,
        },
        "id2label": {str(k): v for k, v in hf.config.id2label.items()},
        "use_simple_decoder": hf.config.use_simple_decoder,
        "scale_factor": hf.config.scale_factor,
    }
    with open(CHECKPOINTS / "vitpose-plus-base-config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Converted {len(sd)} tensors from {MODEL} into {CHECKPOINTS}/")


if __name__ == "__main__":
    convert()
