"""Convert the released baudm/parseq checkpoint to local safetensors.

One-time utility: run once when first setting up PARSeq, or
again if the upstream URL changes. Downloads the .pt checkpoint via
torch.hub, saves it as safetensors with a matching config json under
checkpoints/, where loader.py picks it up.

The released `parseq-bb5792a6.pt` file is already a flat OrderedDict
matching the inner PARSeq nn.Module's state_dict — there's no Lightning
prefix to strip and no key remapping is needed.

Usage:
    uv run python -m selective_mask_propagation.parseq.convert
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from .model import PARSeqConfig

WEIGHTS_URL = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt"

_WEIGHTS_DIR = Path(__file__).parent / "checkpoints"
DST_WEIGHTS = _WEIGHTS_DIR / "parseq.safetensors"
DST_CONFIG = _WEIGHTS_DIR / "parseq-config.json"


def convert():
    print(f"Downloading {WEIGHTS_URL}")
    state_dict = torch.hub.load_state_dict_from_url(
        url=WEIGHTS_URL, map_location="cpu", check_hash=True
    )
    print(f"Loaded {len(state_dict)} tensors from upstream checkpoint")

    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    save_file(dict(state_dict), str(DST_WEIGHTS))
    print(f"Saved: {DST_WEIGHTS}")

    config = asdict(PARSeqConfig())
    config["img_size"] = list(config["img_size"])
    config["patch_size"] = list(config["patch_size"])
    with open(DST_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {DST_CONFIG}")


if __name__ == "__main__":
    convert()
