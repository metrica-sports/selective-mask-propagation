"""Download model checkpoints from HuggingFace.

YOLOX + OSNet are only needed when running detection from scratch
(without --precomputed). ViTPose + PARSeq are needed for --gta; they can
also be regenerated locally via the convert.py in each package.

Usage:
    uv run python scripts/setup/download_checkpoints.py
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "holma91/SAM-Deep-EIoU"

CHECKPOINTS = [
    ("checkpoints/yolox_x_sports_train.pth.tar", "selective_mask_propagation/yolox/checkpoints/yolox_x_sports_train.pth.tar"),
    ("checkpoints/yolox_x_sports_mix.pth.tar", "selective_mask_propagation/yolox/checkpoints/yolox_x_sports_mix.pth.tar"),
    ("checkpoints/yolox_x_dancetrack.pth.tar", "selective_mask_propagation/yolox/checkpoints/yolox_x_dancetrack.pth.tar"),
    ("checkpoints/osnet_sports.pth.tar", "selective_mask_propagation/osnet/checkpoints/sports_model.pth.tar-60"),
    ("checkpoints/parseq.safetensors", "selective_mask_propagation/parseq/checkpoints/parseq.safetensors"),
    ("checkpoints/parseq-config.json", "selective_mask_propagation/parseq/checkpoints/parseq-config.json"),
    ("checkpoints/vitpose-plus-base.safetensors", "selective_mask_propagation/vitpose/checkpoints/vitpose-plus-base.safetensors"),
    ("checkpoints/vitpose-plus-base-config.json", "selective_mask_propagation/vitpose/checkpoints/vitpose-plus-base-config.json"),
]


def main():
    for hf_path, local_path in CHECKPOINTS:
        dest = Path(local_path)
        if dest.exists():
            print(f"  Exists: {dest}")
            continue

        print(f"Downloading {hf_path}...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_path,
            repo_type="dataset",
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Saved: {dest} ({size_mb:.0f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
