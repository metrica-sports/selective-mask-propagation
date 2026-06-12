"""Download precomputed detections and embeddings from HuggingFace.

Downloads YOLOX detections (det/det.txt) and OSNet embeddings (emb/emb.npy)
for each clip. These are required for --precomputed mode.

Usage:
    cd SAM-Deep-EIoU
    uv run python scripts/download_precomputed.py sportsmot --split val       # 1.4 GB
    uv run python scripts/download_precomputed.py sportsmot                   # all splits, 7.5 GB
    uv run python scripts/download_precomputed.py dancetrack                  # val only, 0.6 GB
"""

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "holma91/SAM-Deep-EIoU"

DATASETS = {
    "sportsmot": {
        "splits": ["val", "train", "test"],
        "data_root": Path("data/sportsmot/dataset"),
        "tar_name": lambda split: f"precomputed/{split}.tar",
    },
    "dancetrack": {
        "splits": ["val"],
        "data_root": Path("data/dancetrack"),
        "tar_name": lambda split: f"precomputed/dancetrack_{split}.tar",
    },
}


def download_split(dataset: str, split: str) -> None:
    cfg = DATASETS[dataset]
    split_dir = cfg["data_root"] / split
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist. Download {dataset} first.")
        return

    filename = cfg["tar_name"](split)
    print(f"Downloading {filename}...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )

    print(f"Extracting into {split_dir}...")
    with tarfile.open(local_path, "r") as tar:
        tar.extractall(path=split_dir)

    print(f"{split}: done")


def main():
    parser = argparse.ArgumentParser(description="Download precomputed detections/embeddings")
    parser.add_argument("dataset", choices=DATASETS.keys(), help="Dataset to download")
    parser.add_argument("--split", nargs="+", default=None,
                        help="Splits to download (default: all available)")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    splits = args.split or cfg["splits"]
    for split in splits:
        if split not in cfg["splits"]:
            print(f"Warning: {split} is not available for {args.dataset}, skipping")
            continue
        download_split(args.dataset, split)

    print("\nDone. You can now run the pipeline with --precomputed.")


if __name__ == "__main__":
    main()
