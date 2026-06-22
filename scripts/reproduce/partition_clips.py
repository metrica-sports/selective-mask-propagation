"""Partition clips into N subsets balanced by frame count.

Usage:
    uv run python scripts/reproduce/partition_clips.py data/sportsmot/dataset/test 6
    uv run python scripts/reproduce/partition_clips.py data/sportsmot/dataset/val 3

Writes partitions/<split>_0.txt, partitions/<split>_1.txt, etc.
Each file contains one clip path per line.
"""

import argparse
import configparser
from pathlib import Path


def get_frame_count(clip_dir: Path) -> int:
    cfg = configparser.ConfigParser()
    cfg.read(clip_dir / "seqinfo.ini")
    return int(cfg["Sequence"]["seqLength"])


def partition(clip_dirs: list, n: int, big_threshold: int = 0) -> list:
    """Greedy load-balanced partitioning: assign heaviest clip to lightest bucket.

    If big_threshold > 0, clips with >= that many frames go into partition 0
    first, then the rest are balanced across all N partitions.
    """
    clips_with_frames = [(d, get_frame_count(d)) for d in clip_dirs]
    clips_with_frames.sort(key=lambda x: -x[1])

    buckets = [[] for _ in range(n)]
    bucket_loads = [0] * n

    remaining = []
    for clip_dir, frames in clips_with_frames:
        if big_threshold and frames >= big_threshold:
            buckets[0].append(clip_dir)
            bucket_loads[0] += frames
        else:
            remaining.append((clip_dir, frames))

    for clip_dir, frames in remaining:
        lightest = min(range(n), key=lambda i: bucket_loads[i])
        buckets[lightest].append(clip_dir)
        bucket_loads[lightest] += frames

    for i, b in enumerate(buckets):
        b.sort(key=lambda d: d.name)

    return buckets, bucket_loads


def main():
    parser = argparse.ArgumentParser(description="Partition clips for parallel runs.")
    parser.add_argument("input_dir", help="Directory containing clip subdirectories")
    parser.add_argument("n", type=int, help="Number of partitions")
    parser.add_argument("--big", type=int, default=0, metavar="FRAMES",
                        help="Clips with >= this many frames go into partition 0")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    split = input_dir.name

    clip_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and (d / "seqinfo.ini").exists()])
    print(f"Found {len(clip_dirs)} clips in {input_dir}")

    buckets, loads = partition(clip_dirs, args.n, big_threshold=args.big)

    out_dir = Path("partitions")
    out_dir.mkdir(exist_ok=True)

    for i, (bucket, load) in enumerate(zip(buckets, loads)):
        out_file = out_dir / f"{split}_{i}.txt"
        out_file.write_text("\n".join(str(d) for d in bucket) + "\n")
        max_frames = max(get_frame_count(d) for d in bucket) if bucket else 0
        print(f"  Partition {i}: {len(bucket)} clips, {load} frames (max {max_frames}) -> {out_file}")


if __name__ == "__main__":
    main()
