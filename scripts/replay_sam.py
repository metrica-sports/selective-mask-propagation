"""Tier-1 regression check: replay the SAM step from frozen artifacts.

Loads tracks + margins from a golden artifacts dir, re-runs step_sam with
the current code (GPU), and compares windows, rename_events, match_history,
and sam_masks exactly against the stored artifacts. SAM inference is
bitwise deterministic on fixed hardware, so any diff is a code change.

    uv run python scripts/replay_sam.py <golden_artifacts_dir> <sequence_dir> [--sam3]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import selective_mask_propagation.core.sam2  # noqa: E402

# Artifacts pickled before the rename reference sam_deep_eiou.* paths.
sys.modules.setdefault("sam_deep_eiou", sys.modules["selective_mask_propagation"])
sys.modules.setdefault("sam_deep_eiou.core", sys.modules["selective_mask_propagation.core"])
sys.modules.setdefault("sam_deep_eiou.core.sam2", sys.modules["selective_mask_propagation.core.sam2"])


def _load(artifacts: Path, name: str):
    zst = artifacts / f"{name}.pkl.zst"
    pkl = artifacts / f"{name}.pkl"
    if zst.exists():
        import zstandard as zstd
        with open(zst, "rb") as f, zstd.ZstdDecompressor().stream_reader(f) as r:
            return pickle.load(r)
    with open(pkl, "rb") as f:
        return pickle.load(f)


def _masks_equal(a, b) -> bool:
    if a.keys() != b.keys():
        return False
    for frame in a:
        if a[frame].keys() != b[frame].keys():
            return False
        for cid in a[frame]:
            if not np.array_equal(a[frame][cid], b[frame][cid]):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifacts", help="Golden artifacts dir (tracks/margins in, sam outputs compared)")
    parser.add_argument("source", help="Sequence dir containing img1/ and seqinfo.ini")
    parser.add_argument("--sam3", action="store_true")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    tracks = _load(artifacts, "tracks")
    margins = _load(artifacts, "margins")

    if args.sam3:
        from selective_mask_propagation.core.sam3 import build_predictor, step_sam
    else:
        from selective_mask_propagation.core.sam2 import build_predictor, step_sam

    predictor = build_predictor()
    sam_masks, windows, rename_events, match_history = step_sam(
        predictor, tracks, margins, args.source,
    )

    checks = {
        "windows": windows == _load(artifacts, "windows"),
        "rename_events": rename_events == _load(artifacts, "rename_events"),
        "match_history": match_history == _load(artifacts, "match_history"),
        "sam_masks": _masks_equal(sam_masks, _load(artifacts, "sam_masks")),
    }
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'DIFF'}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
