"""Regression check: replay the merge step from frozen artifacts.

For each clip dir (containing artifacts/), loads the SAM-step outputs,
re-runs step_merge + extract_bboxes with the current code, and compares
the exported MOT byte-for-byte against the stored mot_sam_deep_eiou.txt.

Catches any behavior change in merge/rename/extract logic without GPU
or re-running SAM. Point it at a results tree from a known-good run:

    uv run python scripts/replay_merge.py path/to/results/test/*-sam3
"""

import argparse
import contextlib
import io
import pickle
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import selective_mask_propagation.core.sam2  # noqa: E402
from selective_mask_propagation.core.merge import step_merge, extract_bboxes  # noqa: E402
from selective_mask_propagation.utils.export import export_mot  # noqa: E402

# Artifacts pickled before the selective_mask_propagation rename reference sam_deep_eiou.* paths.
sys.modules.setdefault("sam_deep_eiou", sys.modules["selective_mask_propagation"])
sys.modules.setdefault("sam_deep_eiou.core", sys.modules["selective_mask_propagation.core"])
sys.modules.setdefault("sam_deep_eiou.core.sam2", sys.modules["selective_mask_propagation.core.sam2"])


def _load(artifacts: Path, name: str, allow_missing: bool = False):
    zst = artifacts / f"{name}.pkl.zst"
    pkl = artifacts / f"{name}.pkl"
    if zst.exists():
        import zstandard as zstd
        with open(zst, "rb") as f, zstd.ZstdDecompressor().stream_reader(f) as r:
            return pickle.load(r)
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    if allow_missing:
        return None
    raise FileNotFoundError(pkl)


def replay_clip(clip_dir: Path) -> str:
    artifacts = clip_dir / "artifacts"
    golden = artifacts / "mot_sam_deep_eiou.txt"
    if not golden.exists():
        return "SKIP (no golden MOT)"

    tracks = _load(artifacts, "tracks")
    margins = _load(artifacts, "margins")
    sam_masks = _load(artifacts, "sam_masks")
    windows = _load(artifacts, "windows")
    rename_events = _load(artifacts, "rename_events")
    match_history = _load(artifacts, "match_history", allow_missing=True) or {}

    with contextlib.redirect_stdout(io.StringIO()):
        merged, _, rename_map = step_merge(tracks, sam_masks, windows, margins, rename_events)
        bboxes = extract_bboxes(merged, tracks, rename_map, match_history, windows)

    with tempfile.NamedTemporaryFile(mode="r+", suffix=".txt") as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            export_mot(bboxes, tmp.name)
        replayed = Path(tmp.name).read_text()

    expected = golden.read_text()
    if replayed == expected:
        return "PASS"
    exp_lines = expected.splitlines()
    got_lines = replayed.splitlines()
    n_diff = sum(1 for a, b in zip(exp_lines, got_lines) if a != b)
    n_diff += abs(len(exp_lines) - len(got_lines))
    return f"DIFF ({n_diff}/{len(exp_lines)} lines)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clips", nargs="+", help="Clip result dirs containing artifacts/")
    args = parser.parse_args()

    results = {}
    for clip in sorted(args.clips):
        clip_dir = Path(clip)
        try:
            status = replay_clip(clip_dir)
        except Exception:
            status = "ERROR"
            traceback.print_exc()
        results[str(clip_dir)] = status
        print(f"{status:24s} {clip_dir.name}")

    n_pass = sum(1 for s in results.values() if s == "PASS")
    n_skip = sum(1 for s in results.values() if s.startswith("SKIP"))
    n_bad = len(results) - n_pass - n_skip
    print(f"\n{n_pass} pass, {n_bad} diff/error, {n_skip} skip of {len(results)}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
