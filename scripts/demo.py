"""Run selective mask propagation on a bundled clip: before/after metrics + render.

Zero setup beyond `uv sync`: decodes the clip's video.mp4 to frames (once),
loads its committed Deep-EIoU tracks and margins, runs the SAM 3 step, applies
the corrections, evaluates before/after against the bundled ground truth, and
writes the overlay video to results/demo/. First run downloads the SAM 3
checkpoint.

Metrics are benchmark mode (what the paper reports); the video is the prod
view, where the mask is the output while a window is active. See the README's
"Benchmark mode vs prod mode".

    uv run python scripts/demo.py                     # basketball clip
    uv run python scripts/demo.py --clip football     # or volleyball
    uv run python scripts/demo.py --no-render         # metrics only, skip the video
"""

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

CLIPS = {
    "basketball": "v_00HRwkvvjtQ_c005",
    "football": "v_i2_L4qquVg0_c009",
    "volleyball": "v_0kUtTtmLaJA_c004",
}


def _ensure_frames(clip_dir: Path, n_frames: int) -> None:
    """Decode video.mp4 to img1/*.jpg (once)."""
    img_dir = clip_dir / "img1"
    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) == n_frames:
        return
    img_dir.mkdir(exist_ok=True)
    print(f"  decoding {clip_dir.name}/video.mp4 -> img1/")
    cap = cv2.VideoCapture(str(clip_dir / "video.mp4"))
    i = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), frame)
        i += 1
    cap.release()


def _evaluate(tracks, clip_dir: Path) -> dict:
    """Evaluate tracks against the clip's bundled ground truth."""
    from selective_mask_propagation.core.eval import _eval_mot
    from selective_mask_propagation.utils.export import export_mot

    tmp_dir = tempfile.mkdtemp(prefix="smp_demo_")
    try:
        mot_path = str(Path(tmp_dir) / f"{clip_dir.name}.txt")
        export_mot(tracks, mot_path, quiet=True)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, _ = _eval_mot(clip_dir.name, str(clip_dir), mot_path)
        return metrics
    finally:
        shutil.rmtree(tmp_dir)


def _render(clip_dir: Path, merged_prod) -> Path:
    """Write the overlay video: the prod view, where the mask is the output
    while a healthy window is open and the base tracker's box everywhere else."""
    from selective_mask_propagation.core.render import render_tracker_video

    out_dir = ROOT / "results" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_dir.name}.mp4"
    with contextlib.redirect_stdout(io.StringIO()):
        render_tracker_video(str(clip_dir), str(out_path), merged_prod, no_gt=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip", choices=CLIPS.keys(), default="basketball")
    parser.add_argument("--no-render", dest="render", action="store_false",
                        help="Skip writing the overlay video to results/demo/.")
    args = parser.parse_args()

    from selective_mask_propagation.core.merge import (
        extract_bboxes, step_merge, step_merge_prod,
    )
    from selective_mask_propagation.core.sam3 import build_predictor, step_sam
    from selective_mask_propagation.utils.artifacts import load_pickle
    from selective_mask_propagation.utils.helpers import read_sequence_info

    clip_dir = ROOT / "test" / CLIPS[args.clip]
    with contextlib.redirect_stdout(io.StringIO()):
        n_frames = read_sequence_info(str(clip_dir))["length"]
    print(f"{args.clip}  {clip_dir.name}  ({n_frames} frames)")
    _ensure_frames(clip_dir, n_frames)

    predictor = build_predictor()
    with contextlib.redirect_stdout(io.StringIO()):
        tracks = load_pickle("tracks", clip_dir)
        margins = load_pickle("margins", clip_dir)
        t0 = time.perf_counter()
        sam_masks, windows, rename_events, match_history = step_sam(
            predictor, tracks, margins, str(clip_dir))
        # Benchmark merge for the metrics, prod merge for the render.
        # Both are cheap CPU next to the SAM pass.
        merged, _, rename_map = step_merge(tracks, sam_masks, windows, margins, rename_events)
        corrected = extract_bboxes(merged, tracks, rename_map, match_history, windows)
        merged_prod, _, _ = step_merge_prod(tracks, sam_masks, windows, margins, rename_events)
        dt = time.perf_counter() - t0
    swaps = sum(1 for w in windows if w.outcome.value == "swap")
    print(f"  SAM 3: {len(windows)} windows opened, {swaps} exited as SWAP  ({dt:.0f}s)")

    before = _evaluate(tracks, clip_dir)
    after = _evaluate(corrected, clip_dir)

    line = "=" * 56
    print("\n" + line)
    print(f"  {'':<18}{'HOTA':>8}{'AssA':>8}{'IDF1':>8}{'IDSW':>8}")
    print("-" * 56)
    print(f"  {'Deep-EIoU':<18}{before['hota']:>8.1f}{before['assa']:>8.1f}"
          f"{before['idf1']:>8.1f}{before['idsw']:>8}")
    print(f"  {'SAM3-Deep-EIoU':<18}{after['hota']:>8.1f}{after['assa']:>8.1f}"
          f"{after['idf1']:>8.1f}{after['idsw']:>8}")
    print("-" * 56)
    print(f"  {'delta':<18}{after['hota'] - before['hota']:>+8.1f}"
          f"{after['assa'] - before['assa']:>+8.1f}"
          f"{after['idf1'] - before['idf1']:>+8.1f}"
          f"{after['idsw'] - before['idsw']:>+8}")
    print(line)

    if args.render:
        print("\n  rendering overlay video...")
        out_path = _render(clip_dir, merged_prod)
        print(f"  overlay video: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
