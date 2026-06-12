"""DE + uniform SAM3 baseline runner.

§4.7 controlled comparator. Deep-EIoU produces tracks; SAM3 propagates a
mask for every track from inception to track death — uniformly, with no
margin signal and no exit logic. Same detection input and same track
lifecycle as the framework; the only difference is dispatch (uniform vs
selective). This is the apples-to-apples baseline for the §4.7.1 scaling
chart and the controlled comparator for §4.7.2 accuracy.

Per-frame tracker output:
    bbox = tight bbox of SAM3 mask, when non-empty
    bbox = DE bbox, when SAM3's mask for this track is empty/lost

So uniform's mask drift / collapse / convergence shows up directly in the
exported MOT (which is the whole point of the §4.7 dominance argument).

Multi-clip runner: SAM3 predictor loaded once and reused. Each clip
persists artifacts incrementally. Idempotent on ``timing.json`` existence;
``--force`` invalidates.

CLI:
    uv run python -m sam_deep_eiou.experiments.de_uniform_sam3 <clip>... [--stop-at N] [--force] [--no-render]

Outputs (per clip, in results/<parent>/<clip>/de_uniform_sam3/):
    mot_deep_eiou.txt        Deep-EIoU baseline (always produced).
    mot_de_uniform_sam3.txt  Uniform-SAM3 output (mask bbox or DE fallback).
    de_uniform_sam3.mp4      Render with mask overlays + IDs (and GT unless --no-gt).
    timing.json              Per-clip timing breakdown + peak VRAM.
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..core.detection import detect_precomputed
from ..core.merge import TrackData
from ..core.render import render_tracker_video
from ..core.sam3 import build_predictor, run_sam_uniform
from ..deep_eiou.tracker import step_track
from ..utils.export import export_mot
from ..utils.helpers import read_sequence_info


def _truncate_frames(frame_dict: Dict[int, np.ndarray], stop_at: int) -> Dict[int, np.ndarray]:
    return {f: v for f, v in frame_dict.items() if f < stop_at}


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def run_de_uniform_sam3_on_clip(
    predictor,
    source_path: str,
    stop_at: Optional[int],
) -> Tuple[
    Dict[int, Dict[int, np.ndarray]],
    Dict[int, Dict[int, np.ndarray]],
    Dict[int, Dict[int, TrackData]],
    Dict[str, float],
]:
    """Run Deep-EIoU + uniform SAM3 on one clip.

    Pipeline:
        1. Load precomputed YOLOX detections + OSNet embeddings.
        2. Optionally truncate to first ``stop_at`` frames.
        3. Run Deep-EIoU tracker — produces raw tracks. Margins ignored
           (uniform mode doesn't trigger on them).
        4. Run uniform SAM3: for every DE track, propagate from inception
           to death. No exit logic.
        5. Build per-frame output: mask's tight bbox when SAM has one,
           DE bbox otherwise.

    Returns:
        de_tracks: {frame: {track_id: bbox}} — Deep-EIoU baseline.
        out_tracks: {frame: {track_id: bbox}} — uniform-SAM3 output.
        merged: {frame: {track_id: TrackData(bbox, mask)}} — for render.
        stats: {detect_load_s, track_s, sam_total_s, n_tracks}.
    """
    seq_info = read_sequence_info(source_path)

    t = time.perf_counter()
    detections, embeddings = detect_precomputed(source_path)
    if stop_at is not None:
        detections = _truncate_frames(detections, stop_at)
        embeddings = _truncate_frames(embeddings, stop_at)
    detect_load_s = time.perf_counter() - t

    t = time.perf_counter()
    de_tracks, _ = step_track(
        detections, embeddings,
        seq_info["fps"], seq_info["width"], seq_info["height"],
    )
    track_s = time.perf_counter() - t

    t = time.perf_counter()
    sam_masks = run_sam_uniform(predictor, de_tracks, source_path)
    sam_total_s = time.perf_counter() - t

    out_tracks: Dict[int, Dict[int, np.ndarray]] = {}
    merged: Dict[int, Dict[int, TrackData]] = {}
    for frame_idx, de_frame in de_tracks.items():
        out_frame: Dict[int, np.ndarray] = {}
        merged_frame: Dict[int, TrackData] = {}
        sam_frame = sam_masks.get(frame_idx, {})
        for track_id, de_bbox in de_frame.items():
            mask = sam_frame.get(track_id)
            if mask is not None and mask.any():
                bbox = _bbox_from_mask(mask)
            else:
                bbox = de_bbox
                mask = None
            out_frame[track_id] = bbox
            merged_frame[track_id] = TrackData(bbox=bbox, mask=mask)
        if out_frame:
            out_tracks[frame_idx] = out_frame
            merged[frame_idx] = merged_frame

    stats = {
        "detect_load_s": detect_load_s,
        "track_s": track_s,
        "sam_total_s": sam_total_s,
        "n_frames": len(de_tracks),
        "n_tracks": len({tid for f in de_tracks.values() for tid in f}),
    }
    return de_tracks, out_tracks, merged, stats


def _output_dir(output_root: Path, source_path: str) -> Path:
    clip = Path(source_path).name
    parent = Path(source_path).parent.name
    return output_root / parent / clip / "de_uniform_sam3"


def _process_clip(
    predictor,
    output_root: Path,
    source_path: str,
    stop_at: Optional[int],
    no_gt: bool,
    no_render: bool,
    force: bool,
) -> bool:
    seq_name = Path(source_path).name
    out_dir = _output_dir(output_root, source_path)
    timing_path = out_dir / "timing.json"

    if timing_path.exists() and not force:
        print(f"[{seq_name}/de_uniform] skip (timing.json exists; pass --force to rerun)")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_total = time.perf_counter()

    print(f"\n[{seq_name}/de_uniform] starting")

    de_tracks, out_tracks, merged, stats = run_de_uniform_sam3_on_clip(
        predictor, source_path, stop_at,
    )

    de_path = out_dir / "mot_deep_eiou.txt"
    out_path = out_dir / "mot_de_uniform_sam3.txt"
    export_mot(de_tracks, str(de_path))
    export_mot(out_tracks, str(out_path))

    render_s = 0.0
    if not no_render:
        render_path = out_dir / "de_uniform_sam3.mp4"
        t = time.perf_counter()
        render_tracker_video(source_path, str(render_path), merged, no_gt=no_gt)
        render_s = time.perf_counter() - t

    timing = {
        **stats,
        "render_s": render_s,
        "total_s": time.perf_counter() - t_total,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
        "stop_at": stop_at,
    }
    timing_path.write_text(json.dumps(timing, indent=2))
    print(f"[{seq_name}/de_uniform] done — total={timing['total_s']:.1f}s "
          f"sam={stats['sam_total_s']:.1f}s render={render_s:.1f}s "
          f"peak_vram={timing['peak_vram_mb']:.0f}MB tracks={stats['n_tracks']}")
    return True


def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="DE + uniform SAM3 baseline. Deep-EIoU tracks + SAM3 propagated for every track from inception to death."
    )
    parser.add_argument("clips", nargs="+",
                        help="One or more sequence directories.")
    parser.add_argument("--stop-at", type=int, default=None,
                        help="Limit each clip to first N frames")
    parser.add_argument("--no-gt", action="store_true",
                        help="Skip GT overlay in rendered video")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip rendering (MOT export only)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run clips even if timing.json already exists")
    parser.add_argument("--output-root", type=Path, default=Path("results"),
                        help="Root dir for per-clip output (default: results/). "
                             "Per-clip path is <root>/<parent>/<clip>/de_uniform_sam3/.")
    args = parser.parse_args(args_list)

    todo = []
    skipped = []
    for source_path in args.clips:
        timing_path = _output_dir(args.output_root, source_path) / "timing.json"
        if timing_path.exists() and not args.force:
            skipped.append(source_path)
        else:
            todo.append(source_path)

    for source_path in skipped:
        print(f"[{Path(source_path).name}/de_uniform] skip (timing.json exists)")

    if not todo:
        print("Nothing to do.")
        return

    print(f"\nProcessing {len(todo)} clip(s).\n")
    print("Building SAM3 model...")
    t = time.perf_counter()
    predictor = build_predictor()
    model_load_s = time.perf_counter() - t
    print(f"SAM3 ready ({model_load_s:.1f}s).")

    n_done = 0
    n_failed = 0
    for source_path in todo:
        seq_name = Path(source_path).name
        try:
            _process_clip(
                predictor, args.output_root, source_path,
                stop_at=args.stop_at,
                no_gt=args.no_gt,
                no_render=args.no_render,
                force=args.force,
            )
            n_done += 1
        except Exception as e:
            n_failed += 1
            print(f"[{seq_name}/de_uniform] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\nSummary: {n_done} done, {len(skipped)} skipped, {n_failed} failed; "
          f"model_load_s={model_load_s:.1f} (paid once)")


if __name__ == "__main__":
    main()
