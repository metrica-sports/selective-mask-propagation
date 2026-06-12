"""Pure-SAM3 baseline: native open-vocab tracking with text='player'.

§4.7 baseline: SAM3 used in its native mode (detector + tracker + temporal
disambiguation). SAM3 finds 'player' instances on every frame and assigns
persistent obj_ids on its own. No YOLOX, no IoU gate, no Deep-EIoU.

Multi-clip runner: SAM3 predictor is loaded once and reused across every
clip in the input list. Each clip's artifacts persist incrementally to its
own output dir. A clip with an existing ``timing.json`` is skipped unless
``--force`` is passed; a crash on one clip prints a traceback and continues
to the next. Re-running the same command is idempotent.

CLI:
    uv run python -m smp.experiments.pure_sam3 <clip_dir> [<clip_dir> ...] [--text "player"] [--stop-at N] [--force] [--no-render]

Outputs (per clip, in results/<parent>/<clip>/pure_sam3/):
    mot_pure_sam3.txt    MOT-format tight-mask bboxes per frame.
    pure_sam3.mp4        Render with mask overlays + IDs (and GT unless --no-gt).
    timing.json          Per-clip timing breakdown + peak VRAM. Acts as the
                         "done" marker for skip-if-exists.
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..utils.export import export_mot
from ..utils.helpers import read_sequence_info


def _bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def _to_numpy(x):
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def _build_predictor():
    """Build a single Sam3VideoPredictor for reuse across clips."""
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    repo_root = Path(__file__).parent.parent
    bpe_path = str(repo_root / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")

    print("Building SAM3 video predictor...")
    t = time.perf_counter()
    predictor = Sam3VideoPredictor(bpe_path=bpe_path)
    elapsed = time.perf_counter() - t
    print(f"SAM3 ready ({elapsed:.1f}s).")
    return predictor, elapsed


def run_pure_sam3_on_clip(
    predictor,
    source_path: str,
    text_prompt: str,
    stop_at: Optional[int],
) -> Tuple[
    Dict[int, Dict[int, np.ndarray]],
    Dict[int, Dict[int, np.ndarray]],
    Dict[str, float],
]:
    """Run native SAM3 open-vocab tracking on a single clip.

    Returns (sam_masks, tracks, stats). stats holds session_init_s,
    propagation_s, n_obj_ids, frames.
    """
    img_dir = str(Path(source_path) / "img1")

    t = time.perf_counter()
    response = predictor.start_session(resource_path=img_dir)
    session_id = response["session_id"]
    session_init_s = time.perf_counter() - t

    predictor.add_prompt(
        session_id=session_id,
        frame_idx=0,
        text=text_prompt,
    )

    sam_masks: Dict[int, Dict[int, np.ndarray]] = {}
    tracks: Dict[int, Dict[int, np.ndarray]] = {}

    max_frame_num_to_track = stop_at if stop_at is not None else None

    t = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        iterator = predictor.propagate_in_video(
            session_id=session_id,
            propagation_direction="forward",
            start_frame_idx=0,
            max_frame_num_to_track=max_frame_num_to_track,
        )
        for response in tqdm(iterator, desc="Pure SAM3"):
            frame_idx = int(response["frame_index"])
            out = response["outputs"]

            obj_ids = _to_numpy(out["out_obj_ids"])
            binary_masks = out["out_binary_masks"]

            frame_masks: Dict[int, np.ndarray] = {}
            frame_tracks: Dict[int, np.ndarray] = {}
            for idx, oid in enumerate(obj_ids.tolist()):
                mask = binary_masks[idx]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                mask = np.asarray(mask).astype(bool)
                if not mask.any():
                    continue
                bbox = _bbox_from_mask(mask)
                if bbox is None:
                    continue
                frame_masks[int(oid)] = mask
                frame_tracks[int(oid)] = bbox

            if frame_masks:
                sam_masks[frame_idx] = frame_masks
                tracks[frame_idx] = frame_tracks
    propagation_s = time.perf_counter() - t

    predictor.close_session(session_id=session_id)

    n_objs = len({oid for f in tracks for oid in tracks[f]})
    stats = {
        "session_init_s": session_init_s,
        "propagation_s": propagation_s,
        "n_obj_ids": n_objs,
        "frames": len(tracks),
    }
    return sam_masks, tracks, stats


def _output_dir(output_root: Path, source_path: str) -> Path:
    clip = Path(source_path).name
    parent = Path(source_path).parent.name
    return output_root / parent / clip / "pure_sam3"


def _process_clip(
    predictor,
    output_root: Path,
    source_path: str,
    text_prompt: str,
    stop_at: Optional[int],
    no_gt: bool,
    no_render: bool,
    force: bool,
) -> bool:
    """Run pure SAM3 on one clip, persist artifacts. Returns True on success."""
    seq_name = Path(source_path).name
    out_dir = _output_dir(output_root, source_path)
    timing_path = out_dir / "timing.json"

    if timing_path.exists() and not force:
        print(f"[{seq_name}] skip (timing.json exists; pass --force to rerun)")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_total = time.perf_counter()

    print(f"\n[{seq_name}] starting")
    read_sequence_info(source_path)
    if stop_at is not None:
        print(f"  stop-at: first {stop_at} frames")

    sam_masks, tracks, stats = run_pure_sam3_on_clip(
        predictor, source_path, text_prompt, stop_at,
    )

    mot_path = out_dir / "mot_pure_sam3.txt"
    export_mot(tracks, str(mot_path))

    render_s = 0.0
    if not no_render:
        from ..core.merge import TrackData
        from ..core.render import render_tracker_video

        merged: Dict[int, Dict[int, TrackData]] = {}
        for f, frame_tracks in tracks.items():
            merged[f] = {
                tid: TrackData(bbox=bbox, mask=sam_masks.get(f, {}).get(tid))
                for tid, bbox in frame_tracks.items()
            }

        render_path = out_dir / "pure_sam3.mp4"
        t = time.perf_counter()
        render_tracker_video(source_path, str(render_path), merged, no_gt=no_gt)
        render_s = time.perf_counter() - t

    timing = {
        **stats,
        "render_s": render_s,
        "total_s": time.perf_counter() - t_total,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
        "stop_at": stop_at,
        "text_prompt": text_prompt,
    }
    timing_path.write_text(json.dumps(timing, indent=2))
    print(f"[{seq_name}] done — total={timing['total_s']:.1f}s "
          f"propagation={stats['propagation_s']:.1f}s "
          f"peak_vram={timing['peak_vram_mb']:.0f}MB "
          f"obj_ids={stats['n_obj_ids']}")
    return True


def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Pure-SAM3 baseline (native open-vocab tracking with text prompt). "
                    "Loads SAM3 once and iterates clips. Idempotent: skips clips with existing timing.json."
    )
    parser.add_argument("clips", nargs="+", help="One or more sequence directories.")
    parser.add_argument("--stop-at", type=int, default=None,
                        help="Limit each clip to first N frames")
    parser.add_argument("--text", type=str, default="player",
                        help="Text prompt for SAM3 (default: 'player')")
    parser.add_argument("--no-gt", action="store_true",
                        help="Skip GT overlay in rendered video")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip rendering (MOT export only)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run clips even if timing.json already exists")
    parser.add_argument("--output-root", type=Path, default=Path("results"),
                        help="Root dir for per-clip output (default: results/). "
                             "Per-clip path is <root>/<parent>/<clip>/pure_sam3/.")
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
        print(f"[{Path(source_path).name}] skip (timing.json exists)")

    if not todo:
        print("Nothing to do.")
        return

    print(f"\nProcessing {len(todo)} clip(s).\n")
    predictor, model_load_s = _build_predictor()

    n_done = 0
    n_failed = 0
    for source_path in todo:
        seq_name = Path(source_path).name
        try:
            _process_clip(
                predictor, args.output_root, source_path,
                text_prompt=args.text,
                stop_at=args.stop_at,
                no_gt=args.no_gt,
                no_render=args.no_render,
                force=args.force,
            )
            n_done += 1
        except Exception as e:
            n_failed += 1
            print(f"[{seq_name}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\nSummary: {n_done} done, {len(skipped)} skipped, {n_failed} failed; "
          f"model_load_s={model_load_s:.1f} (paid once)")


if __name__ == "__main__":
    main()
