"""SAM3-Deep-EIoU framework runner: Deep-EIoU + selective SAM3 mask propagation.

§4.7 method: the framework. Margin-triggered selective dispatch from Deep-EIoU
to SAM3 only on windows where the assignment margin signals ambiguity. No GTA
(off-screen re-id is out of scope for §4.7's on-screen dominance argument).

Multi-clip runner: SAM3 predictor is loaded once and reused across every clip.
Each clip persists artifacts incrementally to its own output dir; a clip with
existing ``timing.json`` is skipped unless ``--force`` is passed; one bad clip
prints a traceback and continues. Re-running is idempotent.

Trigger ablation (§4.7.2 signal composition): ``--triggers`` controls which
window-creation signals fire:

    margin                  margin only
    margin_gap              margin + gap (no witness mechanism)
    margin_gap_witness      full (default; matches the thesis configuration)

Output dir is suffixed by the trigger config; the default
(``margin_gap_witness``) uses the unsuffixed ``sam3_deep_eiou/`` so the
canonical run lands in the natural place.

CLI:
    uv run python -m selective_mask_propagation.experiments.sam3_deep_eiou <clip>... [--triggers margin|margin_gap|margin_gap_witness] [--stop-at N] [--force] [--no-render]

Outputs (per clip, in results/<parent>/<clip>/sam3_deep_eiou[-<suffix>]/):
    mot_deep_eiou.txt        Deep-EIoU baseline (always produced; DE runs as substrate).
    mot_sam3_deep_eiou.txt   Framework output after selective SAM3 augmentation.
    sam3_deep_eiou.mp4       Render of framework output with masks (and GT unless --no-gt).
    timing.json              Per-clip timing breakdown + peak VRAM + ablation config.
                             Acts as the "done" marker for skip-if-exists.
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
from ..core.merge import TrackData, canon_key, extract_bboxes, step_merge
from ..core.render import render_tracker_video
from ..core.sam2 import WindowOutcome
from ..core.sam3 import build_predictor, step_sam
from ..deep_eiou.tracker import step_track
from ..utils.export import export_mot
from ..utils.helpers import read_sequence_info


TRIGGER_CHOICES = ["margin", "margin_gap", "margin_gap_witness"]


def _trigger_flags(triggers: str) -> Tuple[bool, bool]:
    """Map --triggers string to (enable_gap, enable_witness)."""
    return ("gap" in triggers, "witness" in triggers)


def _truncate_frames(frame_dict: Dict[int, np.ndarray], stop_at: int) -> Dict[int, np.ndarray]:
    return {f: v for f, v in frame_dict.items() if f < stop_at}


def run_framework_on_clip(
    predictor,
    source_path: str,
    stop_at: Optional[int],
    enable_gap: bool,
    enable_witness: bool,
) -> Tuple[
    Dict[int, Dict[int, np.ndarray]],
    Dict[int, Dict[int, np.ndarray]],
    Dict[int, Dict[int, TrackData]],
    Dict[str, float],
]:
    """Run Deep-EIoU + selective SAM3 framework on one clip.

    Pipeline:
        1. Load precomputed YOLOX detections + OSNet embeddings.
        2. Optionally truncate to first ``stop_at`` frames.
        3. Run Deep-EIoU tracker — produces raw tracks + per-frame margins.
        4. Run selective SAM3 with the requested triggers — produces windows,
           masks, rename events, match history. Predictor is supplied by
           caller; not built here.
        5. Merge SAM3 evidence (only SWAP windows produce overlays; CLEAN /
           EDGE / END / DEGRADED / STALE leave DE untouched).
        6. Extract framework bboxes from merged dict using match_history as
           the authoritative bbox lookup for mask frames.
        7. Materialize a render-ready merged dict: framework bboxes + masks
           where SWAP windows had them.

    Returns:
        de_tracks, sde_bboxes, merged, stats.
    """
    seq_info = read_sequence_info(source_path)

    t = time.perf_counter()
    detections, embeddings = detect_precomputed(source_path)
    if stop_at is not None:
        detections = _truncate_frames(detections, stop_at)
        embeddings = _truncate_frames(embeddings, stop_at)
    detect_load_s = time.perf_counter() - t

    t = time.perf_counter()
    de_tracks, margins = step_track(
        detections, embeddings,
        seq_info["fps"], seq_info["width"], seq_info["height"],
    )
    track_s = time.perf_counter() - t

    t = time.perf_counter()
    sam_masks, windows, rename_events, match_history = step_sam(
        predictor, de_tracks, margins, source_path,
        enable_gap=enable_gap, enable_witness=enable_witness,
    )
    sam_total_s = time.perf_counter() - t

    t = time.perf_counter()
    namespaced_merged, _, rename_map = step_merge(
        de_tracks, sam_masks, windows, margins, rename_events,
    )
    sde_bboxes = extract_bboxes(
        namespaced_merged, de_tracks, rename_map, match_history, windows,
    )

    merged: Dict[int, Dict[int, TrackData]] = {}
    for f, ft in sde_bboxes.items():
        frame_data: Dict[int, TrackData] = {}
        merged_frame = namespaced_merged.get(f, {})
        for tid, bbox in ft.items():
            entry = merged_frame.get(canon_key(tid)) or merged_frame.get(tid)
            mask = entry.mask if entry is not None else None
            frame_data[tid] = TrackData(bbox=bbox, mask=mask)
        merged[f] = frame_data
    merge_s = time.perf_counter() - t

    stats = {
        "detect_load_s": detect_load_s,
        "track_s": track_s,
        "sam_total_s": sam_total_s,
        "merge_s": merge_s,
        "n_frames": len(de_tracks),
        "n_windows": len(windows),
        "n_swaps": sum(1 for w in windows if w.outcome == WindowOutcome.SWAP),
    }
    return de_tracks, sde_bboxes, merged, stats


def _output_dir(output_root: Path, source_path: str, triggers: str) -> Path:
    clip = Path(source_path).name
    parent = Path(source_path).parent.name
    suffix = "" if triggers == "margin_gap_witness" else f"-{triggers}"
    return output_root / parent / clip / f"sam3_deep_eiou{suffix}"


def _process_clip(
    predictor,
    output_root: Path,
    source_path: str,
    triggers: str,
    stop_at: Optional[int],
    no_gt: bool,
    no_render: bool,
    force: bool,
) -> bool:
    """Run framework on one clip, persist artifacts. Returns True on success."""
    seq_name = Path(source_path).name
    out_dir = _output_dir(output_root, source_path, triggers)
    timing_path = out_dir / "timing.json"

    if timing_path.exists() and not force:
        print(f"[{seq_name}/{triggers}] skip (timing.json exists; pass --force to rerun)")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_total = time.perf_counter()

    enable_gap, enable_witness = _trigger_flags(triggers)
    print(f"\n[{seq_name}/{triggers}] starting (gap={enable_gap}, witness={enable_witness})")

    de_tracks, sde_bboxes, merged, stats = run_framework_on_clip(
        predictor, source_path, stop_at, enable_gap, enable_witness,
    )

    de_path = out_dir / "mot_deep_eiou.txt"
    sde_path = out_dir / "mot_sam3_deep_eiou.txt"
    export_mot(de_tracks, str(de_path))
    export_mot(sde_bboxes, str(sde_path))

    render_s = 0.0
    if not no_render:
        render_path = out_dir / "sam3_deep_eiou.mp4"
        t = time.perf_counter()
        render_tracker_video(source_path, str(render_path), merged, no_gt=no_gt)
        render_s = time.perf_counter() - t

    timing = {
        **stats,
        "render_s": render_s,
        "total_s": time.perf_counter() - t_total,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
        "stop_at": stop_at,
        "config": {
            "triggers": triggers,
            "enable_gap": enable_gap,
            "enable_witness": enable_witness,
        },
    }
    timing_path.write_text(json.dumps(timing, indent=2))
    print(f"[{seq_name}/{triggers}] done — total={timing['total_s']:.1f}s "
          f"sam={stats['sam_total_s']:.1f}s render={render_s:.1f}s "
          f"peak_vram={timing['peak_vram_mb']:.0f}MB "
          f"windows={stats['n_windows']} swaps={stats['n_swaps']}")
    return True


def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="SAM3-Deep-EIoU framework (Deep-EIoU + selective SAM3 mask propagation). "
                    "Loads SAM3 once and iterates clips. Idempotent."
    )
    parser.add_argument("clips", nargs="+",
                        help="One or more sequence directories (must contain img1/, det/, emb/, seqinfo.ini).")
    parser.add_argument("--triggers", choices=TRIGGER_CHOICES, default="margin_gap_witness",
                        help="Which window-creation triggers to enable (default: full).")
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
                             "Per-clip path is <root>/<parent>/<clip>/sam3_deep_eiou[-<suffix>]/.")
    args = parser.parse_args(args_list)

    todo = []
    skipped = []
    for source_path in args.clips:
        timing_path = _output_dir(args.output_root, source_path, args.triggers) / "timing.json"
        if timing_path.exists() and not args.force:
            skipped.append(source_path)
        else:
            todo.append(source_path)

    for source_path in skipped:
        print(f"[{Path(source_path).name}/{args.triggers}] skip (timing.json exists)")

    if not todo:
        print("Nothing to do.")
        return

    print(f"\nProcessing {len(todo)} clip(s) with triggers='{args.triggers}'.\n")
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
                triggers=args.triggers,
                stop_at=args.stop_at,
                no_gt=args.no_gt,
                no_render=args.no_render,
                force=args.force,
            )
            n_done += 1
        except Exception as e:
            n_failed += 1
            print(f"[{seq_name}/{args.triggers}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\nSummary: {n_done} done, {len(skipped)} skipped, {n_failed} failed; "
          f"model_load_s={model_load_s:.1f} (paid once); triggers={args.triggers}")


if __name__ == "__main__":
    main()
