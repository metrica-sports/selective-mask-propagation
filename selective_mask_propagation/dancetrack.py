"""DanceTrack pipeline: detect -> track -> sam -> merge -> eval -> render

Runs YOLOX (DanceTrack checkpoint) + OSNet for detections and embeddings,
then a base tracker (Deep-EIoU, ByteTrack, or SORT), then selective mask propagation.

CLI:
    uv run python -m selective_mask_propagation.dancetrack --input "data/dancetrack/val/dancetrack0007"
    uv run python -m selective_mask_propagation.dancetrack --input "data/dancetrack/val/dancetrack0007" --tracker bytetrack
    uv run python -m selective_mask_propagation.dancetrack --input "data/dancetrack/val/dancetrack0007" --step sam -c --sam3
    uv run python -m selective_mask_propagation.dancetrack --input "data/dancetrack/val/dancetrack00*" --step eval

Results layout:
    results/val/{seq}/artifacts/                        # shared detect artifacts
    results/val/{seq}/{tracker}-sam2/artifacts/          # tracker-specific
    results/val/{seq}/{tracker}-sam3/artifacts/

Steps: detect, track, sam, merge, eval, render
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils.artifacts import save_pickle as _save, load_pickle as _load
from .utils.export import export_mot
from .utils.helpers import read_sequence_info as _read_seq_info

STEPS = ["detect", "track", "sam", "merge", "eval", "render"]
TRACKERS = ["deepeiou", "bytetrack", "sort"]

STEP_ARTIFACTS = {
    "detect": ["detections.pkl", "embeddings.pkl"],
    "track": ["tracks.pkl", "margins.pkl", "mot_baseline.txt"],
    "sam": ["sam_masks.pkl", "windows.pkl", "rename_events.pkl", "match_history.pkl"],
    "merge": ["merged.pkl", "renamed_margins.pkl", "mot_sam.txt"],
    "eval": ["eval.md", "eval.json"],
}

YOLOX_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "yolox", "checkpoints", "yolox_x_dancetrack.pth.tar",
)


def _seq_dir(source_path: str) -> Path:
    """results/{parent_name}/{seq_name}/"""
    source = Path(source_path)
    d = Path("results") / source.parent.name / source.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _detect_dir(source_path: str) -> Path:
    """results/{parent_name}/{seq_name}/artifacts/ — shared detect artifacts."""
    d = _seq_dir(source_path) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _experiment_name(tracker: str, sam3: bool, margin_entry: float = None,
                     mode: str = "benchmark") -> str:
    sam_label = "sam3" if sam3 else "sam2"
    name = f"{tracker}-{sam_label}"
    if margin_entry is not None:
        name += f"-m{margin_entry}"
    if mode != "benchmark":
        name += f"-{mode}"
    return name


def _experiment_dir(source_path: str, tracker: str, sam3: bool, margin_entry: float = None,
                    mode: str = "benchmark") -> Path:
    d = _seq_dir(source_path) / _experiment_name(tracker, sam3, margin_entry, mode)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _experiment_artifacts_dir(source_path: str, tracker: str, sam3: bool, margin_entry: float = None,
                              mode: str = "benchmark") -> Path:
    d = _experiment_dir(source_path, tracker, sam3, margin_entry, mode) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _step_detect(
    source_path: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Run YOLOX (DanceTrack checkpoint) + OSNet on a sequence."""
    from .yolox.inference import build_model as build_yolox, detect_sequence
    from .osnet.inference import build_model as build_osnet, extract_embeddings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolox = build_yolox(device, checkpoint=YOLOX_CHECKPOINT)
    detections = detect_sequence(yolox, source_path, device)
    del yolox

    osnet = build_osnet(device)
    embeddings = extract_embeddings(osnet, source_path, detections, device)
    del osnet

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return detections, embeddings


def _step_track(
    tracker_name: str,
    detections: Dict[int, np.ndarray],
    embeddings: Dict[int, np.ndarray],
    fps: int,
    frame_width: int,
    frame_height: int,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Dict[int, float]]]:
    """Run a base tracker. Returns (tracks, margins)."""
    total_frames = max(detections.keys()) + 1
    tracks: Dict[int, Dict[int, np.ndarray]] = {}
    margins: Dict[int, Dict[int, float]] = {}

    if tracker_name == "deepeiou":
        from .deep_eiou.tracker import Deep_EIoU
        tracker = Deep_EIoU(frame_rate=fps, frame_width=frame_width,
                            frame_height=frame_height, with_reid=True)
        for frame_idx in range(total_frames):
            dets = detections.get(frame_idx)
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 5))
                emb = np.empty((0, 512))
            else:
                emb = embeddings.get(frame_idx, np.empty((0, 512)))
            online = tracker.update(dets, emb)
            tracks[frame_idx] = {t.track_id: t.last_tlbr for t in online}
            margins[frame_idx] = {t.track_id: t.margin for t in online}

    elif tracker_name == "bytetrack":
        from .bytetrack.tracker import BYTETracker
        from .deep_eiou.basetrack import BaseTrack
        BaseTrack._count = 0
        tracker = BYTETracker(frame_rate=fps, frame_width=frame_width,
                              frame_height=frame_height)
        for frame_idx in range(total_frames):
            dets = detections.get(frame_idx)
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 5))
            online = tracker.update(dets)
            tracks[frame_idx] = {t.track_id: t.last_tlbr for t in online}
            margins[frame_idx] = {t.track_id: t.margin for t in online}

    elif tracker_name == "sort":
        from .sort.tracker import Sort, KalmanBoxTracker
        KalmanBoxTracker.count = 0
        tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3,
                       frame_width=frame_width, frame_height=frame_height)
        for frame_idx in range(total_frames):
            dets = detections.get(frame_idx)
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 5))
            online = tracker.update(dets)
            tracks[frame_idx] = {t.track_id: t.last_tlbr for t in online}
            margins[frame_idx] = {t.track_id: t.margin for t in online}

    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")

    n_tracked = sum(len(v) for v in tracks.values())
    print(f"Tracker ({tracker_name}): {n_tracked} tracks across {total_frames} frames")
    return tracks, margins


def _step_eval(source_path: str, output_dir: str, artifacts_dir: Path,
               total_frames: int = None) -> None:
    """Evaluate baseline and SAM-augmented MOT files against ground truth."""
    from .core.eval import _eval_mot

    source = Path(source_path)
    if not (source / "gt" / "gt.txt").exists():
        print("Eval: skipping (no ground truth).")
        return

    seq_name = source.name

    variants = [
        ("Baseline", "mot_baseline.txt", "baseline"),
        ("SAM", "mot_sam.txt", "sam"),
    ]

    columns = []
    eval_json = {"seq_name": seq_name}

    for label, filename, json_key in variants:
        mot_path = artifacts_dir / filename
        if not mot_path.exists():
            continue
        r, _ = _eval_mot(seq_name, source_path, str(mot_path))
        columns.append((label, r))
        eval_json[json_key] = r

    if not columns:
        print("No MOT files found to evaluate.")
        return

    header = "| Metric | " + " | ".join(label for label, _ in columns) + " |"
    sep = "|--------|" + "|".join("-------" for _ in columns) + "|"
    rows = [f"# Eval: {seq_name}\n", header, sep]
    for key, name in [
        ("hota", "HOTA"), ("deta", "DetA"), ("assa", "AssA"),
        ("mota", "MOTA"), ("idf1", "IDF1"), ("idsw", "IDSW"),
        ("gt_ids", "GT IDs"), ("pred_ids", "Pred IDs"),
    ]:
        cells = " | ".join(
            str(r[key]) if isinstance(r[key], int) else f"{r[key]:.1f}"
            for _, r in columns
        )
        rows.append(f"| {name} | {cells} |")

    md = "\n".join(rows) + "\n"
    eval_path = Path(output_dir) / "eval.md"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(md)

    timing_path = artifacts_dir / "timing.json"
    if timing_path.exists():
        timings = json.loads(timing_path.read_text())
        for k, v in timings.items():
            eval_json[f"{k}_wall_clock_s"] = v
    if total_frames is not None:
        eval_json["total_frames"] = total_frames
    windows_path = artifacts_dir / "windows.pkl"
    if windows_path.exists():
        with open(windows_path, "rb") as f:
            ws = pickle.load(f)
        sam_frames = set()
        for w in ws:
            for fi in range(w.seed_frame, w.exit_frame + 1):
                sam_frames.add(fi)
        eval_json["sam_active_frames"] = len(sam_frames)
        if total_frames:
            eval_json["sam_active_pct"] = round(100 * len(sam_frames) / total_frames, 1)
        from collections import Counter
        eval_json["window_outcomes"] = dict(Counter(w.outcome.value for w in ws))

    json_path = Path(output_dir) / "eval.json"
    json_path.write_text(json.dumps(eval_json, indent=2) + "\n")
    print(md)


def _artifact_exists(directory: Path, filename: str) -> bool:
    if (directory / filename).exists():
        return True
    if filename.endswith(".pkl") and (directory / (filename + ".zst")).exists():
        return True
    return False


def _step_done(step: str, detect_dir: Path, exp_artifacts_dir: Path) -> bool:
    files = STEP_ARTIFACTS.get(step, [])
    d = detect_dir if step == "detect" else exp_artifacts_dir
    return all(_artifact_exists(d, f) for f in files)


def _clean_steps(steps: list, detect_dir: Path, exp_artifacts_dir: Path) -> None:
    cleaned = []
    for step in steps:
        d = detect_dir if step == "detect" else exp_artifacts_dir
        for filename in STEP_ARTIFACTS.get(step, []):
            for path in [d / filename, d / (filename + ".zst")]:
                if path.exists():
                    path.unlink()
                    cleaned.append(path.name)
    if cleaned:
        print(f"Cleaned: {', '.join(cleaned)}")


def run_pipeline(
    source_path: str,
    start_step: str = "detect",
    continue_to_end: bool = True,
    dev: bool = False,
    sam3: bool = False,
    skip_existing: bool = False,
    tracker: str = "deepeiou",
    margin_entry: float = None,
    precomputed: bool = False,
    mode: str = "benchmark",
):
    seq_info = _read_seq_info(source_path)
    detect_d = _detect_dir(source_path)
    exp_dir = _experiment_dir(source_path, tracker, sam3, margin_entry, mode)
    exp_art = _experiment_artifacts_dir(source_path, tracker, sam3, margin_entry, mode)

    start_idx = STEPS.index(start_step)
    steps = STEPS[start_idx:] if continue_to_end else [start_step]
    if skip_existing:
        skipped = [s for s in steps if _step_done(s, detect_d, exp_art)]
        steps = [s for s in steps if not _step_done(s, detect_d, exp_art)]
        if skipped:
            print(f"Skipping (artifacts exist): {', '.join(skipped)}")
    _clean_steps(steps, detect_d, exp_art)

    import time

    def _save_timing(step_name: str, seconds: float):
        timing_path = exp_art / "timing.json"
        timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        timings[step_name] = round(seconds, 1)
        timing_path.write_text(json.dumps(timings, indent=2) + "\n")

    def _load_timings() -> dict:
        timing_path = exp_art / "timing.json"
        return json.loads(timing_path.read_text()) if timing_path.exists() else {}

    detections = embeddings = tracks = margins = None
    sam_masks = windows = rename_events = match_history = None
    merged = renamed_margins = rename_map = None

    if "detect" in steps:
        print(f"\n{'='*60}\nStep: detect\n{'='*60}")
        t0 = time.time()
        if precomputed:
            from .core.detection import detect_precomputed
            detections, embeddings = detect_precomputed(source_path)
        else:
            detections, embeddings = _step_detect(source_path)
        elapsed = time.time() - t0
        print(f"Detect wall clock: {elapsed:.1f}s")
        _save_timing("detect", elapsed)
        _save("detections", detections, detect_d)
        _save("embeddings", embeddings, detect_d)

    if "track" in steps:
        print(f"\n{'='*60}\nStep: track ({tracker})\n{'='*60}")
        if detections is None:
            detections = _load("detections", detect_d)
            embeddings = _load("embeddings", detect_d)
        t0 = time.time()
        tracks, margins = _step_track(
            tracker, detections, embeddings,
            seq_info["fps"], seq_info["width"], seq_info["height"],
        )
        elapsed = time.time() - t0
        print(f"Track wall clock: {elapsed:.1f}s")
        _save_timing("track", elapsed)
        _save("tracks", tracks, exp_art)
        _save("margins", margins, exp_art)
        export_mot(tracks, str(exp_art / "mot_baseline.txt"))

    if "sam" in steps:
        print(f"\n{'='*60}\nStep: sam\n{'='*60}")
        if sam3:
            from .core.sam3 import build_predictor, step_sam
        else:
            from .core.sam2 import build_predictor, step_sam
        if tracks is None:
            tracks = _load("tracks", exp_art)
        if margins is None:
            margins = _load("margins", exp_art)
        predictor = build_predictor()
        t0 = time.time()
        sam_kwargs = {}
        if margin_entry is not None:
            sam_kwargs["margin_entry"] = margin_entry
        sam_masks, windows, rename_events, match_history = step_sam(predictor, tracks, margins, source_path, **sam_kwargs)
        elapsed = time.time() - t0
        print(f"SAM wall clock: {elapsed:.1f}s")
        _save_timing("sam", elapsed)
        _save("sam_masks", sam_masks, exp_art)
        _save("windows", windows, exp_art)
        _save("rename_events", rename_events, exp_art)
        _save("match_history", match_history, exp_art)

    if "merge" in steps:
        print(f"\n{'='*60}\nStep: merge\n{'='*60}")
        from .core.merge import (TrackData, canon_key, derive_bboxes,
                                 extract_bboxes, step_merge, step_merge_prod)
        if tracks is None:
            tracks = _load("tracks", exp_art)
        if sam_masks is None:
            sam_masks = _load("sam_masks", exp_art)
        if windows is None:
            windows = _load("windows", exp_art)
        if margins is None:
            margins = _load("margins", exp_art)
        if rename_events is None:
            rename_events = _load("rename_events", exp_art)
        if match_history is None:
            match_history = _load("match_history", exp_art, allow_missing=True) or {}

        if mode == "prod":
            merged, renamed_margins, rename_map = step_merge_prod(
                tracks, sam_masks, windows, margins, rename_events)
            sam_bboxes = derive_bboxes(merged)
            export_mot(sam_bboxes, str(exp_art / "mot_sam.txt"))
        else:
            merged, renamed_margins, rename_map = step_merge(tracks, sam_masks, windows, margins, rename_events)

            sam_bboxes = extract_bboxes(merged, tracks, rename_map, match_history, windows)
            export_mot(sam_bboxes, str(exp_art / "mot_sam.txt"))

            materialized = {}
            for f, ft in sam_bboxes.items():
                frame_data = {}
                merged_frame = merged.get(f, {})
                for tid, bbox in ft.items():
                    entry = merged_frame.get(canon_key(tid))
                    if entry is None:
                        entry = merged_frame.get(tid)
                    mask = entry.mask if entry is not None else None
                    frame_data[tid] = TrackData(bbox=bbox, mask=mask)
                materialized[f] = frame_data
            merged = materialized

        _save("renamed_margins", renamed_margins, exp_art)
        _save("merged", merged, exp_art)

    if "eval" in steps:
        print(f"\n{'='*60}\nStep: eval\n{'='*60}")
        _step_eval(source_path, str(exp_dir), exp_art,
                   total_frames=seq_info["length"])

    if "render" in steps:
        print(f"\n{'='*60}\nStep: render\n{'='*60}")
        from .core.render import render_tracker_video, _tracks_to_merged
        if tracks is None:
            tracks = _load("tracks", exp_art)
        if margins is None:
            margins = _load("margins", exp_art)
        if merged is None:
            merged = _load("merged", exp_art)
        if renamed_margins is None:
            renamed_margins = _load("renamed_margins", exp_art)

        render_tracker_video(
            source_path, str(exp_dir / "baseline.mp4"),
            _tracks_to_merged(tracks), margins=margins,
        )
        render_tracker_video(
            source_path, str(exp_dir / "sam.mp4"),
            merged, margins=renamed_margins,
        )

    print("\nDone.")


def main(args_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="DanceTrack pipeline")
    parser.add_argument("--input", required=True, nargs="+", help="Sequence directories or glob pattern.")
    parser.add_argument("--step", choices=STEPS, help="Run a specific step.")
    parser.add_argument("-c", "--continue", dest="continue_pipeline",
                        action="store_true", help="Continue from --step to end.")
    parser.add_argument("-d", "--dev", action="store_true", help="Save debug outputs.")
    parser.add_argument("--tracker", choices=TRACKERS, default="deepeiou", help="Base tracker.")
    parser.add_argument("--sam3", action="store_true", help="Use SAM3 instead of SAM2.")
    parser.add_argument("--mode", choices=["benchmark", "prod"], default="benchmark",
                        help="Output mode from the merge step onward. benchmark (default): "
                             "identity from SAM, bbox geometry from the base tracker. "
                             "prod: SAM masks are authoritative during healthy windows. "
                             "Outputs to separate results dir.")
    parser.add_argument("--margin-entry", type=float, default=None, help="Margin entry threshold (default: 0.05).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps whose artifacts already exist.")
    parser.add_argument("--precomputed", action="store_true",
                        help="Use precomputed det.txt + emb.npy (skip YOLOX+OSNet).")
    args = parser.parse_args(args_list)

    from .utils.helpers import expand_input
    source_paths = []
    for inp in args.input:
        source_paths.extend(expand_input(inp))

    for source_path in source_paths:
        if args.step:
            run_pipeline(source_path, args.step, args.continue_pipeline,
                         dev=args.dev, sam3=args.sam3, skip_existing=args.skip_existing,
                         tracker=args.tracker, margin_entry=args.margin_entry,
                         precomputed=args.precomputed, mode=args.mode)
        else:
            run_pipeline(source_path, "detect", continue_to_end=True,
                         dev=args.dev, sam3=args.sam3, skip_existing=args.skip_existing,
                         tracker=args.tracker, margin_entry=args.margin_entry,
                         precomputed=args.precomputed, mode=args.mode)


if __name__ == "__main__":
    main()
