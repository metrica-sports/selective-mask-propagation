"""Selective vs uniform SAM3 throughput on the bundled benchmark clips.

For each clip in bench/: decodes video.mp4 to frames (once), runs
YOLOX + OSNet (once, cached), runs Deep-EIoU, then measures the SAM
step both ways on the same tracks:

    selective   margin-dispatched windows + merge (this repo's method)
    uniform     SAM3 on every track from inception to death

and reports fps = video frames / SAM wall-clock for each. First run
downloads the SAM 3 checkpoint.

    uv run python scripts/fps_benchmark.py [--render] [--skip-uniform]
                                           [--clips NAME ...] [--stop-at N]
"""

import argparse
import contextlib
import io
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

BENCH_DIR = ROOT / "bench"
CLIPS = [
    ("basketball", "basketball-1"),
    ("basketball", "basketball-2"),
    ("soccer", "soccer-1"),
    ("soccer", "soccer-2"),
]


def _ensure_frames(clip_dir: Path) -> None:
    """Decode video.mp4 to img1/*.jpg and write seqinfo.ini (once)."""
    img_dir = clip_dir / "img1"
    seqinfo = clip_dir / "seqinfo.ini"
    cap = cv2.VideoCapture(str(clip_dir / "video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if seqinfo.exists() and img_dir.exists() and len(list(img_dir.glob("*.jpg"))) == n_frames:
        cap.release()
        return

    img_dir.mkdir(exist_ok=True)
    print(f"  decoding {clip_dir.name}/video.mp4 -> img1/")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), frame)
    cap.release()

    seqinfo.write_text(
        "[Sequence]\n"
        f"name={clip_dir.name}\n"
        "imDir=img1\n"
        f"frameRate={round(fps)}\n"
        f"seqLength={i}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )


def _ensure_detections(clip_dir: Path):
    """Run YOLOX + OSNet once and cache; load from cache afterwards."""
    det_path = clip_dir / "detections.pkl"
    emb_path = clip_dir / "embeddings.pkl"
    if det_path.exists() and emb_path.exists():
        with open(det_path, "rb") as f:
            detections = pickle.load(f)
        with open(emb_path, "rb") as f:
            embeddings = pickle.load(f)
        return detections, embeddings

    from selective_mask_propagation.core.detection import detect_online

    print(f"  running YOLOX + OSNet on {clip_dir.name} (once, cached)")
    detections, embeddings = detect_online(str(clip_dir))
    with open(det_path, "wb") as f:
        pickle.dump(detections, f)
    with open(emb_path, "wb") as f:
        pickle.dump(embeddings, f)
    return detections, embeddings


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def _render(clip_dir: Path, name: str, merged) -> None:
    from selective_mask_propagation.core.render import render_tracker_video

    out_dir = ROOT / "results" / "bench" / clip_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.mp4"
    print(f"  rendering {out_path.relative_to(ROOT)}")
    render_tracker_video(str(clip_dir), str(out_path), merged, no_gt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--render", action="store_true",
                        help="Also render both outputs to results/bench/<clip>/")
    parser.add_argument("--skip-uniform", action="store_true",
                        help="Only measure the selective side")
    parser.add_argument("--clips", nargs="+", default=None, metavar="NAME",
                        help="Subset of bench clips to run (default: all)")
    parser.add_argument("--stop-at", type=int, default=None, metavar="N",
                        help="Truncate each clip to its first N frames")
    args = parser.parse_args()

    clips = CLIPS
    if args.clips:
        unknown = set(args.clips) - {c for _, c in CLIPS}
        if unknown:
            parser.error(f"unknown clips: {sorted(unknown)} "
                         f"(available: {[c for _, c in CLIPS]})")
        clips = [(s, c) for s, c in CLIPS if c in args.clips]

    from selective_mask_propagation.core.merge import (
        TrackData, canon_key, extract_bboxes, step_merge,
    )
    from selective_mask_propagation.core.sam3 import build_predictor, run_sam_uniform, step_sam
    from selective_mask_propagation.deep_eiou.tracker import step_track
    from selective_mask_propagation.utils.helpers import read_sequence_info

    # Decode + detect everything first: building the SAM3 predictor enters a
    # process-wide bf16 autocast (vendor sam3_tracking_predictor), and YOLOX +
    # OSNet must run outside it, exactly as in the main pipeline.
    prepared = {}
    for sport, clip in clips:
        clip_dir = BENCH_DIR / clip
        print(f"\npreparing {clip}")
        _ensure_frames(clip_dir)
        prepared[clip] = _ensure_detections(clip_dir)

    predictor = build_predictor()
    cuda = torch.cuda.is_available()
    rows = []
    for i, (sport, clip) in enumerate(clips, 1):
        clip_dir = BENCH_DIR / clip
        print(f"\n[{i}/{len(clips)}] {sport}  {clip}")
        detections, embeddings = prepared[clip]

        with contextlib.redirect_stdout(io.StringIO()):
            info = read_sequence_info(str(clip_dir))
        frames = info["length"]
        if args.stop_at is not None:
            detections = {f: v for f, v in detections.items() if f < args.stop_at}
            embeddings = {f: v for f, v in embeddings.items() if f < args.stop_at}
            frames = min(frames, args.stop_at)
        with contextlib.redirect_stdout(io.StringIO()):
            tracks, margins = step_track(detections, embeddings,
                                         info["fps"], info["width"], info["height"])
        n_tracks = len({tid for f in tracks.values() for tid in f})

        # Selective: SAM windows + merge, the cost added on top of Deep-EIoU.
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            sam_masks, windows, rename_events, match_history = step_sam(
                predictor, tracks, margins, str(clip_dir))
            merged, _, rename_map = step_merge(tracks, sam_masks, windows, margins, rename_events)
            sde_bboxes = extract_bboxes(merged, tracks, rename_map, match_history, windows)
            sel_s = time.perf_counter() - t0
        sel_vram = torch.cuda.max_memory_allocated() / 1024**3 if cuda else 0.0
        swaps = sum(1 for w in windows if w.outcome.value == "swap")
        print(f"  selective: {frames / sel_s:.1f} fps  ({sel_s:.1f}s, "
              f"{len(windows)} windows, {swaps} swaps, {sel_vram:.1f} GB)")
        if args.render:
            # Flatten to int track IDs aligned with the exported boxes,
            # mirroring the materialization in the main pipeline.
            sel_merged = {}
            for f, ft in sde_bboxes.items():
                frame_data = {}
                merged_frame = merged.get(f, {})
                for tid, bbox in ft.items():
                    entry = merged_frame.get(canon_key(tid))
                    if entry is None:
                        entry = merged_frame.get(tid)
                    mask = entry.mask if entry is not None else None
                    frame_data[tid] = TrackData(bbox=bbox, mask=mask)
                sel_merged[f] = frame_data
            _render(clip_dir, "sam3_deep_eiou", sel_merged)

        # Uniform: SAM3 on every track from inception to death, no windows.
        uni_s, uni_vram, oom = None, None, False
        if not args.skip_uniform:
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    t0 = time.perf_counter()
                    uni_masks = run_sam_uniform(predictor, tracks, str(clip_dir))
                    uni_merged = {}
                    for f, frame_tracks in tracks.items():
                        uni_merged[f] = {}
                        for tid, bbox in frame_tracks.items():
                            mask = uni_masks.get(f, {}).get(tid)
                            if mask is not None and mask.any():
                                uni_merged[f][tid] = TrackData(bbox=_bbox_from_mask(mask), mask=mask)
                            else:
                                uni_merged[f][tid] = TrackData(bbox=bbox, mask=None)
                    uni_s = time.perf_counter() - t0
                uni_vram = torch.cuda.max_memory_allocated() / 1024**3 if cuda else 0.0
                print(f"  uniform:   {frames / uni_s:.1f} fps  ({uni_s:.1f}s, "
                      f"{n_tracks} tracks, {uni_vram:.1f} GB)")
                if args.render:
                    _render(clip_dir, "de_uniform_sam3", uni_merged)
            except torch.OutOfMemoryError:
                oom = True
                torch.cuda.empty_cache()
                print("  uniform:   OUT OF MEMORY")

        rows.append((sport, clip, frames, n_tracks, sel_s, sel_vram, swaps, uni_s, uni_vram, oom))

    line = "=" * 78
    print("\n" + line)
    print("  SAM-step throughput: selective mask propagation vs uniform SAM3")
    print(line)
    print(f"  {'clip':<14}{'frames':>6}{'tracks':>7} | {'selective':>12}{'GB':>5} | {'uniform':>12}{'GB':>5}")
    print("-" * 78)
    for sport, clip, frames, n_tracks, sel_s, sel_vram, swaps, uni_s, uni_vram, oom in rows:
        sel = f"{frames / sel_s:.1f} fps"
        if oom:
            uni, uvr = "OOM", "-"
        elif uni_s is None:
            uni, uvr = "-", "-"
        else:
            uni, uvr = f"{frames / uni_s:.1f} fps", f"{uni_vram:.1f}"
        print(f"  {clip:<14}{frames:>6}{n_tracks:>7} | {sel:>12}{sel_vram:>5.1f} | {uni:>12}{uvr:>5}")
    tot_f = sum(r[2] for r in rows)
    tot_sel = sum(r[4] for r in rows)
    uni_rows = [r for r in rows if r[7] is not None]
    print("-" * 78)
    agg_sel = f"{tot_f / tot_sel:.1f} fps"
    agg = f"  {'aggregate':<14}{tot_f:>6}{'':>7} | {agg_sel:>12}{max(r[5] for r in rows):>5.1f}"
    if uni_rows:
        tot_uni_f = sum(r[2] for r in uni_rows)
        tot_uni = sum(r[7] for r in uni_rows)
        agg_uni = f"{tot_uni_f / tot_uni:.1f} fps"
        agg += f" | {agg_uni:>12}{max(r[8] for r in uni_rows):>5.1f}"
    print(agg)
    print(line)
    print("  fps = video frames / SAM wall-clock (selective includes merge).")
    print("  Same YOLOX detections and Deep-EIoU tracks on both sides; only")
    print("  the dispatch differs.\n")


if __name__ == "__main__":
    main()
