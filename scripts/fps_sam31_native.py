"""Out-of-the-box SAM 3.1 (Object Multiplex) dense tracking on the bench clips.

SAM 3.1's native pipeline tracks every player jointly: its own detector
finds them from a "player" text prompt, and Object Multiplex propagates
all of them in shared-memory buckets. Unlike ``fps_benchmark.py``'s
selective/uniform rows (which share YOLOX detections and Deep-EIoU tracks,
isolating dispatch), this measures the full out-of-the-box system —
detection included, since it is inseparable from propagation.

The SAM 3.1 code lives in ``vendor/sam3-with-3.1``. The paper results run
on the frozen ``vendor/sam3`` snapshot, which SAM 3.1 modifies (see README);
keeping both pinned keeps every reported number reproducible. Both packages
import as ``sam3``, so this script shadows the installed one via sys.path
for its own process only.

    uv run python scripts/fps_sam31_native.py [--render] [--clips NAME ...]
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "sam3-with-3.1"))

import cv2
import numpy as np
import torch

import sam3

assert "sam3-with-3.1" in str(Path(sam3.__file__)), (
    f"expected the vendored SAM 3.1 package, got {sam3.__file__}"
)

BENCH_DIR = ROOT / "bench"
CLIPS = [
    ("basketball", "basketball-1"),
    ("basketball", "basketball-2"),
    ("soccer", "soccer-1"),
    ("soccer", "soccer-2"),
]

# BGR palette for render overlays (matches no specific team, just distinct).
COLORS = [
    (60, 76, 231), (113, 204, 46), (219, 152, 52), (34, 126, 230),
    (156, 89, 182), (47, 168, 241), (133, 160, 22), (182, 89, 155),
    (18, 156, 243), (94, 73, 52), (185, 128, 41), (96, 174, 39),
]


def _shim_init_state(predictor):
    """Upstream bug: Sam3BasePredictor.start_session always passes
    offload_state_to_cpu, which the multiplex init_state doesn't accept."""
    orig = predictor.model.init_state

    def init_state(resource_path, offload_video_to_cpu=False,
                   offload_state_to_cpu=False, **kw):
        return orig(resource_path=resource_path,
                    offload_video_to_cpu=offload_video_to_cpu, **kw)

    predictor.model.init_state = init_state


def _render(video_path: Path, out_path: Path, frames_masks) -> None:
    """Overlay per-object masks on the source video (H.264 via ffmpeg,
    same as core/render.py — cv2's mp4v output isn't playable everywhere)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f"temp_{out_path.name}")
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for obj_id, packed, shape in frames_masks.get(frame_idx, []):
            mask = np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape).astype(bool)
            color = COLORS[obj_id % len(COLORS)]
            overlay = frame.copy()
            overlay[mask] = color
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)
            ys, xs = np.where(mask)
            if len(xs):
                cv2.putText(frame, f"T{obj_id}", (int(xs.min()), max(0, int(ys.min()) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        temp_path.rename(out_path)  # mp4v fallback; some players reject it
    else:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(temp_path),
                        "-vcodec", "libx264", "-crf", "28", str(out_path)], check=True)
        temp_path.unlink()
    print(f"  rendered {out_path.relative_to(ROOT)}")


def _run_clip(predictor, video: Path, clip: str, render: bool):
    """One clip through SAM 3.1's native flow, exactly as in the upstream
    example notebook: start_session, add_prompt("player"), propagate."""
    resp = predictor.handle_request(dict(
        type="start_session",
        resource_path=str(video),
        offload_video_to_cpu=True,
    ))
    session_id = resp["session_id"]
    # The interactivity layer keeps every frame's masks on the GPU so that
    # point refinements can rebuild past outputs. We never refine, and over
    # a 600-frame clip this cache alone overruns a 32 GB card — prune it to
    # a recent window as we stream.
    inference_state = predictor._all_inference_states[session_id]["state"]

    frames_masks = {}
    max_objs = 0
    n_frames = 0
    try:
        t0 = time.perf_counter()
        predictor.handle_request(dict(
            type="add_prompt", session_id=session_id, frame_index=0, text="player",
        ))
        for response in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=session_id,
            propagation_direction="forward", start_frame_index=0,
        )):
            n_frames += 1
            outs = response["outputs"]
            max_objs = max(max_objs, len(outs["out_obj_ids"]))
            if render:
                frames_masks[response["frame_index"]] = [
                    (int(oid), np.packbits(m), m.shape)
                    for oid, m in zip(outs["out_obj_ids"], outs["out_binary_masks"])
                ]
            cache = inference_state.get("cached_frame_outputs", {})
            cache.pop(response["frame_index"] - 10, None)
        dt = time.perf_counter() - t0
        vram = torch.cuda.max_memory_allocated() / 1024**3
    finally:
        # Always release the session — a leaked one keeps its GPU state and
        # starves every clip after it.
        predictor.handle_request(dict(type="close_session", session_id=session_id))

    print(f"  sam3.1: {n_frames / dt:.1f} fps  ({dt:.1f}s, "
          f"{n_frames} frames, up to {max_objs} objects, {vram:.1f} GB)")
    if render:
        _render(video, ROOT / "results" / "bench" / clip / "sam31_native.mp4",
                frames_masks)
    return n_frames, max_objs, dt, vram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--render", action="store_true",
                        help="Render mask overlays to results/bench/<clip>/")
    parser.add_argument("--clips", nargs="+", default=None, metavar="NAME",
                        help="Subset of bench clips to run (default: all)")
    args = parser.parse_args()

    clips = CLIPS
    if args.clips:
        unknown = set(args.clips) - {c for _, c in CLIPS}
        if unknown:
            parser.error(f"unknown clips: {sorted(unknown)} "
                         f"(available: {[c for _, c in CLIPS]})")
        clips = [(s, c) for s, c in CLIPS if c in args.clips]

    from sam3.model_builder import build_sam3_multiplex_video_predictor

    # No FlashAttention 3 wheel for Blackwell (sm_120); the math path is used.
    # max_num_objects raised from the default 16 so soccer (20+ people on
    # screen) is fully covered rather than silently capped at one bucket.
    predictor = build_sam3_multiplex_video_predictor(use_fa3=False, max_num_objects=32)
    _shim_init_state(predictor)

    # Two memory adjustments for a 32 GB card (the upstream example targets
    # 80 GB H100s). Batched grounding runs the detector on 16 frames at once,
    # which dominates peak VRAM — halve it. And offload each frame's tracker
    # outputs to CPU once computed (upstream's own eval flag for long videos,
    # see sam3_tracker_base.py).
    predictor.model.batched_grounding_batch_size = 8
    predictor.model.tracker.model.offload_output_to_cpu_for_eval = True

    rows = []
    for i, (sport, clip) in enumerate(clips, 1):
        video = BENCH_DIR / clip / "video.mp4"
        print(f"\n[{i}/{len(clips)}] {sport}  {clip}")
        torch.cuda.reset_peak_memory_stats()
        try:
            rows.append((sport, clip) + _run_clip(predictor, video, clip, args.render))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            vram = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  sam3.1: OUT OF MEMORY (peak {vram:.1f} GB)")
            rows.append((sport, clip, None, None, None, vram))

    line = "=" * 62
    print("\n" + line)
    print("  Out-of-the-box SAM 3.1 (multiplex, own detection)")
    print(line)
    print(f"  {'clip':<14}{'frames':>6}{'objects':>8} | {'fps':>8}{'GB':>6}")
    print("-" * 62)
    for sport, clip, n_frames, max_objs, dt, vram in rows:
        if n_frames is None:
            print(f"  {clip:<14}{'-':>6}{'-':>8} | {'OOM':>8}{vram:>6.1f}")
        else:
            print(f"  {clip:<14}{n_frames:>6}{max_objs:>8} | {n_frames / dt:>8.1f}{vram:>6.1f}")
    done = [r for r in rows if r[2] is not None]
    print("-" * 62)
    if done:
        tot_f = sum(r[2] for r in done)
        tot_t = sum(r[4] for r in done)
        print(f"  {'aggregate':<14}{tot_f:>6}{'':>8} | {tot_f / tot_t:>8.1f}{max(r[5] for r in done):>6.1f}")
    print(line)
    print("  fps = video frames / wall-clock(prompt + propagation).")
    print("  Detection is built in and inseparable, so it is included;")
    print("  fps_benchmark.py's rows exclude detection (~0.001 s/frame).\n")


if __name__ == "__main__":
    main()
