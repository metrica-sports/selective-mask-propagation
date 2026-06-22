"""Show selective-mask-propagation SAM-step throughput on the bundled test clips.

Decodes each test clip's video.mp4 to frames (once), loads its committed
Deep-EIoU tracks/margins, runs the SAM step, and reports fps = frames / SAM
wall-clock. No detection, no downloads.

    uv run python scripts/show_fps.py
"""

import contextlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch

from selective_mask_propagation.core.sam3 import build_predictor, step_sam
from selective_mask_propagation.utils.artifacts import load_pickle
from selective_mask_propagation.utils.helpers import read_sequence_info

TEST_DIR = ROOT / "test"
CLIPS = [
    ("basketball", "v_00HRwkvvjtQ_c005"),
    ("football", "v_i2_L4qquVg0_c009"),
    ("volleyball", "v_0kUtTtmLaJA_c004"),
]


def _ensure_frames(clip_dir: Path, n_frames: int) -> None:
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


def main() -> None:
    predictor = build_predictor()
    rows = []
    for i, (sport, clip) in enumerate(CLIPS, 1):
        clip_dir = TEST_DIR / clip
        with contextlib.redirect_stdout(io.StringIO()):
            frames = read_sequence_info(str(clip_dir))["length"]
        print(f"\n[{i}/{len(CLIPS)}] {sport}  {clip}  ({frames} frames)")
        _ensure_frames(clip_dir, frames)
        with contextlib.redirect_stdout(io.StringIO()):
            tracks = load_pickle("tracks", clip_dir)
            margins = load_pickle("margins", clip_dir)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            _, windows, _, _ = step_sam(predictor, tracks, margins, str(clip_dir))
            dt = time.perf_counter() - t0
        vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        swaps = sum(1 for w in windows if w.outcome.value == "swap")
        rows.append((sport, clip, frames, dt, frames / dt, swaps, vram))

    tot_f = sum(r[2] for r in rows)
    tot_t = sum(r[3] for r in rows)
    max_vram = max(r[6] for r in rows)
    line = "=" * 64
    print("\n" + line)
    print("  selective mask propagation: SAM-step throughput")
    print(line)
    for sport, clip, frames, dt, fps, swaps, _ in rows:
        print(f"  {sport:<11}{clip:<21}{frames:>4} frames{dt:>8.1f}s{fps:>7.1f} fps{swaps:>3} swap")
    print("-" * 64)
    print(f"  {'aggregate':<32}{tot_f:>4} frames{tot_t:>8.1f}s{tot_f / tot_t:>7.1f} fps   {max_vram / 1024:.1f} GB peak")
    print(line)
    print("  fps = frames / SAM wall-clock, over all frames\n")


if __name__ == "__main__":
    main()
