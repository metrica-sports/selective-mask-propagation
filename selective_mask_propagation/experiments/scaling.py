"""§4.7.1 scaling microbenchmark.

For each N in the sweep, this caps each frame's YOLOX detections to the top-N
by score, runs the configured variant for a fixed window of frames, and
records mean per-frame latency and peak GPU memory. Designed to produce the
log-log scaling figure in §4.7.1: latency-vs-N and VRAM-vs-N, one line per
variant.

One process loads SAM3 once, warms CUDA, then sweeps N for the chosen variant
on every clip in the input list. Per-clip artifacts persist incrementally to
``results/<parent>/<clip>/scaling/<variant_filename>.json``. A clip with that
file already present is skipped unless ``--force`` is passed.

CLI:
    uv run python -m selective_mask_propagation.experiments.scaling <clip>... --variant de_uniform_sam3
    uv run python -m selective_mask_propagation.experiments.scaling <clip>... --variant sam3_deep_eiou
    uv run python -m selective_mask_propagation.experiments.scaling <clip>... --variant sam3_deep_eiou --triggers margin_gap

Variants and the file they persist to:
    de_uniform_sam3                                → de_uniform_sam3.json
    sam3_deep_eiou + triggers margin_gap_witness   → sam3_deep_eiou.json (default)
    sam3_deep_eiou + triggers margin_gap           → sam3_deep_eiou-margin_gap.json
    sam3_deep_eiou + triggers margin               → sam3_deep_eiou-margin.json

Output schema (per variant file):
    {
      "variant": "<name>",
      "config":  {<runner-specific knobs>},
      "frames":  <int>,
      "warmup":  <int>,
      "sweep": [
        {"N": <int>, "latency_s_per_frame": <float>, "peak_vram_mb": <float>,
         "frames_measured": <int>, "actual_active_objects_mean": <float>},
        ...
      ]
    }
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..core.detection import detect_precomputed
from ..core.sam3 import build_predictor, run_sam_uniform, step_sam
from ..deep_eiou.tracker import step_track
from ..utils.helpers import read_sequence_info


VARIANT_DE_UNIFORM = "de_uniform_sam3"
VARIANT_FRAMEWORK = "sam3_deep_eiou"
TRIGGER_CHOICES = ["margin", "margin_gap", "margin_gap_witness"]


def _trigger_flags(triggers: str) -> Tuple[bool, bool]:
    """Map --triggers string to (enable_gap, enable_witness)."""
    return ("gap" in triggers, "witness" in triggers)


def _variant_filename(variant: str, triggers: Optional[str]) -> str:
    """Encode (variant, triggers) into a single filename stem.

    The framework's default triggers (full) get the unsuffixed name so the
    canonical run lands in the natural place; non-default triggers get a
    suffix that matches the runner's per-clip output dir convention.
    """
    if variant == VARIANT_DE_UNIFORM:
        return "de_uniform_sam3"
    if variant == VARIANT_FRAMEWORK:
        if triggers == "margin_gap_witness":
            return "sam3_deep_eiou"
        return f"sam3_deep_eiou-{triggers}"
    raise ValueError(f"Unknown variant: {variant}")


def _cap_detections_top_n(
    detections: Dict[int, np.ndarray],
    embeddings: Dict[int, np.ndarray],
    N: int,
    frames: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Keep only the top-N detections by score on each of the first ``frames`` frames.

    Empty frames stay empty. Embeddings are reindexed to match the surviving
    detection rows so DE's ReID step still receives consistent (det, emb) rows.
    """
    capped_dets: Dict[int, np.ndarray] = {}
    capped_embs: Dict[int, np.ndarray] = {}
    for f, dets in detections.items():
        if f >= frames:
            continue
        if len(dets) == 0:
            capped_dets[f] = dets
            capped_embs[f] = embeddings.get(f, np.empty((0, 512)))
            continue
        scores = dets[:, 4]
        order = np.argsort(-scores)[:N]
        capped_dets[f] = dets[order]
        emb = embeddings.get(f)
        capped_embs[f] = emb[order] if emb is not None and len(emb) > 0 else np.empty((0, 512))
    return capped_dets, capped_embs


def _measure_one(
    predictor,
    variant: str,
    triggers: Optional[str],
    de_tracks: Dict[int, Dict[int, np.ndarray]],
    margins: Dict[int, Dict[int, float]],
    source_path: str,
) -> Tuple[float, float]:
    """Run the chosen variant once and return (total_seconds, peak_vram_mb).

    cuda.synchronize() bookends the timed region so we measure GPU work, not
    just queue submissions. Peak VRAM is the maximum allocator usage observed
    during the call.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    if variant == VARIANT_DE_UNIFORM:
        run_sam_uniform(predictor, de_tracks, source_path)
    else:
        enable_gap, enable_witness = _trigger_flags(triggers)
        step_sam(predictor, de_tracks, margins, source_path,
                  enable_gap=enable_gap, enable_witness=enable_witness)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    return elapsed, peak_vram_mb


def _warmup(predictor, detections, embeddings, source_path, seq_info) -> None:
    """One short uniform-SAM3 pass to warm CUDA kernels before the sweep.

    The first SAM3 forward of a process pays compilation, autotune, and kernel
    cache costs that would otherwise inflate the lowest-N point of the sweep.
    A throwaway pass over a few frames at N=2 makes every subsequent
    measurement steady-state without polluting the recorded data.
    """
    warm_dets, warm_embs = _cap_detections_top_n(detections, embeddings, N=2, frames=10)
    if not warm_dets:
        return
    warm_tracks, _ = step_track(
        warm_dets, warm_embs,
        seq_info["fps"], seq_info["width"], seq_info["height"],
    )
    if not warm_tracks:
        return
    run_sam_uniform(predictor, warm_tracks, source_path)


def run_sweep_on_clip(
    predictor,
    source_path: str,
    variant: str,
    triggers: Optional[str],
    N_values: List[int],
    frames: int,
    warmup: int,
) -> dict:
    """Sweep N for a single clip, return the schema for one scaling JSON file."""
    seq_info = read_sequence_info(source_path)
    detections, embeddings = detect_precomputed(source_path)

    print("  Warming up CUDA...")
    _warmup(predictor, detections, embeddings, source_path, seq_info)

    sweep_results = []
    for N in N_values:
        capped_dets, capped_embs = _cap_detections_top_n(detections, embeddings, N=N, frames=frames)
        if not capped_dets:
            print(f"  N={N}: no detections, skipping")
            continue

        de_tracks, margins = step_track(
            capped_dets, capped_embs,
            seq_info["fps"], seq_info["width"], seq_info["height"],
        )
        if not de_tracks:
            print(f"  N={N}: DE produced 0 tracks, skipping")
            continue

        n_frames_measured = len(de_tracks)
        active_counts = [len(de_tracks[f]) for f in de_tracks]
        actual_active_mean = sum(active_counts) / len(active_counts)

        elapsed, peak_vram_mb = _measure_one(
            predictor, variant, triggers, de_tracks, margins, source_path,
        )
        latency = elapsed / n_frames_measured

        sweep_results.append({
            "N": N,
            "latency_s_per_frame": latency,
            "peak_vram_mb": peak_vram_mb,
            "frames_measured": n_frames_measured,
            "actual_active_objects_mean": actual_active_mean,
        })
        print(f"  N={N}: {latency*1000:.1f} ms/frame  vram={peak_vram_mb:.0f} MB  "
              f"active={actual_active_mean:.1f}  frames={n_frames_measured}")

    config = {"variant": variant}
    if variant == VARIANT_FRAMEWORK:
        config["triggers"] = triggers
        config["enable_gap"], config["enable_witness"] = _trigger_flags(triggers)

    return {
        "variant": _variant_filename(variant, triggers),
        "config": config,
        "frames": frames,
        "warmup": warmup,
        "sweep": sweep_results,
    }


def _output_path(output_root: Path, source_path: str, variant: str, triggers: Optional[str]) -> Path:
    clip = Path(source_path).name
    parent = Path(source_path).parent.name
    out_dir = output_root / parent / clip / "scaling"
    return out_dir / f"{_variant_filename(variant, triggers)}.json"


def _process_clip(
    predictor,
    output_root: Path,
    source_path: str,
    variant: str,
    triggers: Optional[str],
    N_values: List[int],
    frames: int,
    warmup: int,
    force: bool,
) -> bool:
    seq_name = Path(source_path).name
    out_path = _output_path(output_root, source_path, variant, triggers)
    label = _variant_filename(variant, triggers)

    if out_path.exists() and not force:
        print(f"[{seq_name}/{label}] skip ({out_path.name} exists)")
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[{seq_name}/{label}] sweep starting")
    record = run_sweep_on_clip(
        predictor, source_path, variant, triggers, N_values, frames, warmup,
    )
    out_path.write_text(json.dumps(record, indent=2))
    print(f"[{seq_name}/{label}] done — {len(record['sweep'])} N points → {out_path}")
    return True


def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="§4.7.1 scaling microbenchmark for de_uniform_sam3 / sam3_deep_eiou."
    )
    parser.add_argument("clips", nargs="+", help="One or more sequence directories.")
    parser.add_argument("--variant", required=True,
                        choices=[VARIANT_DE_UNIFORM, VARIANT_FRAMEWORK],
                        help="Which scaling line to measure.")
    parser.add_argument("--triggers", choices=TRIGGER_CHOICES, default="margin_gap_witness",
                        help="Trigger config for the framework variant (ignored for de_uniform_sam3).")
    parser.add_argument("--N", default="1,2,4,8,16,32",
                        help="Comma-separated list of N values to sweep.")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames to propagate per N (default 100).")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Frames at the start of each measurement that are conceptually 'warmup'. "
                             "We warm CUDA once before the sweep; this number is recorded for transparency.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run clips even if the variant's scaling JSON already exists.")
    parser.add_argument("--output-root", type=Path, default=Path("results"),
                        help="Root dir for per-clip output (default: results/). "
                             "Per-clip path is <root>/<parent>/<clip>/scaling/<variant>.json.")
    args = parser.parse_args(args_list)

    N_values = sorted({int(s) for s in args.N.split(",") if s.strip()})

    triggers = args.triggers if args.variant == VARIANT_FRAMEWORK else None

    todo = []
    skipped = []
    for source_path in args.clips:
        out_path = _output_path(args.output_root, source_path, args.variant, triggers)
        if out_path.exists() and not args.force:
            skipped.append(source_path)
        else:
            todo.append(source_path)

    label = _variant_filename(args.variant, triggers)
    for source_path in skipped:
        print(f"[{Path(source_path).name}/{label}] skip (scaling json exists)")

    if not todo:
        print("Nothing to do.")
        return

    print(f"\nVariant: {label}; N sweep: {N_values}; frames: {args.frames}")
    print(f"Processing {len(todo)} clip(s).\n")
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
                variant=args.variant, triggers=triggers,
                N_values=N_values, frames=args.frames, warmup=args.warmup,
                force=args.force,
            )
            n_done += 1
        except Exception as e:
            n_failed += 1
            print(f"[{seq_name}/{label}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\nSummary: {n_done} done, {len(skipped)} skipped, {n_failed} failed; "
          f"model_load_s={model_load_s:.1f} (paid once); variant={label}")


if __name__ == "__main__":
    main()
