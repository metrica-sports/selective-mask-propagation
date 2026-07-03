"""Augment a base tracker's output with selective mask propagation.

Given tracks and assignment margins from any Hungarian-based tracker,
detects ambiguous windows, propagates SAM masks through them, and
corrects identity switches. Only modifies the output when positive
evidence of a swap is found.

API:
    from selective_mask_propagation.augment import augment
    corrected_tracks = augment(tracks, margins, "path/to/sequence")

Test:
    uv run python -m selective_mask_propagation.augment data/sportsmot/dataset/val/v_00HRwkvvjtQ_c005 --sam3
    uv run python -m selective_mask_propagation.augment data/dancetrack/val/dancetrack0007 --sam3
"""

from typing import Dict

import numpy as np


def augment(
    tracks: Dict[int, Dict[int, np.ndarray]],
    margins: Dict[int, Dict[int, float]],
    source_path: str,
    sam3: bool = False,
    mode: str = "benchmark",
) -> Dict[int, Dict[int, np.ndarray]]:
    """Augment tracker output with selective SAM mask propagation.

    Args:
        tracks: {frame: {track_id: bbox}} where bbox is [x1, y1, x2, y2].
        margins: {frame: {track_id: margin}} where margin is the difference
            between the second-best and best assignment cost in the cost matrix.
        source_path: Path to sequence directory containing img1/.
        sam3: Use SAM3 instead of SAM2.
        mode: "benchmark" (default) keeps the base tracker's bbox geometry and
            uses SAM only to correct identities — what benchmark evaluators
            reward. "prod" makes SAM masks authoritative during healthy
            windows, so boxes are mask-derived through occlusions — better
            per-frame identity and position, lower benchmark score.

    Returns:
        Corrected tracks in the same format as the input.
    """
    if mode not in ("benchmark", "prod"):
        raise ValueError(f"mode must be 'benchmark' or 'prod', got {mode!r}")
    if sam3:
        from .core.sam3 import build_predictor, step_sam
    else:
        from .core.sam2 import build_predictor, step_sam

    from .core.merge import derive_bboxes, extract_bboxes, step_merge, step_merge_prod

    predictor = build_predictor()
    sam_masks, windows, rename_events, match_history = step_sam(predictor, tracks, margins, source_path)
    if mode == "prod":
        merged, _, _ = step_merge_prod(tracks, sam_masks, windows, margins, rename_events)
        corrected = derive_bboxes(merged)
    else:
        merged, _, rename_map = step_merge(tracks, sam_masks, windows, margins, rename_events)
        corrected = extract_bboxes(merged, tracks, rename_map, match_history, windows)

    return corrected


def _evaluate(tracks, source_path, label):
    import shutil
    import tempfile
    from pathlib import Path

    from .core.eval import _eval_mot
    from .utils.export import export_mot

    gt_path = Path(source_path) / "gt" / "gt.txt"
    if not gt_path.exists():
        print(f"  {label}: no ground truth, skipping eval")
        return

    seq_name = Path(source_path).name
    tmp_dir = tempfile.mkdtemp()
    try:
        mot_path = str(Path(tmp_dir) / f"{seq_name}.txt")
        export_mot(tracks, mot_path)
        r, _ = _eval_mot(seq_name, source_path, mot_path)
        print(f"  {label}: HOTA={r['hota']:.1f}  AssA={r['assa']:.1f}  "
              f"IDF1={r['idf1']:.1f}  IDSW={r['idsw']}")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    import argparse

    from .core.detection import detect_precomputed
    from .deep_eiou.tracker import step_track
    from .utils.helpers import read_sequence_info

    parser = argparse.ArgumentParser(description="Test augment API")
    parser.add_argument("input", help="Sequence directory")
    parser.add_argument("--sam3", action="store_true", help="Use SAM3 instead of SAM2")
    parser.add_argument("--mode", choices=["benchmark", "prod"], default="benchmark",
                        help="Output mode (see augment docstring)")
    args = parser.parse_args()

    seq_info = read_sequence_info(args.input)

    print("Loading precomputed detections...")
    detections, embeddings = detect_precomputed(args.input)

    print("Running Deep-EIoU...")
    tracks, margins = step_track(detections, embeddings, seq_info["fps"],
                                  seq_info["width"], seq_info["height"])

    print("Augmenting with SAM...")
    corrected = augment(tracks, margins, args.input, sam3=args.sam3, mode=args.mode)

    print("\nResults:")
    _evaluate(tracks, args.input, "Deep-EIoU")
    _evaluate(corrected, args.input, "SAM-Deep-EIoU")
