"""Augment a base tracker's output with selective mask propagation.

Given tracks and assignment margins from any Hungarian-based tracker,
detects ambiguous windows, propagates SAM masks through them, and
corrects identity switches. Only modifies the output when positive
evidence of a swap is found.

API:
    from smp.augment import augment
    corrected_tracks = augment(tracks, margins, "path/to/sequence")

Test:
    uv run python -m smp.augment data/sportsmot/dataset/val/v_00HRwkvvjtQ_c005 --sam3
    uv run python -m smp.augment data/dancetrack/val/dancetrack0007 --sam3
"""

from typing import Dict

import numpy as np


def augment(
    tracks: Dict[int, Dict[int, np.ndarray]],
    margins: Dict[int, Dict[int, float]],
    source_path: str,
    sam3: bool = False,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Augment tracker output with selective SAM mask propagation.

    Args:
        tracks: {frame: {track_id: bbox}} where bbox is [x1, y1, x2, y2].
        margins: {frame: {track_id: margin}} where margin is the difference
            between the second-best and best assignment cost in the cost matrix.
        source_path: Path to sequence directory containing img1/.
        sam3: Use SAM3 instead of SAM2.

    Returns:
        Corrected tracks in the same format as the input.
    """
    if sam3:
        from .core.sam3 import build_predictor, step_sam
    else:
        from .core.sam2 import build_predictor, step_sam

    from .core.merge import step_merge, extract_bboxes

    predictor = build_predictor()
    sam_masks, windows, rename_events, match_history = step_sam(predictor, tracks, margins, source_path)
    merged, _, rename_map = step_merge(tracks, sam_masks, windows, margins, rename_events)
    corrected = extract_bboxes(merged, tracks, rename_map, match_history, windows)

    return corrected


def _evaluate(tracks, source_path, label):
    import shutil
    import tempfile
    from pathlib import Path

    import trackeval

    from .utils.export import export_mot

    gt_path = Path(source_path) / "gt" / "gt.txt"
    if not gt_path.exists():
        print(f"  {label}: no ground truth, skipping eval")
        return

    seq_name = Path(source_path).name
    gt_folder = str(Path(source_path).parent)

    tmp_dir = tempfile.mkdtemp()
    try:
        tracker_dir = Path(tmp_dir) / "tracker" / "data"
        tracker_dir.mkdir(parents=True)
        export_mot(tracks, str(tracker_dir / f"{seq_name}.txt"))

        dataset = trackeval.datasets.MotChallenge2DBox({
            "GT_FOLDER": gt_folder,
            "TRACKERS_FOLDER": tmp_dir,
            "TRACKERS_TO_EVAL": ["tracker"],
            "BENCHMARK": "", "SPLIT_TO_EVAL": "",
            "SKIP_SPLIT_FOL": True, "DO_PREPROC": False,
            "SEQ_INFO": {seq_name: None},
            "CLASSES_TO_EVAL": ["pedestrian"],
            "TRACKER_SUB_FOLDER": "data",
            "PRINT_CONFIG": False,
        })

        raw_data = dataset.get_raw_seq_data("tracker", seq_name)
        data = dataset.get_preprocessed_seq_data(raw_data, "pedestrian")

        cfg = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
        metrics = [
            trackeval.metrics.HOTA(cfg),
            trackeval.metrics.CLEAR(cfg),
            trackeval.metrics.Identity(cfg),
        ]
        res = {}
        for m in metrics:
            res[m.get_name()] = m.eval_sequence(data)

        hota = float(np.mean(res["HOTA"]["HOTA"])) * 100
        assa = float(np.mean(res["HOTA"]["AssA"])) * 100
        idf1 = float(res["Identity"]["IDF1"]) * 100
        idsw = int(res["CLEAR"]["IDSW"])
        print(f"  {label}: HOTA={hota:.1f}  AssA={assa:.1f}  IDF1={idf1:.1f}  IDSW={idsw}")
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
    args = parser.parse_args()

    seq_info = read_sequence_info(args.input)

    print("Loading precomputed detections...")
    detections, embeddings = detect_precomputed(args.input)

    print("Running Deep-EIoU...")
    tracks, margins = step_track(detections, embeddings, seq_info["fps"],
                                  seq_info["width"], seq_info["height"])

    print("Augmenting with SAM...")
    corrected = augment(tracks, margins, args.input, sam3=args.sam3)

    print("\nResults:")
    _evaluate(tracks, args.input, "Deep-EIoU")
    _evaluate(corrected, args.input, "SAM-Deep-EIoU")
