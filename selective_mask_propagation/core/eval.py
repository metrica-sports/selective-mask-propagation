"""TrackEval wrapper for SportsMOT evaluation."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.artifacts import get_artifacts_dir


def step_eval(source_path: str, output_dir: str, suffix: str = "", no_gta: bool = False) -> dict:
    """Evaluate all available MOT files against ground truth.

    Returns dict of frame errors keyed by json_key (e.g. "deep_eiou",
    "sam_deep_eiou", "de_gta", "sde_gta").
    """
    source = Path(source_path)
    if not (source / "gt" / "gt.txt").exists():
        print("Eval: skipping (no ground truth).")
        return {}

    artifacts_dir = get_artifacts_dir(source_path, suffix)
    seq_name = Path(source_path).name

    columns = []
    eval_json = {"seq_name": seq_name}
    all_frame_errors = {}

    variants = [
        ("Deep-EIoU", "mot_deep_eiou.txt", "deep_eiou"),
        ("SAM-Deep-EIoU", "mot_sam_deep_eiou.txt", "sam_deep_eiou"),
    ]
    if not no_gta:
        variants += [
            ("DE+GTA", "mot_de_gta_interp.txt", "de_gta"),
            ("SDE+GTA", "mot_sde_gta_interp.txt", "sde_gta"),
        ]

    for label, filename, json_key in variants:
        mot_path = artifacts_dir / filename
        if not mot_path.exists():
            continue
        r, frame_errors = _eval_mot(seq_name, source_path, str(mot_path))
        columns.append((label, r))
        eval_json[json_key] = r
        all_frame_errors[json_key] = frame_errors

    if not columns:
        print("No MOT files found to evaluate.")
        return {}

    md = f"# Eval: {seq_name}\n\n"
    md += _fmt_summary_table(columns)

    for label, r in columns:
        md += f"\n**{label}**\n\n"
        md += f"| TP | FP | FN | Frag | MOTP |\n"
        md += f"|----|----|----|----- |------|\n"
        md += f"| {r['clr_tp']} | {r['clr_fp']} | {r['clr_fn']} | {r['frag']} | {r['motp']:.1f} |\n"

    eval_path = Path(output_dir) / "eval.md"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(md)

    json_path = Path(output_dir) / "eval.json"
    json_path.write_text(json.dumps(eval_json, indent=2) + "\n")
    print(f"Eval: {eval_path}")

    return all_frame_errors


def _eval_mot(seq_name: str, source_path: str, mot_path: str) -> Tuple[dict, dict]:
    """Evaluate a MOT file and extract per-frame errors.

    Sets up TrackEval dataset, runs metrics directly on preprocessed data,
    and extracts per-frame FP/FN using CLEAR's exact matching protocol.

    Returns (metrics_dict, frame_errors).
    """
    import trackeval

    gt_folder = str(Path(source_path).parent)
    tmp_dir = tempfile.mkdtemp(prefix="trackeval_")
    try:
        tracker_data_dir = Path(tmp_dir) / "tracker" / "data"
        tracker_data_dir.mkdir(parents=True)
        shutil.copy2(mot_path, str(tracker_data_dir / f"{seq_name}.txt"))

        dataset = trackeval.datasets.MotChallenge2DBox({
            "GT_FOLDER": gt_folder,
            "TRACKERS_FOLDER": tmp_dir,
            "TRACKERS_TO_EVAL": ["tracker"],
            "BENCHMARK": "",
            "SPLIT_TO_EVAL": "",
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": False,
            "SEQ_INFO": {seq_name: None},
            "CLASSES_TO_EVAL": ["pedestrian"],
            "TRACKER_SUB_FOLDER": "data",
            "PRINT_CONFIG": False,
        })

        raw_data = dataset.get_raw_seq_data("tracker", seq_name)
        data = dataset.get_preprocessed_seq_data(raw_data, "pedestrian")
        gt_reverse, tracker_reverse = _build_id_reverse_maps(raw_data)

        metrics_config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
        metrics = [
            trackeval.metrics.HOTA(metrics_config),
            trackeval.metrics.CLEAR(metrics_config),
            trackeval.metrics.Identity(metrics_config),
            trackeval.metrics.Count(),
        ]
        seq_res = {}
        for metric in metrics:
            seq_res[metric.get_name()] = metric.eval_sequence(data)

        r = {
            "hota": float(np.mean(seq_res["HOTA"]["HOTA"])) * 100,
            "deta": float(np.mean(seq_res["HOTA"]["DetA"])) * 100,
            "assa": float(np.mean(seq_res["HOTA"]["AssA"])) * 100,
            "mota": float(seq_res["CLEAR"]["MOTA"]) * 100,
            "idsw": int(seq_res["CLEAR"]["IDSW"]),
            "idf1": float(seq_res["Identity"]["IDF1"]) * 100,
            "gt_ids": int(seq_res["Count"]["GT_IDs"]),
            "pred_ids": int(seq_res["Count"]["IDs"]),
            "clr_tp": int(seq_res["CLEAR"]["CLR_TP"]),
            "clr_fp": int(seq_res["CLEAR"]["CLR_FP"]),
            "clr_fn": int(seq_res["CLEAR"]["CLR_FN"]),
            "frag": int(seq_res["CLEAR"]["Frag"]),
            "motp": float(seq_res["CLEAR"]["MOTP"]) * 100,
        }

        frame_errors = _extract_frame_errors(data, gt_reverse, tracker_reverse)
        return r, frame_errors
    finally:
        shutil.rmtree(tmp_dir)


def _build_id_reverse_maps(
    raw_data: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build remapped_id -> original_id maps from raw TrackEval data.

    Replicates the ID collection from get_preprocessed_seq_data (with
    DO_PREPROC=False): all tracker IDs survive, GT IDs survive unless
    zero-marked.
    """
    unique_gt = set()
    unique_tracker = set()
    for t in range(raw_data["num_timesteps"]):
        zero_marked = raw_data["gt_extras"][t]["zero_marked"]
        keep = zero_marked != 0
        unique_gt.update(raw_data["gt_ids"][t][keep].tolist())
        unique_tracker.update(raw_data["tracker_ids"][t].tolist())
    return (
        np.array(sorted(unique_gt), dtype=np.int64),
        np.array(sorted(unique_tracker), dtype=np.int64),
    )


def _extract_frame_errors(
    data: dict,
    gt_reverse: np.ndarray,
    tracker_reverse: np.ndarray,
) -> dict:
    """Extract per-frame FP/FN using CLEAR's exact matching protocol.

    Replays CLEAR's identity-aware Hungarian matching: a 1000x score bonus
    for pairings that maintain the same tracker-to-GT assignment from the
    previous frame, then Hungarian on the augmented score matrix with a
    0.5 IoU threshold.

    Returns {frame_idx: {"fn": [(gt_id, x1, y1, x2, y2), ...],
                          "fp": [(pred_id, x1, y1, x2, y2), ...]}}.
    Frame indices are 0-based. IDs are original (pre-remap).
    """
    threshold = 0.5
    num_gt_ids = data["num_gt_ids"]
    prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)

    def _orig_gt(remapped):
        return int(gt_reverse[remapped]) if len(gt_reverse) > 0 else int(remapped)

    def _orig_tracker(remapped):
        return int(tracker_reverse[remapped]) if len(tracker_reverse) > 0 else int(remapped)

    def _det_to_xyxy(det):
        x, y, w, h = float(det[0]), float(det[1]), float(det[2]), float(det[3])
        return x, y, x + w, y + h

    errors = {}
    for t in range(data["num_timesteps"]):
        gt_ids_t = data["gt_ids"][t]
        tracker_ids_t = data["tracker_ids"][t]
        gt_dets_t = data["gt_dets"][t]
        tracker_dets_t = data["tracker_dets"][t]

        fn_list = []
        fp_list = []
        idsw_list = []

        if len(gt_ids_t) == 0:
            for i in range(len(tracker_ids_t)):
                fp_list.append((_orig_tracker(tracker_ids_t[i]), *_det_to_xyxy(tracker_dets_t[i])))
        elif len(tracker_ids_t) == 0:
            for i in range(len(gt_ids_t)):
                fn_list.append((_orig_gt(gt_ids_t[i]), *_det_to_xyxy(gt_dets_t[i])))
        else:
            # CLEAR's identity-aware scoring: 1000x bonus for maintaining
            # the same tracker-GT pairing from the previous frame.
            similarity = data["similarity_scores"][t]
            score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < threshold - np.finfo("float").eps] = 0

            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched = score_mat[match_rows, match_cols] > 0 + np.finfo("float").eps
            match_rows = match_rows[actually_matched]
            match_cols = match_cols[actually_matched]

            matched_gt_ids = gt_ids_t[match_rows]
            matched_tracker_ids = tracker_ids_t[match_cols]

            # Detect IDSW before updating prev state: GT matched to a
            # different tracker than previous frame.
            for row, col in zip(match_rows, match_cols):
                gt_rem = gt_ids_t[row]
                tr_rem = tracker_ids_t[col]
                prev_tr = prev_timestep_tracker_id[gt_rem]
                if not np.isnan(prev_tr) and int(prev_tr) != tr_rem:
                    idsw_list.append((
                        _orig_gt(gt_rem),
                        _orig_tracker(int(prev_tr)),
                        _orig_tracker(tr_rem),
                        *_det_to_xyxy(gt_dets_t[row])
                    ))

            # Update exactly like CLEAR: clear all, then set matched pairs.
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

            matched_gt = set(match_rows.tolist())
            matched_pred = set(match_cols.tolist())

            for i in range(len(gt_ids_t)):
                if i not in matched_gt:
                    fn_list.append((_orig_gt(gt_ids_t[i]), *_det_to_xyxy(gt_dets_t[i])))
            for i in range(len(tracker_ids_t)):
                if i not in matched_pred:
                    fp_list.append((_orig_tracker(tracker_ids_t[i]), *_det_to_xyxy(tracker_dets_t[i])))

        if fn_list or fp_list or idsw_list:
            errors[t] = {"fn": fn_list, "fp": fp_list, "idsw": idsw_list}

    total_fn = sum(len(v["fn"]) for v in errors.values())
    total_fp = sum(len(v["fp"]) for v in errors.values())
    total_idsw = sum(len(v.get("idsw", [])) for v in errors.values())
    print(f"  Frame errors: {total_fn} FN, {total_fp} FP, {total_idsw} IDSW")
    return errors


def _fmt_metric(r: dict, key: str) -> str:
    v = r[key]
    return str(v) if isinstance(v, int) else f"{v:.1f}"


def _fmt_summary_table(columns: List[Tuple[str, dict]]) -> str:
    """Build a markdown summary table from [(label, metrics_dict), ...]."""
    labels = [label for label, _ in columns]
    header = "| Metric | " + " | ".join(labels) + " |"
    sep = "|--------|" + "|".join("-------" for _ in columns) + "|"

    rows = [header, sep]
    for key, name in [
        ("hota", "HOTA"), ("deta", "DetA"), ("assa", "AssA"),
        ("mota", "MOTA"), ("idf1", "IDF1"), ("idsw", "IDSW"),
        ("gt_ids", "GT IDs"), ("pred_ids", "Pred IDs"),
    ]:
        cells = " | ".join(_fmt_metric(r, key) for _, r in columns)
        rows.append(f"| {name} | {cells} |")

    return "\n".join(rows) + "\n"
