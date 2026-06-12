"""Debug artifacts for GTA steps.

Ported from SAM-SORT: debug.py
"""

import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from ..utils.export import parse_mot_tracks
from .pose import is_legible, crop_torso, TORSO_KP_INDICES

HIGHLIGHT = (0, 255, 0)
WHITE = (255, 255, 255)
TORSO_COLOR = (0, 255, 0)
SKELETON_COLOR = (255, 100, 0)
TORSO_KEYPOINTS = {5, 6, 11, 12}
TORSO_EDGES = {(5, 6), (5, 11), (6, 12), (11, 12)}
KEYPOINT_RADIUS = 2
CONFIDENCE_THRESHOLD = 0.3

COCO_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def _read_frames(source_path: str, frame_indices) -> Dict[int, np.ndarray]:
    frame_cache = {}
    frame_dir = Path(source_path) / "img1"
    frame_files = sorted(frame_dir.glob("*.jpg"))
    for fidx in sorted(frame_indices):
        if 0 <= fidx < len(frame_files):
            frame = cv2.imread(str(frame_files[fidx]))
            if frame is not None:
                frame_cache[fidx] = frame
    return frame_cache


def _sample_entries(entries: list, n: int) -> list:
    if len(entries) <= n:
        return entries
    step = len(entries) / n
    return [entries[int(i * step)] for i in range(n)]



def _make_crop_cell(
    frame: np.ndarray,
    bbox_xywh: list,
    cell_size: tuple,
    pad_fraction: float,
    label: str,
    bbox_color: tuple,
) -> np.ndarray:
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    fh, fw = frame.shape[:2]

    pad_x = int((x2 - x1) * pad_fraction)
    pad_y = int((y2 - y1) * pad_fraction)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(fw, x2 + pad_x)
    cy2 = min(fh, y2 + pad_y)

    crop = frame[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    bx1, by1 = x1 - cx1, y1 - cy1
    bx2, by2 = x2 - cx1, y2 - cy1
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), bbox_color, 2)

    cell_w, cell_h = cell_size
    crop = cv2.resize(crop, (cell_w, cell_h))
    cv2.putText(crop, label, (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1)
    return crop


def save_classify_debug(
    colors_dict: dict,
    color_frames: list,
    classifications: dict,
    grids: dict,
    output_dir: str,
    variant: str = "",
) -> None:
    """Save classify debug: color frames, per-track grids with classification, log JSON."""
    debug_dir = Path(output_dir) / "debug" / variant / "classify"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(color_frames):
        cv2.imwrite(str(debug_dir / f"color_frame_{i}.jpg"), frame)

    HEADER_H = 28
    CLASS_COLORS = {
        "team_a": (182, 107, 0),
        "team_b": (51, 122, 0),
        "other": (128, 128, 128),
    }

    for tid, grid in sorted(grids.items()):
        cls = classifications.get(str(tid), classifications.get(tid, "other"))
        header_color = CLASS_COLORS.get(cls, (128, 128, 128))
        header_text = f"Track {tid} -> {cls}"

        grid_with_header = np.zeros(
            (grid.shape[0] + HEADER_H, grid.shape[1], 3), dtype=np.uint8
        )
        cv2.rectangle(
            grid_with_header, (0, 0), (grid.shape[1], HEADER_H), header_color, -1
        )
        cv2.putText(
            grid_with_header, header_text, (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        grid_with_header[HEADER_H:, :] = grid
        cv2.imwrite(str(debug_dir / f"T{tid:03d}_classify.jpg"), grid_with_header)

    log = {
        "colors": colors_dict,
        "classifications": {str(k): v for k, v in classifications.items()},
    }
    with open(debug_dir / "classify_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"Classify debug saved to: {debug_dir}")


def save_pose_debug(
    detections: Dict[int, np.ndarray],
    pose_data: Dict[int, list],
    source_path: str,
    output_dir: str,
    n_samples: int = 20,
) -> None:
    """Save pose debug: sample frames with COCO skeleton overlays on all detections."""
    debug_dir = Path(output_dir) / "debug" / "jersey" / "pose"
    debug_dir.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(f for f, poses in pose_data.items() if poses)
    if not all_frames:
        return

    stride = max(1, len(all_frames) // n_samples)
    sample_frames = all_frames[::stride][:n_samples]
    frame_cache = _read_frames(source_path, sample_frames)

    for frame_idx in sample_frames:
        frame = frame_cache.get(frame_idx)
        if frame is None:
            continue
        frame = frame.copy()

        poses = pose_data.get(frame_idx, [])
        if not poses:
            continue

        for det_idx, pose in enumerate(poses):
            if pose is None:
                continue

            keypoints = pose["keypoints"]
            scores = pose["scores"]

            valid_kps = {}
            for kp_idx, (kp, score) in enumerate(zip(keypoints, scores)):
                if score > CONFIDENCE_THRESHOLD:
                    valid_kps[kp_idx] = (int(kp[0]), int(kp[1]))

            if not valid_kps:
                continue

            for kp_idx1, kp_idx2 in COCO_SKELETON_EDGES:
                if kp_idx1 in valid_kps and kp_idx2 in valid_kps:
                    is_torso = (kp_idx1, kp_idx2) in TORSO_EDGES
                    color = TORSO_COLOR if is_torso else SKELETON_COLOR
                    cv2.line(frame, valid_kps[kp_idx1], valid_kps[kp_idx2],
                             color, 1, cv2.LINE_AA)

            for kp_idx, (x, y) in valid_kps.items():
                color = TORSO_COLOR if kp_idx in TORSO_KEYPOINTS else SKELETON_COLOR
                cv2.circle(frame, (x, y), KEYPOINT_RADIUS, color, -1, cv2.LINE_AA)

            if 0 in valid_kps:
                cv2.putText(frame, f"D{det_idx}",
                            (valid_kps[0][0], valid_kps[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, SKELETON_COLOR, 2, cv2.LINE_AA)

        cv2.putText(frame, f"f{frame_idx}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)
        cv2.imwrite(str(debug_dir / f"f{frame_idx:04d}.jpg"), frame)

    print(f"Pose debug saved to: {debug_dir} ({len(sample_frames)} images)")


def save_crop_debug(
    detections: Dict[int, np.ndarray],
    pose_data: Dict[int, list],
    source_path: str,
    output_dir: str,
    ocr_results: Dict[int, list],
    n_samples: int = 20,
    cell_size: tuple = (80, 120),
    grid_cols: int = 10,
) -> None:
    """Save torso crop debug: two-section grid per frame (all crops | legible+OCR) + log."""
    debug_dir = Path(output_dir) / "debug" / "jersey" / "crops"
    debug_dir.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(f for f, poses in pose_data.items() if poses)
    if not all_frames:
        return

    stride = max(1, len(all_frames) // n_samples)
    sample_frames = all_frames[::stride][:n_samples]
    frame_cache = _read_frames(source_path, sample_frames)

    cell_w, cell_h = cell_size
    SEP_H = 3
    saved = 0
    log = []

    for frame_idx in sample_frames:
        frame = frame_cache.get(frame_idx)
        if frame is None:
            continue

        poses = pose_data.get(frame_idx, [])
        frame_ocr = ocr_results.get(frame_idx, [])
        all_crops_list = []
        legible_crops = []
        frame_log = {"frame": frame_idx, "detections": []}

        for det_idx, pose in enumerate(poses):
            if pose is None:
                frame_log["detections"].append({"det": det_idx, "pose": False})
                continue

            scores = pose["scores"]
            torso_scores = [round(float(scores[i]), 3) for i in TORSO_KP_INDICES]
            legible = is_legible(pose["keypoints"], scores)
            crop, _ = crop_torso(frame, pose["keypoints"], scores)
            has_crop = crop is not None

            ocr = frame_ocr[det_idx] if det_idx < len(frame_ocr) else None
            det_log = {
                "det": det_idx, "pose": True,
                "torso_scores": torso_scores, "has_crop": has_crop, "legible": legible,
            }
            if ocr is not None:
                det_log["ocr_label"] = ocr["label"]
                det_log["ocr_confidence"] = round(ocr["confidence"], 4)
            frame_log["detections"].append(det_log)

            if crop is None:
                continue
            all_crops_list.append((det_idx, crop))
            if legible:
                legible_crops.append((det_idx, crop, ocr))

        frame_log["total"] = len(poses)
        frame_log["cropped"] = len(all_crops_list)
        frame_log["legible"] = len(legible_crops)
        log.append(frame_log)

        if not all_crops_list:
            continue

        n_all = len(all_crops_list)
        rows_all = (n_all + grid_cols - 1) // grid_cols
        top_h = rows_all * cell_h

        n_leg = len(legible_crops)
        rows_leg = max(1, (n_leg + grid_cols - 1) // grid_cols)
        bot_h = rows_leg * cell_h

        grid_w = grid_cols * cell_w
        grid = np.zeros((top_h + SEP_H + bot_h, grid_w, 3), dtype=np.uint8)

        for idx, (det_idx, crop) in enumerate(all_crops_list):
            resized = cv2.resize(crop, (cell_w, cell_h))
            row, col = idx // grid_cols, idx % grid_cols
            grid[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = resized
            cv2.putText(grid, f"D{det_idx}", (col * cell_w + 2, row * cell_h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)

        grid[top_h:top_h + SEP_H, :] = HIGHLIGHT

        y_off = top_h + SEP_H
        for idx, (det_idx, crop, ocr) in enumerate(legible_crops):
            resized = cv2.resize(crop, (cell_w, cell_h))
            row, col = idx // grid_cols, idx % grid_cols
            gy = y_off + row * cell_h
            gx = col * cell_w
            grid[gy:gy + cell_h, gx:gx + cell_w] = resized
            cv2.putText(grid, f"D{det_idx}", (gx + 2, gy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, HIGHLIGHT, 1)
            if ocr is not None:
                if ocr["label"]:
                    label = f"{ocr['label']} {ocr['confidence']:.2f}"
                    color = HIGHLIGHT if ocr["confidence"] >= 0.8 else (0, 0, 255)
                else:
                    label = "-"
                    color = (0, 0, 255)
                cv2.putText(grid, label, (gx + 2, gy + cell_h - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        cv2.imwrite(str(debug_dir / f"f{frame_idx:04d}.jpg"), grid)
        saved += 1

    (debug_dir / "crops_log.json").write_text(json.dumps(log, indent=2))
    print(f"Crop debug saved to: {debug_dir} ({saved} frames)")


def save_jersey_track_debug(
    ocr_results: Dict[int, list],
    crops: Dict[int, dict],
    assignments: Dict[int, Dict[int, int]],
    jersey_map: Dict[int, dict],
    output_dir: str,
    variant: str = "",
    max_samples: int = 20,
    cell_size: tuple = (80, 120),
    grid_cols: int = 10,
) -> None:
    """Save per-tracklet jersey debug: grid of sampled torso crops with OCR labels + vote."""
    debug_dir = Path(output_dir) / "debug" / variant / "jersey" / "track"
    debug_dir.mkdir(parents=True, exist_ok=True)

    HEADER_H = 24
    cell_w, cell_h = cell_size

    track_entries: Dict[int, list] = {}
    for frame_idx, frame_assignments in assignments.items():
        for det_idx, track_id in frame_assignments.items():
            frame_crops = crops.get(frame_idx, {})
            crop = frame_crops.get(det_idx)
            if crop is None:
                continue
            ocr = None
            frame_ocr = ocr_results.get(frame_idx, [])
            if det_idx < len(frame_ocr):
                ocr = frame_ocr[det_idx]
            track_entries.setdefault(track_id, []).append((frame_idx, crop, ocr))

    for tid in track_entries:
        track_entries[tid].sort(key=lambda x: x[0])

    top_k = 5
    saved = 0
    for tid, entries in sorted(track_entries.items()):
        sampled = _sample_entries(entries, max_samples)
        n = len(sampled)
        n_rows = (n + grid_cols - 1) // grid_cols

        vote = jersey_map.get(tid)
        voted_number = vote["number"] if vote else -1

        if voted_number != -1:
            header = f"Track {tid} -> #{voted_number}"
        else:
            header = f"Track {tid} -> illegible"
        if vote and vote.get("detail", {}).get("candidates"):
            top = sorted(vote["detail"]["candidates"].items(), key=lambda x: -x[1])[:3]
            header += "  " + " ".join(f"{k}:{v}" for k, v in top)

        top_ocr = sorted(
            [(f, c, o) for f, c, o in entries if o is not None and o["label"]],
            key=lambda x: -x[2]["confidence"],
        )[:top_k]

        top_section_h = (HEADER_H + cell_h) if top_ocr else 0
        grid_h = HEADER_H + n_rows * cell_h + top_section_h
        grid_w = grid_cols * cell_w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        header_color = HIGHLIGHT if voted_number != -1 else (0, 0, 255)
        cv2.putText(grid, header, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, header_color, 1, cv2.LINE_AA)

        for idx, (frame_idx, crop, ocr) in enumerate(sampled):
            resized = cv2.resize(crop, (cell_w, cell_h))
            row, col = idx // grid_cols, idx % grid_cols
            gy = HEADER_H + row * cell_h
            gx = col * cell_w
            grid[gy:gy + cell_h, gx:gx + cell_w] = resized

            cv2.putText(grid, f"f{frame_idx}", (gx + 2, gy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1)

            if ocr is not None and ocr["label"]:
                matches_vote = voted_number != -1 and ocr["label"] == str(voted_number)
                color = HIGHLIGHT if matches_vote else (0, 0, 255)
                label = f"{ocr['label']} {ocr['confidence']:.2f}"
            else:
                color = (100, 100, 100)
                label = "-"
            cv2.putText(grid, label, (gx + 2, gy + cell_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        if top_ocr:
            section_y = HEADER_H + n_rows * cell_h
            cv2.putText(grid, f"top {len(top_ocr)} OCR reads", (4, section_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            for idx, (frame_idx, crop, ocr) in enumerate(top_ocr):
                resized = cv2.resize(crop, (cell_w, cell_h))
                gx = idx * cell_w
                gy = section_y + HEADER_H
                grid[gy:gy + cell_h, gx:gx + cell_w] = resized
                cv2.putText(grid, f"f{frame_idx}", (gx + 2, gy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1)
                label = f"{ocr['label']} {ocr['confidence']:.2f}"
                matches_vote = voted_number != -1 and ocr["label"] == str(voted_number)
                color = HIGHLIGHT if matches_vote else (0, 0, 255)
                cv2.putText(grid, label, (gx + 2, gy + cell_h - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        cv2.imwrite(str(debug_dir / f"T{tid:03d}.jpg"), grid)
        saved += 1

    print(f"Jersey track debug saved to: {debug_dir} ({saved} tracklets)")


def save_embed_debug(
    track_embeddings: Dict[int, np.ndarray],
    output_dir: str,
    variant: str = "",
) -> None:
    """Save PCA scatter plot of mean OSNet embeddings per track."""
    from sklearn.decomposition import PCA

    tids = sorted(track_embeddings.keys())
    if len(tids) < 3:
        return

    means = np.array([track_embeddings[tid].mean(axis=0) for tid in tids])
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(means)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X2d[:, 0], X2d[:, 1], s=60, edgecolors="black", linewidths=0.5)
    for i, tid in enumerate(tids):
        ax.annotate(str(tid), (X2d[i, 0], X2d[i, 1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("Track appearance embeddings (PCA of mean OSNet)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    debug_dir = Path(output_dir) / "debug" / variant / "embed"
    debug_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(debug_dir / "clusters.png"), dpi=150)
    plt.close(fig)
    print(f"Embed scatter saved to: {debug_dir / 'clusters.png'}")


def save_gta_debug(
    merge_events: list,
    canonical_to_gta: dict,
    mot_path: str,
    source_path: str,
    output_dir: str,
    variant: str = "",
    samples_per_tracklet: int = 5,
    cell_size: tuple = (100, 140),
    pad_fraction: float = 0.3,
) -> None:
    """Save GTA debug: per-merge pair image + JSON log."""
    debug_dir = Path(output_dir) / "debug" / variant / "gta"
    debug_dir.mkdir(parents=True, exist_ok=True)

    c2g = {int(k): v for k, v in canonical_to_gta.items()}

    log = []
    for i, event in enumerate(merge_events):
        entry = {"event_id": i, **event}
        entry["gta_id"] = int(c2g.get(event["kept_id"], -1))
        log.append(entry)
    (debug_dir / "gta_log.json").write_text(json.dumps(log, indent=2))

    if not merge_events:
        print(f"GTA debug saved to: {debug_dir} (no merges)")
        return

    mot_tracks = parse_mot_tracks(mot_path)

    merge_data = []
    frames_needed = set()
    for event in merge_events:
        kept_entries = []
        for tid in event["kept_members"]:
            kept_entries.extend(mot_tracks.get(tid, []))
        kept_entries.sort(key=lambda x: x[0])

        absorbed_entries = []
        for tid in event["absorbed_members"]:
            absorbed_entries.extend(mot_tracks.get(tid, []))
        absorbed_entries.sort(key=lambda x: x[0])

        kept_samples = _sample_entries(kept_entries, samples_per_tracklet)
        absorbed_samples = _sample_entries(absorbed_entries, samples_per_tracklet)

        for frame_0idx, _ in kept_samples + absorbed_samples:
            frames_needed.add(frame_0idx)
        merge_data.append((event, kept_samples, absorbed_samples))

    frame_cache = _read_frames(source_path, frames_needed)

    KEPT_COLOR = HIGHLIGHT
    ABSORBED_COLOR = (0, 180, 255)
    HEADER_H = 28
    SEP_H = 3
    cell_w, cell_h = cell_size

    for idx, (event, kept_samples, absorbed_samples) in enumerate(merge_data):
        gta_id = canonical_to_gta.get(event["kept_id"], "?")
        n_cols = max(len(kept_samples), len(absorbed_samples), 1)

        img_w = n_cols * cell_w
        img_h = HEADER_H + cell_h + SEP_H + cell_h
        grid = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        kept_str = "+".join(f"C{t}" for t in event["kept_members"])
        absorbed_str = "+".join(f"C{t}" for t in event["absorbed_members"])
        header = f"{kept_str} + {absorbed_str} -> G{gta_id} | dist: {event['distance']}"
        cv2.putText(grid, header, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

        for col, (frame_0idx, bbox) in enumerate(kept_samples):
            frame = frame_cache.get(frame_0idx)
            if frame is None:
                continue
            cell = _make_crop_cell(frame, bbox, cell_size, pad_fraction,
                                   f"f{frame_0idx}", KEPT_COLOR)
            grid[HEADER_H:HEADER_H + cell_h, col * cell_w:(col + 1) * cell_w] = cell

        sep_y = HEADER_H + cell_h
        grid[sep_y:sep_y + SEP_H, :] = KEPT_COLOR

        row_y = HEADER_H + cell_h + SEP_H
        for col, (frame_0idx, bbox) in enumerate(absorbed_samples):
            frame = frame_cache.get(frame_0idx)
            if frame is None:
                continue
            cell = _make_crop_cell(frame, bbox, cell_size, pad_fraction,
                                   f"f{frame_0idx}", ABSORBED_COLOR)
            grid[row_y:row_y + cell_h, col * cell_w:(col + 1) * cell_w] = cell

        cv2.imwrite(str(debug_dir / f"merge_{idx:03d}.jpg"), grid)

    print(f"GTA debug saved to: {debug_dir} ({len(merge_events)} merges)")
