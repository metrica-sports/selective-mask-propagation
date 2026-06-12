"""ViTPose estimation for jersey OCR.

Runs ViTPose on all detection bboxes to get body keypoints. The torso
keypoints (shoulders + hips) define the crop region for jersey OCR.

Ported from SAM-SORT: pose.py
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

TORSO_KP_INDICES = [5, 6, 11, 12]  # left_shoulder, right_shoulder, left_hip, right_hip

VITPOSE_MODEL = "usyd-community/vitpose-plus-base"


POSE_BATCH = 256  # crops per forward; frames accumulate until this fills


def _frame_generator(source_path: str):
    """Yield (total_frames, frame_iterator) for an image sequence directory."""
    frame_dir = Path(source_path) / "img1"
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No jpg frames found in {frame_dir}")

    def gen():
        for f in frame_files:
            yield cv2.imread(str(f))

    return len(frame_files), gen()


class _FrameDataset(torch.utils.data.Dataset):
    """Worker-decoded RGB frames for an image sequence directory."""

    def __init__(self, frame_files: List[Path]):
        self.frame_files = frame_files

    def __len__(self) -> int:
        return len(self.frame_files)

    def __getitem__(self, idx: int):
        frame = cv2.imread(str(self.frame_files[idx]))
        return idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _flush_pose_batch(pending: List[dict], processor, model, pose_data: Dict[int, list]) -> None:
    """One forward over all accumulated frames' crops, scattered back per frame."""
    if not pending:
        return

    boxes_groups = []
    for p in pending:
        xyxy = p["boxes_xyxy"]
        xywh = xyxy.copy()
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        boxes_groups.append(xywh)

    inputs = processor([p["image"] for p in pending], boxes=boxes_groups,
                       return_tensors="pt").to(model.device)
    n_crops = sum(len(g) for g in boxes_groups)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
        inputs = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in inputs.items()}
        dataset_index = torch.zeros(n_crops, dtype=torch.long, device=model.device)
        outputs = model(**inputs, dataset_index=dataset_index)

    if outputs.heatmaps.dtype == torch.bfloat16:
        outputs.heatmaps = outputs.heatmaps.float()

    pose_results = processor.post_process_pose_estimation(outputs, boxes=boxes_groups)

    for p, group in zip(pending, pose_results):
        results = [{
            "keypoints": r["keypoints"].cpu().numpy(),
            "scores": r["scores"].cpu().numpy(),
        } for r in group]
        frame_poses = []
        result_idx = 0
        for i in range(p["n_dets"]):
            if p["valid"][i]:
                frame_poses.append(results[result_idx])
                result_idx += 1
            else:
                frame_poses.append(None)
        pose_data[p["frame_idx"]] = frame_poses


def estimate_all_poses(
    source_path: str,
    detections: Dict[int, np.ndarray],
    processor,
    model,
) -> Dict[int, list]:
    """Run ViTPose on every detection bbox in every frame.

    Frames are decoded by DataLoader workers and their crops accumulate
    into one forward per POSE_BATCH crops — a 256x192-crop ViT is far
    too small to saturate a GPU at one-frame-per-forward.

    Returns {frame_idx: [{"keypoints": (17,2), "scores": (17,)}, ...]}.
    List is aligned with detections[frame_idx] — pose_data[i][j] corresponds
    to detections[i][j].
    """
    frame_dir = Path(source_path) / "img1"
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No jpg frames found in {frame_dir}")

    loader = torch.utils.data.DataLoader(
        _FrameDataset(frame_files), batch_size=None, num_workers=4, prefetch_factor=4,
    )

    pose_data: Dict[int, list] = {}
    pending: List[dict] = []
    pending_crops = 0

    for frame_idx, image in tqdm(loader, total=len(frame_files), desc="Pose"):
        frame_idx = int(frame_idx)
        dets = detections.get(frame_idx)
        if dets is None or len(dets) == 0:
            pose_data[frame_idx] = []
            continue

        boxes_xyxy = dets[:, :4]
        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        valid = (widths > 0) & (heights > 0)

        if not valid.any():
            pose_data[frame_idx] = [None] * len(dets)
            continue

        pending.append({
            "frame_idx": frame_idx,
            "image": image.numpy() if isinstance(image, torch.Tensor) else image,
            "boxes_xyxy": boxes_xyxy[valid],
            "valid": valid,
            "n_dets": len(dets),
        })
        pending_crops += int(valid.sum())

        if pending_crops >= POSE_BATCH:
            _flush_pose_batch(pending, processor, model, pose_data)
            pending, pending_crops = [], 0

    _flush_pose_batch(pending, processor, model, pose_data)
    return pose_data


def is_legible(
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_confidence: float = 0.5,
) -> bool:
    """Check if all 4 torso keypoints are confident enough for OCR."""
    torso_scores = scores[TORSO_KP_INDICES]
    return bool((torso_scores > min_confidence).all())


def crop_torso(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    padding: int = 5,
    min_confidence: float = 0.3,
    min_visible: int = 3,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Crop torso region defined by shoulder+hip keypoints.

    Padding on left, right, and bottom only (per Koshkina & Elder 2024).
    Returns (BGR crop, (x1, y1, x2, y2)) or (None, None).
    """
    torso_kps = keypoints[TORSO_KP_INDICES]
    torso_scores = scores[TORSO_KP_INDICES]

    visible = torso_scores > min_confidence
    if visible.sum() < min_visible:
        return None, None

    pts = torso_kps[visible]
    x1 = int(pts[:, 0].min()) - padding
    y1 = int(pts[:, 1].min())
    x2 = int(pts[:, 0].max()) + padding
    y2 = int(pts[:, 1].max()) + padding

    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
