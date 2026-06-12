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
from PIL import Image
from tqdm import tqdm

TORSO_KP_INDICES = [5, 6, 11, 12]  # left_shoulder, right_shoulder, left_hip, right_hip

VITPOSE_MODEL = "usyd-community/vitpose-plus-base"


def _run_pose(image, boxes_xyxy: np.ndarray, processor, model) -> List[dict]:
    """Run ViTPose on an image with given bounding boxes.

    Returns list of {"keypoints": (17,2), "scores": (17,)} per box.
    """
    if len(boxes_xyxy) == 0:
        return []

    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    inputs = processor(image, boxes=[boxes_xywh], return_tensors="pt").to(model.device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
        inputs = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in inputs.items()}
        dataset_index = torch.zeros(len(boxes_xyxy), dtype=torch.long, device=model.device)
        outputs = model(**inputs, dataset_index=dataset_index)

    if outputs.heatmaps.dtype == torch.bfloat16:
        outputs.heatmaps = outputs.heatmaps.float()

    pose_results = processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])

    results = []
    for pose_result in pose_results[0]:
        results.append({
            "keypoints": pose_result["keypoints"].cpu().numpy(),
            "scores": pose_result["scores"].cpu().numpy(),
        })
    return results


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


def estimate_all_poses(
    source_path: str,
    detections: Dict[int, np.ndarray],
    processor,
    model,
) -> Dict[int, list]:
    """Run ViTPose on every detection bbox in every frame.

    Algorithm:
        1. For each frame, extract detection bboxes (xyxy)
        2. Filter degenerate boxes (zero width/height)
        3. Run ViTPose batch inference
        4. Map results back to original detection indices

    Returns {frame_idx: [{"keypoints": (17,2), "scores": (17,)}, ...]}.
    List is aligned with detections[frame_idx] — pose_data[i][j] corresponds
    to detections[i][j].
    """
    total_frames, frame_gen = _frame_generator(source_path)
    pose_data: Dict[int, list] = {}

    for frame_idx, frame in enumerate(tqdm(frame_gen, total=total_frames, desc="Pose")):
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

        valid_boxes = boxes_xyxy[valid]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = _run_pose(image, valid_boxes, processor, model)

        frame_poses = []
        result_idx = 0
        for i in range(len(dets)):
            if valid[i]:
                frame_poses.append(results[result_idx])
                result_idx += 1
            else:
                frame_poses.append(None)
        pose_data[frame_idx] = frame_poses

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
