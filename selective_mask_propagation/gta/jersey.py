"""Jersey number recognition: torso crop → PARSeq OCR → vote aggregation.

Two entry points:
  - run_jersey_ocr: detection-level OCR, runs once (shared by DE and SDE)
  - aggregate_jersey_numbers: per-track vote, runs per variant

Ported from SAM-SORT: jersey.py + vote.py
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from .pose import crop_torso, is_legible

MIN_OCR_CONFIDENCE = 0.9
MIN_READ_COUNT = 4
CROP_OCCLUSION_THRESHOLD = 0.30


def _crop_occluded(crop_bbox, det_idx, dets, threshold=CROP_OCCLUSION_THRESHOLD):
    """Check if another player is in front and overlapping our crop region.

    Two conditions must both be true to skip:
    1. IoA >= threshold (another detection covers our crop)
    2. The other detection's bbox bottom is below ours (higher y2 = closer
       to camera in elevated sports broadcasts = in front of us)

    When our player is in front (our y2 is lower), the crop shows our
    jersey clearly even though bboxes overlap — so we keep the read.
    """
    cx1, cy1, cx2, cy2 = crop_bbox
    crop_area = (cx2 - cx1) * (cy2 - cy1)
    if crop_area <= 0:
        return True
    our_bottom = dets[det_idx][3]
    for i, det in enumerate(dets):
        if i == det_idx:
            continue
        dx1, dy1, dx2, dy2 = det[:4]
        # Only consider detections that are in front of us
        if dy2 <= our_bottom:
            continue
        ix1 = max(cx1, dx1)
        iy1 = max(cy1, dy1)
        ix2 = min(cx2, dx2)
        iy2 = min(cy2, dy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter / crop_area >= threshold:
            return True
    return False


class _CropDataset(torch.utils.data.Dataset):
    """Worker-side decode → torso crop → occlusion filter → PARSeq transform.

    One item per frame that has at least one legible pose. The expensive
    per-frame CPU work (JPEG decode, crop, PIL bicubic resize) runs in
    DataLoader workers so the main loop only sees ready crop tensors.
    """

    def __init__(self, entries: List[Tuple[int, List[int]]], frame_files: List[Path],
                 detections: Dict[int, np.ndarray], pose_data: Dict[int, list],
                 img_size: Tuple[int, int]):
        self.entries = entries
        self.frame_files = frame_files
        self.detections = detections
        self.pose_data = pose_data
        self.transform = T.Compose([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        frame_idx, candidates = self.entries[i]
        frame = cv2.imread(str(self.frame_files[frame_idx]))
        dets = self.detections[frame_idx]
        poses = self.pose_data[frame_idx]

        crop_indices = []
        crop_tensors = []
        raw_crops = []
        for det_idx in candidates:
            pose = poses[det_idx]
            crop, crop_bbox = crop_torso(frame, pose["keypoints"], pose["scores"])
            if crop is None:
                continue
            # Skip if another detection overlaps the crop region
            if _crop_occluded(crop_bbox, det_idx, dets):
                continue
            raw_crops.append(crop)
            crop_rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_tensors.append(self.transform(crop_rgb))
            crop_indices.append(det_idx)

        return frame_idx, crop_indices, crop_tensors, raw_crops


def run_jersey_ocr(
    source_path: str,
    detections: Dict[int, np.ndarray],
    pose_data: Dict[int, list],
    parseq: Tuple,
) -> Tuple[Dict[int, list], Dict[int, dict]]:
    """Run jersey OCR on all legible detections.

    Purely detection-level — no track assignments needed. Every detection
    with a legible pose gets OCR'd. Track mapping happens downstream in
    aggregate_jersey_numbers.

    A metadata sweep over pose_data picks the frames worth decoding, then
    DataLoader workers decode and prepare crop tensors while the main loop
    runs PARSeq one frame-batch at a time (batch composition — and thus
    output — is identical to the previous per-frame loop).

    Args:
        parseq: (model, tokenizer) from gta.models.get_parseq().

    Returns (ocr_results, crops):
        ocr_results: {frame_idx: [{"label": str, "confidence": float} | None, ...]}
        crops: {frame_idx: {det_idx: np.ndarray}} — raw torso crops for debug
    """
    parseq_model, tokenizer = parseq
    device = next(parseq_model.parameters()).device
    img_size = parseq_model.encoder.patch_embed.img_size

    frame_dir = Path(source_path) / "img1"
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No jpg frames found in {frame_dir}")

    ocr_results: Dict[int, list] = {}
    all_crops: Dict[int, dict] = {}

    entries: List[Tuple[int, List[int]]] = []
    for frame_idx in range(len(frame_files)):
        dets = detections.get(frame_idx)
        if dets is None or len(dets) == 0:
            ocr_results[frame_idx] = []
            continue
        ocr_results[frame_idx] = [None] * len(dets)
        poses = pose_data.get(frame_idx, [])
        candidates = [det_idx for det_idx, pose in enumerate(poses)
                      if pose is not None and is_legible(pose["keypoints"], pose["scores"])]
        if candidates:
            entries.append((frame_idx, candidates))

    loader = torch.utils.data.DataLoader(
        _CropDataset(entries, frame_files, detections, pose_data, img_size),
        batch_size=None, num_workers=4, prefetch_factor=4,
    )

    for frame_idx, crop_indices, crop_tensors, raw_crops in tqdm(
        loader, total=len(entries), desc="Jersey OCR"
    ):
        if not crop_indices:
            continue

        batch = torch.stack(crop_tensors).to(device)
        with torch.no_grad():
            logits = parseq_model(tokenizer, batch)
            probs = logits.softmax(-1)
            labels, confidences = tokenizer.decode(probs)

        frame_results = ocr_results[frame_idx]
        frame_crops = {}
        for i, det_idx in enumerate(crop_indices):
            crop = raw_crops[i]
            frame_crops[det_idx] = crop.numpy() if isinstance(crop, torch.Tensor) else crop
            char_confs = confidences[i]
            conf = char_confs.prod().item() if len(char_confs) > 0 else 0.0
            label = labels[i]

            if not label.isdigit() or not (1 <= int(label) <= 99) or int(label) in (1, 7):
                label = ""
                conf = 0.0

            frame_results[det_idx] = {
                "label": label,
                "confidence": conf,
            }

        all_crops[frame_idx] = frame_crops

    return ocr_results, all_crops


def aggregate_jersey_numbers(
    ocr_results: Dict[int, list],
    assignments: Dict[int, Dict[int, int]],
) -> Dict[int, dict]:
    """Aggregate per-detection OCR into per-tracklet jersey numbers.

    Algorithm:
        1. Collect all OCR reads per track (via assignments)
        2. Filter reads below MIN_OCR_CONFIDENCE
        3. Majority vote: most frequent number wins
        4. Require MIN_READ_COUNT reads to accept

    Returns {track_id: {"number": int, "detail": {...}}}.
    Number is -1 for illegible tracklets.
    """
    tracklet_preds: Dict[int, List[List[float]]] = {}
    for frame_idx, frame_ocr in ocr_results.items():
        frame_assignments = assignments.get(frame_idx, {})
        for det_idx, ocr in enumerate(frame_ocr):
            if ocr is None or not ocr["label"]:
                continue
            track_id = frame_assignments.get(det_idx)
            if track_id is None:
                continue
            tracklet_preds.setdefault(track_id, []).append(
                [int(ocr["label"]), ocr["confidence"]]
            )

    jersey_map: Dict[int, dict] = {}
    for track_id, preds in tracklet_preds.items():
        arr = np.array(preds)
        number, detail = _vote_count(arr)
        jersey_map[track_id] = {"number": number, "detail": detail}

    assigned = sum(1 for v in jersey_map.values() if v["number"] != -1)
    print(f"Jersey vote: {assigned}/{len(jersey_map)} tracklets assigned a number")
    return jersey_map


def _vote_count(predictions: np.ndarray) -> Tuple[int, dict]:
    """Majority vote with confidence floor and minimum count."""
    mask = predictions[:, 1] >= MIN_OCR_CONFIDENCE
    filtered = predictions[mask]

    if len(filtered) == 0:
        return -1, {"n_raw": len(predictions), "n_filtered": 0}

    counts = Counter(int(v) for v in filtered[:, 0])
    best_number, best_count = counts.most_common(1)[0]

    detail = {
        "n_raw": len(predictions),
        "n_filtered": len(filtered),
        "candidates": {str(k): v for k, v in counts.most_common()},
    }

    if best_count < MIN_READ_COUNT:
        return -1, detail

    return best_number, detail
