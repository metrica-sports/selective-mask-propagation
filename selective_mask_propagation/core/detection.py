"""Detection and embedding extraction (online or precomputed)."""

import configparser
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def step_detect(
    source_path: str,
    precomputed: bool,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    if precomputed:
        return detect_precomputed(source_path)
    return detect_online(source_path)


def detect_online(
    source_path: str,
    device: torch.device = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Run YOLOX + OSNet on a sequence.

    Returns (detections, embeddings):
        detections: {frame_idx: (N, 5) ndarray [x1,y1,x2,y2,score]}
        embeddings: {frame_idx: (N, 512) ndarray}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from ..yolox.inference import build_model as build_yolox, detect_sequence
    from ..osnet.inference import build_model as build_osnet, extract_embeddings

    yolox = build_yolox(device)
    detections = detect_sequence(yolox, source_path, device)
    del yolox

    osnet = build_osnet(device)
    embeddings = extract_embeddings(osnet, source_path, detections, device)
    del osnet

    return detections, embeddings


def detect_precomputed(
    source_path: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Load precomputed detections from det/det.txt and embeddings from emb/emb.npy.

    Returns (detections, embeddings):
        detections: {frame_idx: (N, 5) ndarray [x1,y1,x2,y2,score]}
        embeddings: {frame_idx: (N, 512) ndarray}
    """
    detections = _load_det_txt(source_path)
    embeddings = _load_emb_npy(source_path)
    return detections, embeddings


def _load_det_txt(source_path: str) -> Dict[int, np.ndarray]:
    """Load precomputed YOLOX detections from det/det.txt."""
    det_path = Path(source_path) / "det" / "det.txt"
    if not det_path.exists():
        raise FileNotFoundError(f"No detections found: {det_path}")

    frame_boxes = defaultdict(list)
    for line in det_path.read_text().strip().split("\n"):
        parts = line.split(",")
        frame_idx = int(parts[0]) - 1
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        score = float(parts[6])
        frame_boxes[frame_idx].append([x, y, x + w, y + h, score])

    total_frames = _get_seq_length(source_path, frame_boxes)

    result: Dict[int, np.ndarray] = {}
    for frame_idx in range(total_frames):
        dets = frame_boxes.get(frame_idx, [])
        if not dets:
            result[frame_idx] = np.empty((0, 5))
        else:
            result[frame_idx] = np.array(dets, dtype=np.float32)

    total = sum(len(d) for d in result.values())
    print(f"Precomputed detections: {total} boxes across {total_frames} frames")
    return result


def _load_emb_npy(source_path: str) -> Dict[int, np.ndarray]:
    """Load precomputed OSNet embeddings from emb/emb.npy."""
    emb_path = Path(source_path) / "emb" / "emb.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"No embeddings found: {emb_path}")

    raw = np.load(str(emb_path), allow_pickle=True)
    result: Dict[int, np.ndarray] = {}
    for i, frame_embs in enumerate(raw):
        if len(frame_embs) == 0:
            result[i] = np.empty((0, 512))
        else:
            result[i] = np.array(frame_embs).squeeze(1)

    total = sum(len(e) for e in result.values())
    print(f"Precomputed embeddings: {total} vectors across {len(result)} frames")
    return result


def _get_seq_length(source_path: str, frame_data: dict) -> int:
    """Get sequence length from seqinfo.ini, falling back to max frame index."""
    seqinfo_path = Path(source_path) / "seqinfo.ini"
    if seqinfo_path.exists():
        cfg = configparser.ConfigParser()
        cfg.read(seqinfo_path)
        return int(cfg["Sequence"]["seqLength"])
    return max(frame_data.keys()) + 1 if frame_data else 0
