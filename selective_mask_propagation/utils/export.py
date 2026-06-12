"""Export tracking results to MOT challenge format."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def export_mot(
    tracks: Dict[int, Dict[int, np.ndarray]],
    output_path: str,
    quiet: bool = False,
) -> None:
    """Export tracks to MOT challenge format.

    tracks: {frame_idx: {track_id: xyxy_bbox}}.
    MOT format: frame_id,track_id,x,y,w,h,conf,-1,-1,-1 (1-indexed frames).
    """
    lines = []
    for frame_idx in sorted(tracks.keys()):
        frame_id = frame_idx + 1
        for track_id, bbox in tracks[frame_idx].items():
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame_id},{track_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},1,-1,-1,-1")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")
    if not quiet:
        print(f"MOT export: {len(lines)} entries -> {output_path}")


def parse_mot(mot_path: str) -> Dict[int, Dict[int, np.ndarray]]:
    """Parse MOT file back into tracks dict.

    MOT format: frame_id,track_id,x,y,w,h,conf,-1,-1,-1 (1-indexed frames).
    Returns {frame_idx: {track_id: xyxy_bbox}} (0-indexed frames).
    """
    data = _load_mot(mot_path)
    tracks: Dict[int, Dict[int, np.ndarray]] = {}
    for frame_idx, track_id, x, y, w, h in data:
        bbox = np.array([x, y, x + w, y + h])
        tracks.setdefault(frame_idx, {})[track_id] = bbox
    return tracks


def parse_mot_tracks(mot_path: str) -> Dict[int, List[Tuple[int, list]]]:
    """Parse MOT file into per-track entries.

    Returns {track_id: [(frame_idx, [x, y, w, h]), ...]} sorted by frame.
    Frame indices are 0-based.
    """
    data = _load_mot(mot_path)
    tracks: Dict[int, list] = {}
    for frame_idx, track_id, x, y, w, h in data:
        tracks.setdefault(track_id, []).append((frame_idx, [x, y, w, h]))
    for entries in tracks.values():
        entries.sort(key=lambda e: e[0])
    return tracks


def _load_mot(mot_path: str) -> List[Tuple[int, int, float, float, float, float]]:
    """Load MOT file into list of (frame_idx, track_id, x, y, w, h)."""
    raw = np.genfromtxt(mot_path, dtype=float, delimiter=",")
    return [(int(r[0]) - 1, int(r[1]), r[2], r[3], r[4], r[5]) for r in raw]
