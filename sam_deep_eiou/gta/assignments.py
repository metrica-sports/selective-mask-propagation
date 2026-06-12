"""Build detection-index-to-track-id mappings.

Maps each detection on each frame to the track that owns it, by matching
track bboxes back to detection bboxes. The tracker assigns detections to
tracks but doesn't expose the det_idx→track_id mapping directly.
"""

from typing import Dict

import numpy as np


def build_assignments(
    tracks: Dict[int, Dict[int, np.ndarray]],
    detections: Dict[int, np.ndarray],
) -> Dict[int, Dict[int, int]]:
    """Match track bboxes to detection indices by coordinate proximity.

    For each frame, each track's bbox is matched to the closest detection
    bbox (xyxy). The tracker's bbox IS the detection bbox (possibly with
    minor floating-point differences from internal format conversions).

    Returns {frame_idx: {det_idx: track_id}}.
    """
    assignments: Dict[int, Dict[int, int]] = {}
    for frame_idx, frame_tracks in tracks.items():
        frame_dets = detections.get(frame_idx)
        if frame_dets is None or len(frame_dets) == 0:
            continue
        det_coords = frame_dets[:, :4]
        frame_assignments: Dict[int, int] = {}
        for track_id, bbox in frame_tracks.items():
            diffs = np.abs(det_coords - bbox).sum(axis=1)
            best_idx = int(diffs.argmin())
            if diffs[best_idx] < 1.0:
                frame_assignments[best_idx] = track_id
        if frame_assignments:
            assignments[frame_idx] = frame_assignments
    return assignments
