"""Linear bbox interpolation for track gaps."""

from typing import Dict, Optional, Set

import numpy as np


N_MIN = 5       # Minimum track length to consider interpolation
N_DTI = 15      # Maximum gap length (frames) to interpolate
DIST_THRESH = 200  # Maximum center-to-center distance for interpolation


def interpolate_tracks(
    tracks: Dict[int, Dict[int, np.ndarray]],
    skip_ids: Optional[Set[int]] = None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Fill track gaps with linearly interpolated bboxes.

    Scans each track for gaps of 2..N_DTI frames. If the track has at
    least N_MIN total frames and the endpoint bboxes are within
    DIST_THRESH pixels (center-to-center), fills each missing frame
    with a linearly interpolated xyxy bbox.

    Args:
        tracks: {frame_idx: {track_id: xyxy_bbox}} — original track data.

    Returns:
        Sparse dict of only the interpolated entries:
        {frame_idx: {track_id: xyxy_bbox}}.
    """
    track_frames: Dict[int, list] = {}
    for frame_idx, frame_tracks in tracks.items():
        for track_id, bbox in frame_tracks.items():
            track_frames.setdefault(track_id, []).append((frame_idx, bbox))

    interp: Dict[int, Dict[int, np.ndarray]] = {}
    total = 0

    for track_id, entries in track_frames.items():
        if skip_ids and track_id in skip_ids:
            continue
        if len(entries) < N_MIN:
            continue
        entries.sort(key=lambda e: e[0])

        for i in range(1, len(entries)):
            left_frame, left_bbox = entries[i - 1]
            right_frame, right_bbox = entries[i]
            gap = right_frame - left_frame
            if gap < 2 or gap > N_DTI:
                continue

            lx = (left_bbox[0] + left_bbox[2]) / 2
            ly = (left_bbox[1] + left_bbox[3]) / 2
            rx = (right_bbox[0] + right_bbox[2]) / 2
            ry = (right_bbox[1] + right_bbox[3]) / 2
            dist = ((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5
            if dist > DIST_THRESH:
                continue

            for j in range(1, gap):
                t = j / gap
                bbox = left_bbox + t * (right_bbox - left_bbox)
                frame_idx = left_frame + j
                interp.setdefault(frame_idx, {})[track_id] = bbox
                total += 1

    print(f"Interpolation: {total} frames filled across {len(track_frames)} tracks")
    return interp
