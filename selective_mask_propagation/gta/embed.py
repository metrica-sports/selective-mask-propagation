"""Embedding aggregation: per-detection → per-track.

Deep-EIoU already has per-detection OSNet embeddings from the detect step.
This step aggregates them by canonical track ID — no model, no image loading.
"""

from typing import Dict

import numpy as np


def aggregate_embeddings(
    embeddings: Dict[int, np.ndarray],
    assignments: Dict[int, Dict[int, int]],
) -> Dict[int, np.ndarray]:
    """Aggregate per-detection embeddings by track ID.

    For each frame, looks up which detection belongs to which track (via
    assignments), collects that detection's embedding, and stacks all
    embeddings for each track into a single array.

    Returns {track_id: np.ndarray of shape (N, 512)} where N is the
    number of frames the track appears in. Each row is the raw embedding
    (already L2-normalized from the OSNet extractor).
    """
    track_features: Dict[int, list] = {}
    for frame_idx, frame_assignments in assignments.items():
        frame_embs = embeddings.get(frame_idx)
        if frame_embs is None or len(frame_embs) == 0:
            continue
        for det_idx, track_id in frame_assignments.items():
            if det_idx < len(frame_embs):
                feat = frame_embs[det_idx]
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat = feat / norm
                track_features.setdefault(track_id, []).append(feat)

    result = {tid: np.stack(feats) for tid, feats in track_features.items() if feats}
    print(f"Embed: {len(result)} tracks, {sum(v.shape[0] for v in result.values())} total embeddings")
    return result
