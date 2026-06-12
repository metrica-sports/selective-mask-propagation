"""Hierarchical tracklet matching.

Two-tier merge:
  1. Jersey + team (deterministic): player tracklets with same confident
     jersey number and same team are merged unconditionally.
  2. Appearance: greedy agglomerative clustering on OSNet cosine distance.
     Vetoes (in order): temporal overlap, opposite-edge exit/entry,
     different teams, conflicting jersey numbers.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..utils.export import parse_mot_tracks

OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def _edge(bbox: list, width: int, height: int) -> str:
    """Which frame edge is this bbox center closest to."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    distances = {"left": cx, "right": width - cx, "top": cy, "bottom": height - cy}
    return min(distances, key=distances.get)


def _check_edges(trk1: dict, trk2: dict, width: int, height: int) -> bool:
    """Check that no exit/entry transition crosses opposite edges.

    Walks all detections from both tracklets in time order. At every
    source transition, the exit edge and entry edge must not be opposite
    (e.g. exit-left → enter-right is blocked).
    """
    pairs = [(t, b, 0) for t, b in zip(trk1["times"], trk1["bboxes"])]
    pairs += [(t, b, 1) for t, b in zip(trk2["times"], trk2["bboxes"])]
    pairs.sort(key=lambda e: e[0])

    prev_bbox = pairs[0][1]
    prev_src = pairs[0][2]
    for _, bbox, src in pairs[1:]:
        if src != prev_src:
            exit_edge = _edge(prev_bbox, width, height)
            entry_edge = _edge(bbox, width, height)
            if entry_edge == OPPOSITE[exit_edge]:
                return False
        prev_bbox = bbox
        prev_src = src
    return True


def _jersey_merge(
    tracklets: Dict[int, dict],
    embeddings: Dict[int, np.ndarray],
    jersey_map: Dict[int, dict],
    track_teams: Dict[int, dict],
) -> Tuple[Dict[int, dict], Dict[int, np.ndarray], Dict[int, int]]:
    """Merge player tracklets with same confident jersey number and same team."""
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for tid in list(tracklets.keys()):
        team_info = track_teams.get(tid)
        if team_info is None or team_info["team_id"] is None:
            continue
        jersey_info = jersey_map.get(tid)
        if jersey_info is None or jersey_info["number"] == -1:
            continue
        groups[(team_info["team_id"], jersey_info["number"])].append(tid)

    absorbed = {}
    n_merges = 0
    for (_team_id, jersey_num), tids in groups.items():
        if len(tids) < 2:
            continue
        tids.sort()
        keeper = tids[0]
        keeper_times = set(tracklets[keeper]["times"])
        for tid in tids[1:]:
            if keeper_times & set(tracklets[tid]["times"]):
                continue
            keeper_times.update(tracklets[tid]["times"])
            tracklets[keeper]["times"] += tracklets[tid]["times"]
            tracklets[keeper]["bboxes"] += tracklets[tid]["bboxes"]
            if keeper in embeddings and tid in embeddings:
                embeddings[keeper] = np.concatenate([embeddings[keeper], embeddings[tid]])
            absorbed[tid] = keeper
            del tracklets[tid]
            if tid in embeddings:
                del embeddings[tid]
            n_merges += 1

    print(f"Jersey merge: {n_merges} merges across {len([g for g in groups.values() if len(g) > 1])} groups")
    return tracklets, embeddings, absorbed


def _cosine_distance(feats1: np.ndarray, feats2: np.ndarray) -> float:
    """Mean pairwise cosine distance between two L2-normalized feature sets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t1 = torch.tensor(feats1, dtype=torch.float32, device=device)
    t2 = torch.tensor(feats2, dtype=torch.float32, device=device)
    cos_dist = 1 - torch.matmul(t1, t2.T)
    return float(cos_dist.sum() / (len(t1) * len(t2)))


def _get_distance(
    tid1: int, tid2: int,
    tracklets: dict, embeddings: dict,
    jersey_map: Dict[int, dict], track_teams: Dict[int, dict],
    width: int, height: int,
) -> float:
    """Cosine distance with vetoes.

    Returns 1.0 (blocked) if any veto fires:
      1. Temporal overlap
      2. Opposite-edge exit/entry
      3. Different teams (player vs player)
      4. Conflicting confident jersey numbers
    """
    if tid1 != tid2 and set(tracklets[tid1]["times"]) & set(tracklets[tid2]["times"]):
        return 1.0

    if not _check_edges(tracklets[tid1], tracklets[tid2], width, height):
        return 1.0

    t1 = (track_teams.get(tid1) or {}).get("team_id")
    t2 = (track_teams.get(tid2) or {}).get("team_id")
    if t1 is not None and t2 is not None:
        if t1 != t2:
            return 1.0
        j1 = jersey_map.get(tid1)
        j2 = jersey_map.get(tid2)
        if (j1 is not None and j1["number"] != -1
                and j2 is not None and j2["number"] != -1
                and j1["number"] != j2["number"]):
            return 1.0

    return _cosine_distance(embeddings[tid1], embeddings[tid2])


def _distance_matrix(
    track_ids: list, tracklets: dict, embeddings: dict,
    jersey_map: Dict[int, dict], track_teams: Dict[int, dict],
    width: int, height: int,
) -> np.ndarray:
    n = len(track_ids)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = _get_distance(
                track_ids[i], track_ids[j], tracklets, embeddings,
                jersey_map, track_teams, width, height,
            )
            dist[i][j] = d
            dist[j][i] = d
    return dist


def match_tracklets(
    mot_path: str,
    track_embeddings: Dict[int, np.ndarray],
    jersey_map: Dict[int, dict],
    track_teams: Dict[int, dict],
    width: int,
    height: int,
    merge_dist_thres: float = 0.45,
) -> Tuple[Dict[int, int], list]:
    """Hierarchical tracklet matching: jersey merge then appearance clustering.

    Algorithm:
        1. Jersey merge: group player tracklets by (team_id, jersey_number),
           merge each group into the earliest canonical ID.
        2. Appearance merge: greedy agglomerative clustering on cosine distance.
           All vetoes (temporal, spatial, team, jersey) are baked into the
           distance function — blocked pairs get distance 1.0.
        3. Build global ID map: global_id = min(canonical_ids) per merge group.

    Returns (canonical_to_gta, merge_events).
    """
    raw = parse_mot_tracks(mot_path)
    tracklets = {
        tid: {"times": [t for t, _ in entries], "bboxes": [b for _, b in entries]}
        for tid, entries in raw.items() if tid in track_embeddings
    }
    n_initial = len(tracklets)
    embeddings = {tid: feats.copy() for tid, feats in track_embeddings.items()
                  if tid in tracklets}

    tracklets, embeddings, jersey_absorbed = _jersey_merge(
        tracklets, embeddings, jersey_map, track_teams,
    )

    idx2tid = dict(enumerate(tracklets.keys()))
    dist = _distance_matrix(
        list(tracklets.keys()), tracklets, embeddings,
        jersey_map, track_teams, width, height,
    )

    appearance_absorbed = {}
    merge_events = []
    groups = {tid: [tid] for tid in tracklets}

    diagonal_mask = np.eye(dist.shape[0], dtype=bool)
    non_diagonal_mask = ~diagonal_mask

    while np.any(dist[non_diagonal_mask] < merge_dist_thres):
        min_index = np.argmin(dist[non_diagonal_mask])
        masked_indices = np.where(non_diagonal_mask)
        idx1, idx2 = masked_indices[0][min_index], masked_indices[1][min_index]

        tid1, tid2 = idx2tid[idx1], idx2tid[idx2]

        merge_events.append({
            "kept_id": tid1,
            "absorbed_id": tid2,
            "kept_members": list(groups[tid1]),
            "absorbed_members": list(groups[tid2]),
            "distance": round(float(dist[idx1, idx2]), 4),
        })

        tracklets[tid1]["times"] += tracklets[tid2]["times"]
        tracklets[tid1]["bboxes"] += tracklets[tid2]["bboxes"]
        embeddings[tid1] = np.concatenate([embeddings[tid1], embeddings[tid2]])

        appearance_absorbed[tid2] = tid1
        groups[tid1] = groups[tid1] + groups[tid2]
        del groups[tid2]
        del tracklets[tid2]
        del embeddings[tid2]

        dist = np.delete(dist, idx2, axis=0)
        dist = np.delete(dist, idx2, axis=1)
        idx2tid = dict(enumerate(tracklets.keys()))

        for idx in range(dist.shape[0]):
            d = _get_distance(
                idx2tid[idx1], idx2tid[idx], tracklets, embeddings,
                jersey_map, track_teams, width, height,
            )
            dist[idx1, idx] = d
            dist[idx, idx1] = d

        diagonal_mask = np.eye(dist.shape[0], dtype=bool)
        non_diagonal_mask = ~diagonal_mask

    all_absorbed = {**jersey_absorbed, **appearance_absorbed}

    def _resolve_root(tid: int) -> int:
        while tid in all_absorbed:
            tid = all_absorbed[tid]
        return tid

    all_canonical_ids = set(track_embeddings.keys())
    root_groups: Dict[int, List[int]] = defaultdict(list)
    for cid in all_canonical_ids:
        root_groups[_resolve_root(cid)].append(cid)

    canonical_to_gta = {}
    for members in root_groups.values():
        global_id = min(members)
        for cid in members:
            canonical_to_gta[cid] = global_id

    n_jersey = len(jersey_absorbed)
    n_appearance = len(appearance_absorbed)
    n_after = len(tracklets)
    print(f"Match: {n_initial} → {n_after} tracklets "
          f"({n_jersey} jersey + {n_appearance} appearance merges)")

    return canonical_to_gta, merge_events


def save_mot_remapped(
    output_path: str, canonical_to_gta: Dict[int, int], mot_path: str,
) -> None:
    """Write MOT file by remapping track IDs."""
    with open(mot_path) as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        parts = line.strip().split(",")
        if not parts or not parts[0]:
            continue
        canonical_id = int(parts[1])
        parts[1] = str(canonical_to_gta.get(canonical_id, canonical_id))
        rows.append((int(parts[0]), ",".join(parts)))

    rows.sort(key=lambda x: x[0])
    with open(output_path, "w") as f:
        for _, row in rows:
            f.write(row + "\n")
