"""Merge step: combine Deep-EIoU tracks with SAM evidence.

Two modes, two answers to "what is the output while SAM is active?":

Benchmark mode (step_merge + extract_bboxes) — identity from SAM,
geometry from Deep-EIoU. Only SWAP windows modify the output, and even
then the exported boxes are the DE boxes the mask settled into
(via match_history), never mask-derived boxes: tight mask boxes cover
only visible pixels and are systematically IoU-punished against
full-extent ground-truth annotations. All paper numbers use this mode.

Prod mode (step_merge_prod + derive_bboxes) — the mask is ground truth
while a healthy window (SWAP/CLEAN/EDGE/END) is open: it replaces the
DE box from entry to exit. During CLEAN windows DE can scramble
identities mid-occlusion even though they resolve by exit; prod mode
has the correct player throughout, which is what downstream consumers
(pose, event attribution) need, at the cost of benchmark score.

In both modes, renames from SWAP windows are the single source of
post-exit identity continuity.
"""

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from .sam2 import SamWindow, WindowOutcome

HEALTHY_OUTCOMES = {
    WindowOutcome.SWAP,
    WindowOutcome.CLEAN,
    WindowOutcome.EDGE,
    WindowOutcome.END,
}


@dataclass
class TrackData:
    bbox: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


RAW_NAMESPACE = "raw"
CANON_NAMESPACE = "canon"
MergedKey = Tuple[str, int]


def raw_key(track_id: int) -> MergedKey:
    return (RAW_NAMESPACE, track_id)


def canon_key(track_id: int) -> MergedKey:
    return (CANON_NAMESPACE, track_id)


def _parse_merged_key(key: Hashable) -> Optional[MergedKey]:
    """Parse merged key, supporting both namespaced and legacy int keys."""
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str) and isinstance(key[1], int):
        if key[0] in {RAW_NAMESPACE, CANON_NAMESPACE}:
            return key[0], key[1]
        return None
    if isinstance(key, int):
        # Backward compatibility for older merged artifacts.
        return CANON_NAMESPACE, key
    return None


def step_merge(
    tracks: Dict[int, Dict[int, np.ndarray]],
    sam_masks: Dict[int, Dict[int, np.ndarray]],
    windows: List[SamWindow],
    margins: Dict[int, Dict[int, float]],
    rename_events: List[Tuple[int, int, int]],
) -> Tuple[Dict[int, Dict[MergedKey, TrackData]], Dict[int, Dict[int, float]], Dict[int, List[Tuple[int, int]]]]:
    """Merge Deep-EIoU tracks with SAM evidence.

    Algorithm:
        1. Build base raw space: every Deep-EIoU frame/track under raw keys
        2. Overlay SWAP zones in canonical space (mask warmup + authoritative)
        3. Keep rename projection in extract_bboxes (single flattening point)

    Only SWAP windows get mask overlays. All other outcomes leave DE
    bboxes untouched.

    Returns (merged_tracks, renamed_margins, rename_map).
    """
    rename_map = _build_rename_map(rename_events)

    merged: Dict[int, Dict[MergedKey, TrackData]] = {}
    for frame_idx, frame_tracks in tracks.items():
        merged[frame_idx] = {raw_key(tid): TrackData(bbox=bbox) for tid, bbox in frame_tracks.items()}

    # Only SWAP windows get mask overlays. CLEAN/EDGE/END confirmed DE
    # was correct or had no identity resolution — DE bboxes pass through
    # untouched, guaranteeing SDE is never worse than DE on those frames.
    for w in windows:
        if w.outcome != WindowOutcome.SWAP:
            continue
        # Warmup: add mask alongside existing DE bbox (non-authoritative, for render only)
        for frame_idx in range(w.seed_frame + 1, w.entry_frame):
            mask = sam_masks.get(frame_idx, {}).get(w.canonical_id)
            if mask is None:
                continue
            frame_data = merged.setdefault(frame_idx, {})
            key = canon_key(w.canonical_id)
            existing = frame_data.get(key)
            bbox = existing.bbox if existing is not None else None
            frame_data[key] = TrackData(bbox=bbox, mask=mask)
        # Authoritative: mask replaces DE bbox
        for frame_idx in range(w.entry_frame, w.exit_frame + 1):
            mask = sam_masks.get(frame_idx, {}).get(w.canonical_id)
            if mask is None:
                continue
            frame_data = merged.setdefault(frame_idx, {})
            frame_data[canon_key(w.canonical_id)] = TrackData(mask=mask)

    renamed_margins = _apply_renames(margins, rename_map)

    print(f"Merge: {len(rename_events)} rename events applied")

    return merged, renamed_margins, rename_map


def step_merge_prod(
    tracks: Dict[int, Dict[int, np.ndarray]],
    sam_masks: Dict[int, Dict[int, np.ndarray]],
    windows: List[SamWindow],
    margins: Dict[int, Dict[int, float]],
    rename_events: List[Tuple[int, int, int]],
) -> Tuple[Dict[int, Dict[int, TrackData]], Dict[int, Dict[int, float]], Dict[int, List[Tuple[int, int]]]]:
    """Prod-mode merge: SAM masks are authoritative during healthy windows.

    Per player, per frame, binary state:
      - mask only:  SAM authoritative zone (entry to exit, healthy window)
      - bbox+mask:  SAM warmup zone (seed to entry)
      - bbox only:  DE mode (no window, or DEGRADED/STALE)

    Unlike benchmark mode there is no extract_bboxes/match_history pass:
    identity comes from the rename map applied directly to the base layer,
    and bbox geometry for mask-only entries is derived on demand
    (derive_bboxes).

    Returns (merged, renamed_margins, rename_map) with merged already
    flattened to {frame: {canonical_track_id: TrackData}}.
    """
    rename_map = _build_rename_map(rename_events)

    merged: Dict[int, Dict[int, TrackData]] = {}
    for frame_idx, frame_tracks in tracks.items():
        frame_data: Dict[int, TrackData] = {}
        for raw_tid, bbox in frame_tracks.items():
            canonical = _resolve(raw_tid, frame_idx, rename_map)
            frame_data[canonical] = TrackData(bbox=bbox)
        merged[frame_idx] = frame_data

    healthy = [w for w in windows if w.outcome in HEALTHY_OUTCOMES]
    for w in healthy:
        # Warmup: mask alongside DE bbox
        for frame_idx in range(w.seed_frame + 1, w.entry_frame):
            mask = sam_masks.get(frame_idx, {}).get(w.canonical_id)
            if mask is None:
                continue
            frame_data = merged.setdefault(frame_idx, {})
            existing = frame_data.get(w.canonical_id)
            bbox = existing.bbox if existing is not None else None
            frame_data[w.canonical_id] = TrackData(bbox=bbox, mask=mask)
        # Authoritative: mask replaces DE bbox
        for frame_idx in range(w.entry_frame, w.exit_frame + 1):
            mask = sam_masks.get(frame_idx, {}).get(w.canonical_id)
            if mask is None:
                continue
            merged.setdefault(frame_idx, {})[w.canonical_id] = TrackData(mask=mask)

    renamed_margins = _apply_renames(margins, rename_map)

    print(f"Merge (prod): {len(rename_events)} renames, {len(healthy)} healthy windows, "
          f"{len(windows) - len(healthy)} unhealthy")

    return merged, renamed_margins, rename_map


def derive_bboxes(
    merged: Dict[int, Dict[int, TrackData]],
) -> Dict[int, Dict[int, np.ndarray]]:
    """Flatten prod-mode merged tracks to {frame: {track_id: bbox}}.

    DE-mode entries return their bbox directly; mask-only entries derive a
    tight bbox from the mask pixels.
    """
    result: Dict[int, Dict[int, np.ndarray]] = {}
    for frame_idx, frame_data in merged.items():
        frame_bboxes: Dict[int, np.ndarray] = {}
        for tid, td in frame_data.items():
            if td.bbox is not None:
                frame_bboxes[tid] = td.bbox
            elif td.mask is not None and td.mask.any():
                frame_bboxes[tid] = _bbox_from_mask(td.mask)
        if frame_bboxes:
            result[frame_idx] = frame_bboxes
    return result


def _build_rename_map(
    rename_events: List[Tuple[int, int, int]],
) -> Dict[int, List[Tuple[int, int]]]:
    """Build frame-indexed rename map from run_sam's rename events.

    Events include both swaps and displacements — run_sam is the single
    authority on canonical ID assignment.

    Returns {raw_track_id: [(effective_frame, canonical_id), ...]} sorted by frame.
    """
    if not rename_events:
        return {}

    rename_map: Dict[int, List[Tuple[int, int]]] = {}
    for raw_id, canonical_id, effective_frame in rename_events:
        rename_map.setdefault(raw_id, []).append((effective_frame, canonical_id))

    for entries in rename_map.values():
        entries.sort()

    return rename_map


def _apply_renames(data: dict, rename_map: dict) -> dict:
    """Apply rename map to a {frame_idx: {track_id: value}} dict."""
    if not rename_map:
        return data

    corrected = {}
    for frame_idx, frame_data in data.items():
        defaults = {}
        explicit = {}
        for track_id, value in frame_data.items():
            canonical = _resolve(track_id, frame_idx, rename_map)
            if canonical == track_id:
                defaults[canonical] = value
            else:
                explicit[canonical] = value
        corrected[frame_idx] = {**defaults, **explicit}
    return corrected


def _resolve(track_id: int, frame: int, rename_map: dict) -> int:
    """Resolve track_id to canonical identity at given frame.

    Finds the latest rename entry with effective_frame <= frame.
    """
    entries = rename_map.get(track_id)
    if entries:
        best = None
        for ef, canonical in entries:
            if ef <= frame:
                best = canonical
            else:
                break
        if best is not None:
            return best
    return track_id


def _resolve_with_frame(track_id: int, frame: int, rename_map: dict) -> Tuple[int, Optional[int]]:
    """Resolve track_id and return (canonical_id, effective_frame|None)."""
    entries = rename_map.get(track_id)
    if entries:
        best_canonical = None
        best_frame = None
        for ef, canonical in entries:
            if ef <= frame:
                best_canonical = canonical
                best_frame = ef
            else:
                break
        if best_canonical is not None:
            return best_canonical, best_frame
    return track_id, None


def extract_bboxes(
    merged: Dict[int, Dict[Hashable, TrackData]],
    tracks: Dict[int, Dict[int, np.ndarray]],
    rename_map: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    match_history: Optional[Dict[int, Dict[int, int]]] = None,
    windows: Optional[List[SamWindow]] = None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Extract bboxes with SAM as the sole identity assignment source.

    No spatial re-matching is done in merge. `match_history` from run_sam
    provides the per-frame canonical->raw_track assignment used to materialize
    bbox geometry for mask entries.

    When assignment is missing (no history entry), fall back to existing bbox
    if present (warmup), otherwise tight mask bbox. Non-mask entries whose raw
    bbox has already been claimed by a mask are suppressed to avoid duplicates.
    Remaining DE bboxes are recovered in a final fallback pass.
    """
    result: Dict[int, Dict[int, np.ndarray]] = {}
    assignment_hits = 0
    assignment_fallbacks = 0
    suppressed_duplicates = 0
    fallback_count = 0
    rmap = rename_map or {}
    mhist = match_history or {}
    swap_windows = windows or []
    authoritative_ranges: Dict[int, List[Tuple[int, int]]] = {}
    for w in swap_windows:
        if w.outcome == WindowOutcome.SWAP:
            authoritative_ranges.setdefault(w.canonical_id, []).append((w.entry_frame, w.exit_frame))
    for ranges in authoritative_ranges.values():
        ranges.sort()
    all_ids = {tid for frame in tracks.values() for tid in frame}
    for raw_tid, entries in rmap.items():
        all_ids.add(raw_tid)
        for _, canonical in entries:
            all_ids.add(canonical)
    alias_stride = (max(all_ids) + 1) if all_ids else 1

    for frame_idx in sorted(set(merged.keys()) | set(tracks.keys())):
        frame_data = merged.get(frame_idx, {})
        raw_frame_tracks = tracks.get(frame_idx, {})
        frame_bboxes: Dict[int, np.ndarray] = {}
        claimed_raw_tids: set[int] = set()
        mask_entries: List[Tuple[int, TrackData]] = []

        for key, td in frame_data.items():
            parsed = _parse_merged_key(key)
            if parsed is None:
                continue
            namespace, track_id = parsed
            has_mask = td.mask is not None and td.mask.any()
            if namespace == CANON_NAMESPACE and has_mask:
                mask_entries.append((track_id, td))

        # Mask entries first: authoritative identity source.
        for canonical_id, td in sorted(mask_entries):
            # Warmup masks are render-only. They must not affect export IDs/bboxes.
            ranges = authoritative_ranges.get(canonical_id, [])
            in_authoritative = any(start <= frame_idx <= end for start, end in ranges)
            if not in_authoritative:
                continue
            raw_tid = mhist.get(canonical_id, {}).get(frame_idx)
            if raw_tid is not None and raw_tid in raw_frame_tracks and raw_tid not in claimed_raw_tids:
                frame_bboxes[canonical_id] = raw_frame_tracks[raw_tid]
                claimed_raw_tids.add(raw_tid)
                assignment_hits += 1
                continue
            if td.bbox is not None:
                frame_bboxes[canonical_id] = td.bbox
                raw_tid_from_bbox = _find_raw_tid_for_bbox(td.bbox, raw_frame_tracks, claimed_raw_tids)
                if raw_tid_from_bbox is not None:
                    claimed_raw_tids.add(raw_tid_from_bbox)
                assignment_fallbacks += 1
                continue
            best_raw_tid, best_bbox = _best_unclaimed_bbox(td.mask, raw_frame_tracks, claimed_raw_tids)
            if best_bbox is not None and best_raw_tid is not None:
                frame_bboxes[canonical_id] = best_bbox
                claimed_raw_tids.add(best_raw_tid)
            else:
                frame_bboxes[canonical_id] = _bbox_from_mask(td.mask)
            assignment_fallbacks += 1

        # Add remaining raw tracks. For canonical collisions, choose owner deterministically:
        # explicit rename (canonical != raw) > default, then newer effective_frame, then raw_id.
        remaining_groups: Dict[int, List[Tuple[int, np.ndarray, Optional[int], bool]]] = {}
        for raw_tid, bbox in raw_frame_tracks.items():
            if raw_tid in claimed_raw_tids:
                suppressed_duplicates += 1
                continue
            canonical, ef = _resolve_with_frame(raw_tid, frame_idx, rmap)
            remaining_groups.setdefault(canonical, []).append((raw_tid, bbox, ef, canonical != raw_tid))

        for canonical in sorted(remaining_groups):
            candidates = remaining_groups[canonical]
            candidates.sort(
                key=lambda item: (
                    1 if item[3] else 0,              # explicit over default
                    item[2] if item[2] is not None else -1,  # newer effective_frame wins
                    -item[0],                          # stable tie-break
                ),
                reverse=True,
            )
            for idx, (raw_tid, bbox, _, _) in enumerate(candidates):
                if idx == 0 and canonical not in frame_bboxes:
                    frame_bboxes[canonical] = bbox
                    continue
                if raw_tid not in frame_bboxes:
                    frame_bboxes[raw_tid] = bbox
                    fallback_count += 1
                else:
                    alias = raw_tid + alias_stride
                    frame_bboxes[alias] = bbox
                    fallback_count += 1

        if frame_bboxes:
            result[frame_idx] = frame_bboxes

    print(
        f"  extract_bboxes: {assignment_hits} assignment hits, "
        f"{assignment_fallbacks} assignment fallbacks"
    )
    if suppressed_duplicates:
        print(f"  extract_bboxes: {suppressed_duplicates} suppressed non-mask duplicates")
    if fallback_count:
        print(f"  extract_bboxes: {fallback_count} fallback recoveries")
    return result


def _best_unclaimed_bbox(
    mask: np.ndarray,
    raw_frame_tracks: Dict[int, np.ndarray],
    claimed_raw_tids: set[int],
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """Find the unclaimed DE bbox with highest IoMA to the mask. No threshold."""
    mask_area = int(mask.sum())
    if mask_area == 0:
        return None, None
    h, w = mask.shape
    best_ioma = 0.0
    best_bbox = None
    best_raw_tid = None
    for raw_tid, bbox in raw_frame_tracks.items():
        if raw_tid in claimed_raw_tids:
            continue
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        ioma = int(mask[y1:y2, x1:x2].sum()) / mask_area
        if ioma > best_ioma:
            best_ioma = ioma
            best_bbox = bbox
            best_raw_tid = raw_tid
    return best_raw_tid, best_bbox


def _find_raw_tid_for_bbox(
    bbox: np.ndarray,
    raw_frame_tracks: Dict[int, np.ndarray],
    claimed_raw_tids: set[int],
) -> Optional[int]:
    """Best-effort mapping from a bbox value back to its raw track id."""
    for raw_tid, raw_bbox in raw_frame_tracks.items():
        if raw_tid in claimed_raw_tids:
            continue
        if np.array_equal(raw_bbox, bbox):
            return raw_tid
    return None


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    """Derive tight bbox from mask pixels."""
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
