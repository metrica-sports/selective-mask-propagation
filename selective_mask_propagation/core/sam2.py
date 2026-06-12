"""SAM step: margin-triggered SAM2 propagation with runtime exit.

Scans base-tracker margins to find ambiguous windows (margin <
MARGIN_ENTRY), seeds SAM2 from clean pre-ambiguity frames (margin >=
SEED_MARGIN), and propagates through the window. Exit is determined at
runtime: each frame, IoMA identifies which track bbox contains the
mask, and that track's margin is checked. Single forward pass with
dynamic object add/remove.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..config import (
    MARGIN_ENTRY,
    MARGIN_EXIT,
    EXIT_CONSECUTIVE,
    SEED_MARGIN,
    SEED_CONSECUTIVE,
    IOMA_EXIT,
    SEED_CLEAN_IOU,
    MASK_OVERLAP_EXIT,
    AREA_DEGRADATION,
    GAP_TRIGGER,
    SAM_BORDER_MARGIN,
)


class WindowOutcome(Enum):
    SWAP = "swap"
    CLEAN = "clean"
    DEGRADED = "degraded"
    STALE = "stale"
    EDGE = "edge"
    END = "end"


@dataclass
class WindowSpec:
    raw_id: int
    seed_frame: int
    entry_frame: int
    trigger: str


@dataclass
class SamWindow:
    before_id: int
    canonical_id: int
    seed_frame: int
    entry_frame: int
    exit_frame: int
    outcome: WindowOutcome
    after_id: Optional[int]
    trigger: str


RenameEvent = Tuple[int, int, int]  # (raw_id, canonical_id, effective_frame)
_SWAP_TAG = "swap"
_DISPLACEMENT_TAG = "displacement"
_REVERT_TAG = "revert"


def step_sam(
    predictor,
    tracks: Dict[int, Dict[int, np.ndarray]],
    margins: Dict[int, Dict[int, float]],
    source_path: str,
    *,
    margin_entry: float = MARGIN_ENTRY,
    enable_gap: bool = True,
    enable_witness: bool = True,
) -> Tuple[
    Dict[int, Dict[int, np.ndarray]],
    List[SamWindow],
    List[RenameEvent],
    Dict[int, Dict[int, int]],
]:
    specs = find_windows(tracks, margins,
                          margin_entry=margin_entry,
                          enable_gap=enable_gap,
                          enable_witness=enable_witness)
    sam_masks, windows, rename_events, match_history = run_sam(predictor, specs, tracks, margins, source_path)
    return sam_masks, windows, rename_events, match_history


def find_windows(
    tracks: Dict[int, Dict[int, np.ndarray]],
    margins: Dict[int, Dict[int, float]],
    margin_entry: float = MARGIN_ENTRY,
    enable_gap: bool = True,
    enable_witness: bool = True,
) -> List[WindowSpec]:
    """Scan margins to find low-margin windows per track, with seed frames.

    Algorithm:
        1. Build per-track margin timelines from the margins dict
        2. For each track, identify contiguous low-margin regions
           (margin < MARGIN_ENTRY, ending when margin >= MARGIN_EXIT
           for EXIT_CONSECUTIVE frames)
        3. For each region, walk back from entry to find a seed
           (SEED_CONSECUTIVE frames with margin >= SEED_MARGIN)
        4. Discard entries without valid seeds
        5. For each margin-triggered window, find tracks with bbox overlap
           at the entry frame and add witness windows for them

    Entry and seed are deterministic (pre-SAM). Exit is set during run_sam.
    """
    track_timelines: Dict[int, List[Tuple[int, float]]] = {}
    for frame_idx in sorted(margins.keys()):
        for track_id, margin in margins[frame_idx].items():
            track_timelines.setdefault(track_id, []).append((frame_idx, margin))

    specs = []
    n_discarded = 0
    for track_id, timeline in track_timelines.items():
        timeline.sort()
        n_discarded += _extract_windows(track_id, timeline, tracks, specs, margin_entry=margin_entry)
    n_margin = len(specs)

    existing = {(s.raw_id, s.entry_frame) for s in specs}

    n_gap = 0
    if enable_gap:
        n_gap, n_gap_discarded = _extract_gap_windows(tracks, track_timelines, existing, specs)
        n_discarded += n_gap_discarded

    n_witnesses = 0
    if enable_witness:
        for spec in list(specs):
            entry_bboxes = tracks.get(spec.entry_frame, {})
            trigger_bbox = entry_bboxes.get(spec.raw_id)
            if trigger_bbox is None:
                continue
            for other_id, other_bbox in entry_bboxes.items():
                if other_id == spec.raw_id:
                    continue
                if (other_id, spec.entry_frame) in existing:
                    continue
                if _box_iou(trigger_bbox, other_bbox) > 0:
                    timeline = track_timelines.get(other_id)
                    if timeline is None:
                        continue
                    seed = _find_seed_frame(other_id, spec.entry_frame, timeline, tracks)
                    if seed is not None:
                        specs.append(WindowSpec(raw_id=other_id, seed_frame=seed, entry_frame=spec.entry_frame, trigger="witness"))
                        existing.add((other_id, spec.entry_frame))
                        n_witnesses += 1

    print(f"SAM: {len(specs)} windows ({n_margin} margin + {n_gap} gap + {n_witnesses} witness, "
          f"{n_discarded} discarded) across {len(set(s.raw_id for s in specs))} tracks")
    for s in specs:
        print(f"  T{s.raw_id}: seed={s.seed_frame} entry={s.entry_frame}")

    return specs


def _extract_windows(
    track_id: int,
    timeline: List[Tuple[int, float]],
    tracks: Dict[int, Dict[int, np.ndarray]],
    specs: List[WindowSpec],
    margin_entry: float = MARGIN_ENTRY,
) -> int:
    """Extract low-margin windows for a single track.

    Identifies contiguous low-margin regions (margin < margin_entry). A region
    ends when margin >= MARGIN_EXIT for EXIT_CONSECUTIVE frames, confirming the
    ambiguity genuinely resolved. Entries without a valid seed are discarded.

    Returns number of discarded entries (no seed found).
    """
    in_window = False
    entry_frame = -1
    high_streak = 0
    discarded = 0

    for frame_idx, margin in timeline:
        if not in_window:
            if margin < margin_entry:
                in_window = True
                entry_frame = frame_idx
                high_streak = 0
        else:
            if margin >= MARGIN_EXIT:
                high_streak += 1
                if high_streak >= EXIT_CONSECUTIVE:
                    seed = _find_seed_frame(track_id, entry_frame, timeline, tracks)
                    if seed is not None:
                        specs.append(WindowSpec(raw_id=track_id, seed_frame=seed, entry_frame=entry_frame, trigger="margin"))
                    else:
                        discarded += 1
                    in_window = False
            else:
                high_streak = 0

    if in_window:
        seed = _find_seed_frame(track_id, entry_frame, timeline, tracks)
        if seed is not None:
            specs.append(WindowSpec(raw_id=track_id, seed_frame=seed, entry_frame=entry_frame, trigger="margin"))
        else:
            discarded += 1

    return discarded


def _extract_gap_windows(
    tracks: Dict[int, Dict[int, np.ndarray]],
    track_timelines: Dict[int, List[Tuple[int, float]]],
    existing: set,
    specs: List[WindowSpec],
) -> Tuple[int, int]:
    """Find tracks lost for >= GAP_TRIGGER frames that reappear.

    Entry = last frame before the gap (track is present, stale check works).
    Seed = walk back from entry using _find_seed_frame (same as margin windows).

    Returns (n_created, n_discarded).
    """
    track_frames: Dict[int, List[int]] = {}
    for frame_idx, frame_tracks in tracks.items():
        for track_id in frame_tracks:
            track_frames.setdefault(track_id, []).append(frame_idx)

    n_created = 0
    n_discarded = 0
    for track_id, frames in track_frames.items():
        frames.sort()
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap < GAP_TRIGGER:
                continue
            entry_frame = frames[i]
            if (track_id, entry_frame) in existing:
                continue
            timeline = track_timelines.get(track_id)
            if timeline is None:
                n_discarded += 1
                continue
            seed = _find_seed_frame(track_id, entry_frame, timeline, tracks)
            if seed is not None:
                specs.append(WindowSpec(raw_id=track_id, seed_frame=seed, entry_frame=entry_frame, trigger="gap"))
                existing.add((track_id, entry_frame))
                n_created += 1
            else:
                n_discarded += 1

    return n_created, n_discarded


def _seed_is_clean(
    track_id: int,
    frame_idx: int,
    tracks: Dict[int, Dict[int, np.ndarray]],
) -> bool:
    """Check that no other bbox overlaps the target at the seed frame."""
    bbox = tracks.get(frame_idx, {}).get(track_id)
    if bbox is None:
        return False
    for other_id, other_bbox in tracks[frame_idx].items():
        if other_id == track_id:
            continue
        if _box_iou(bbox, other_bbox) >= SEED_CLEAN_IOU:
            return False
    return True


def _deferred_seed_frame_is_valid(
    track_id: int,
    frame_idx: int,
    tracks: Dict[int, Dict[int, np.ndarray]],
    margins: Dict[int, Dict[int, float]],
) -> bool:
    """Check a single frame is valid for deferred seeding: high margin and spatial isolation.

    Used when a spec's original seed frame collides with an active window.
    Same per-frame checks as _find_seed_frame but without the streak requirement,
    since the original spec already validated a nearby streak.
    """
    if track_id not in tracks.get(frame_idx, {}):
        return False
    margin = margins.get(frame_idx, {}).get(track_id, 0)
    if margin < SEED_MARGIN:
        return False
    return _seed_is_clean(track_id, frame_idx, tracks)


def _find_seed_frame(
    track_id: int,
    entry_frame: int,
    timeline: List[Tuple[int, float]],
    tracks: Dict[int, Dict[int, np.ndarray]],
) -> Optional[int]:
    """Walk back from entry to find SEED_CONSECUTIVE consecutive frames
    with margin >= SEED_MARGIN and spatial isolation. Returns the first
    frame of that streak where no other bbox overlaps (IoU < 0.10).

    Gaps in the timeline (brief detector dropouts) are skipped over.
    """
    streak = 0
    candidate = None
    for frame_idx, margin in reversed(timeline):
        if frame_idx >= entry_frame:
            continue
        if margin >= SEED_MARGIN and track_id in tracks.get(frame_idx, {}):
            streak += 1
            candidate = frame_idx
            if streak >= SEED_CONSECUTIVE and _seed_is_clean(track_id, candidate, tracks):
                return candidate
        else:
            streak = 0
            candidate = None
    return None


def _mask_in_box(mask: np.ndarray, bbox: np.ndarray) -> float:
    """IoMA: Intersection over Mask Area, |M ∩ B| / |M|."""
    total = mask.sum()
    if total == 0:
        return 0.0
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    inside = mask[y1:y2, x1:x2].sum()
    return inside / total


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _mask_at_border(mask: np.ndarray) -> bool:
    m = SAM_BORDER_MARGIN
    return (mask[:m, :].any() or mask[-m:, :].any() or
            mask[:, :m].any() or mask[:, -m:].any())


def build_predictor():
    """Build SAM2 video predictor."""
    from sam2.build_sam import build_sam2_video_predictor

    repo_root = Path(__file__).parent.parent.parent
    checkpoint = repo_root / "vendor" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    print("Building SAM2 predictor...")
    predictor = build_sam2_video_predictor(config, str(checkpoint))
    print("SAM2 ready.")
    return predictor


def run_sam(
    predictor,
    specs: List[WindowSpec],
    tracks: Dict[int, Dict[int, np.ndarray]],
    margins: Dict[int, Dict[int, float]],
    source_path: str,
) -> Tuple[
    Dict[int, Dict[int, np.ndarray]],
    List[SamWindow],
    List[RenameEvent],
    Dict[int, Dict[int, int]],
]:
    """Run SAM2 propagation with runtime exit determination.

    Single authority on canonical ID assignment. Maintains a live rename
    map (raw Deep-EIoU track ID → canonical ID) that handles both swaps
    and displacement:
      - Swap: raw track A ends up in canonical C's mask → live_renames[A] = C
      - Displacement: when a swap claims canonical C, raw track C (which
        defaulted to C) gets a new unique ID → live_renames[C] = new_id

    Every rename (swap + displacement) is recorded as a RenameEvent and
    returned alongside windows. The merge step builds its frame-indexed
    rename map directly from these events — one source of truth.

    Returns (sam_masks, windows, rename_events, match_history).
    """
    if not specs:
        print("SAM: no windows")
        return {}, [], [], {}

    img_dir = str(Path(source_path) / "img1")

    live_renames: Dict[int, int] = {}
    _tagged_events: list = []  # (raw_id, canonical_id, effective_frame, tag)
    all_ids = {tid for frame in tracks.values() for tid in frame}
    next_displacement_id = max(all_ids) + 1 if all_ids else 1

    specs_by_seed: Dict[int, List[WindowSpec]] = {}
    for s in specs:
        specs_by_seed.setdefault(s.seed_frame, []).append(s)

    sam_masks: Dict[int, Dict[int, np.ndarray]] = {}
    active_specs: Dict[int, WindowSpec] = {}

    exit_streaks: Dict[int, int] = {}
    streak_tracks: Dict[int, Optional[int]] = {}
    prev_border: Dict[int, bool] = {}
    convergence_streaks: Dict[frozenset, int] = {}
    match_history: Dict[int, Dict[int, int]] = {}  # {cid: {frame: best_track}}

    completed: List[SamWindow] = []

    start_frame = min(s.seed_frame for s in specs)
    last_seed = max(s.seed_frame for s in specs)
    total_frames = max(tracks.keys()) + 1

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(img_dir)
        predictor.non_overlap_masks = False

        for frame_idx in tqdm(range(start_frame, total_frames), desc="SAM propagation"):
            if frame_idx in specs_by_seed:
                for spec in specs_by_seed[frame_idx]:
                    canonical = live_renames.get(spec.raw_id, spec.raw_id)

                    if canonical in active_specs:
                        next_seed = spec.seed_frame + 1
                        if (next_seed < spec.entry_frame
                                and _deferred_seed_frame_is_valid(spec.raw_id, next_seed, tracks, margins)):
                            deferred = WindowSpec(raw_id=spec.raw_id, seed_frame=next_seed, entry_frame=spec.entry_frame, trigger=spec.trigger)
                            specs_by_seed.setdefault(next_seed, []).append(deferred)
                            last_seed = max(last_seed, next_seed)
                        continue

                    bbox = tracks[spec.seed_frame][spec.raw_id]

                    predictor.add_new_points_or_box(
                        state,
                        frame_idx=frame_idx,
                        obj_id=canonical,
                        box=bbox.tolist(),
                    )
                    active_specs[canonical] = spec
                    exit_streaks[canonical] = 0
                    streak_tracks[canonical] = None
                    prev_border[canonical] = False

            if not active_specs:
                if frame_idx >= last_seed:
                    break
                continue

            for _, obj_ids_out, video_res_masks in predictor.propagate_in_video(
                state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=0,
            ):
                frame_masks = {}
                for idx, obj_id in enumerate(obj_ids_out):
                    cid = int(obj_id)
                    if cid not in active_specs:
                        continue
                    mask = (video_res_masks[idx][0] > 0.0).cpu().numpy()
                    if mask.any():
                        frame_masks[cid] = mask

                if frame_masks:
                    sam_masks[frame_idx] = frame_masks

            to_remove = []

            # Convergence kill (before per-window logic): find all mask pairs
            # with pixel IoU >= threshold for EXIT_CONSECUTIVE frames.
            # Convergence is a pair property — both masks are killed
            # regardless of warmup/authoritative zone.
            converged = set()
            frame_masks_now = sam_masks.get(frame_idx, {})
            active_cids = [c for c in active_specs if c in frame_masks_now and frame_masks_now[c].any()]
            seen_pairs = set()
            for i, cid_a in enumerate(active_cids):
                mask_a = frame_masks_now[cid_a]
                for cid_b in active_cids[i + 1:]:
                    pair = frozenset((cid_a, cid_b))
                    seen_pairs.add(pair)
                    mask_b = frame_masks_now[cid_b]
                    overlap = (mask_a & mask_b).sum()
                    union = (mask_a | mask_b).sum()
                    if union > 0 and overlap / union >= MASK_OVERLAP_EXIT:
                        convergence_streaks[pair] = convergence_streaks.get(pair, 0) + 1
                        if convergence_streaks[pair] >= EXIT_CONSECUTIVE:
                            converged.update((cid_a, cid_b))
                    else:
                        convergence_streaks[pair] = 0
            # Reset streaks for pairs no longer both active
            for pair in list(convergence_streaks):
                if pair not in seen_pairs:
                    del convergence_streaks[pair]
            for cid in converged:
                spec = active_specs[cid]
                completed.append(SamWindow(
                    before_id=spec.raw_id, canonical_id=cid,
                    seed_frame=spec.seed_frame, entry_frame=spec.entry_frame,
                    exit_frame=frame_idx, outcome=WindowOutcome.DEGRADED,
                    after_id=None, trigger=spec.trigger,
                ))
                to_remove.append(cid)

            for cid, spec in active_specs.items():
                if cid in converged:
                    continue
                if frame_idx < spec.entry_frame:
                    continue

                mask = frame_masks_now.get(cid)

                # Stale check: at entry, verify mask still covers before_id's bbox.
                # If not, DE reassigned the raw track — seed and entry are different people.
                if frame_idx == spec.entry_frame and mask is not None:
                    before_bbox = tracks.get(frame_idx, {}).get(spec.raw_id)
                    if before_bbox is not None:
                        ioma = _mask_in_box(mask, before_bbox)
                        if ioma < IOMA_EXIT:
                            completed.append(SamWindow(
                                before_id=spec.raw_id, canonical_id=cid,
                                seed_frame=spec.seed_frame, entry_frame=spec.entry_frame,
                                exit_frame=frame_idx, outcome=WindowOutcome.STALE,
                                after_id=None, trigger=spec.trigger,
                            ))
                            to_remove.append(cid)
                            continue

                # Edge exit: mask touched border last frame, empty this frame
                if prev_border.get(cid, False) and mask is None:
                    completed.append(SamWindow(
                        before_id=spec.raw_id, canonical_id=cid,
                        seed_frame=spec.seed_frame, entry_frame=spec.entry_frame,
                        exit_frame=frame_idx - 1, outcome=WindowOutcome.EDGE,
                        after_id=None, trigger=spec.trigger,
                    ))
                    to_remove.append(cid)
                    continue

                prev_border[cid] = mask is not None and _mask_at_border(mask)

                if mask is None:
                    exit_streaks[cid] = 0
                    streak_tracks[cid] = None
                    continue

                # Match mask to best DE track
                mask_bbox = _bbox_from_mask(mask)
                best_iou = 0.0
                best_track = None
                for track_id, bbox in tracks.get(frame_idx, {}).items():
                    ioma = _mask_in_box(mask, bbox)
                    if ioma < IOMA_EXIT:
                        continue
                    iou = _box_iou(mask_bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_track = track_id

                if best_track is not None:
                    match_history.setdefault(cid, {})[frame_idx] = best_track
                    margin = margins.get(frame_idx, {}).get(best_track, float('inf'))
                    best_bbox = tracks[frame_idx][best_track]
                    # DE bbox isolation
                    de_isolated = all(
                        _box_iou(best_bbox, bbox) < SEED_CLEAN_IOU
                        for tid, bbox in tracks[frame_idx].items()
                        if tid != best_track
                    )
                    # Mask isolation: no other active mask's derived bbox overlaps
                    mask_isolated = True
                    for other_cid in active_specs:
                        if other_cid == cid:
                            continue
                        other_mask = sam_masks.get(frame_idx, {}).get(other_cid)
                        if other_mask is None or not other_mask.any():
                            continue
                        other_bbox = _bbox_from_mask(other_mask)
                        if _box_iou(mask_bbox, other_bbox) >= SEED_CLEAN_IOU:
                            mask_isolated = False
                            break

                    if margin >= MARGIN_EXIT and de_isolated and mask_isolated:
                        if best_track == streak_tracks[cid]:
                            exit_streaks[cid] += 1
                        else:
                            exit_streaks[cid] = 1
                            streak_tracks[cid] = best_track

                        if exit_streaks[cid] >= EXIT_CONSECUTIVE:
                            # Area degradation check: mask shrank vs seed
                            seed_mask = sam_masks.get(spec.seed_frame, {}).get(cid)
                            if seed_mask is not None and mask.sum() < seed_mask.sum() * AREA_DEGRADATION:
                                completed.append(SamWindow(
                                    before_id=spec.raw_id, canonical_id=cid,
                                    seed_frame=spec.seed_frame, entry_frame=spec.entry_frame,
                                    exit_frame=frame_idx, outcome=WindowOutcome.DEGRADED,
                                    after_id=None, trigger=spec.trigger,
                                ))
                                to_remove.append(cid)
                                continue

                            best_canonical = live_renames.get(best_track, best_track)
                            if best_canonical != cid:
                                # Find when the mask first settled into after_id's track.
                                # Walk backward from exit to find the start of the last
                                # contiguous run where best_track matches after_id.
                                swap_frame = spec.entry_frame
                                cid_history = match_history.get(cid, {})
                                for f in range(frame_idx, spec.entry_frame - 1, -1):
                                    if cid_history.get(f) != best_track:
                                        swap_frame = f + 1
                                        break

                                completed.append(SamWindow(
                                    before_id=spec.raw_id, canonical_id=cid,
                                    seed_frame=spec.seed_frame, entry_frame=spec.entry_frame,
                                    exit_frame=frame_idx, outcome=WindowOutcome.SWAP,
                                    after_id=best_track, trigger=spec.trigger,
                                ))
                                evicted = [k for k, v in live_renames.items() if v == cid]
                                live_renames = {k: v for k, v in live_renames.items() if v != cid}
                                for k in evicted:
                                    _tagged_events.append((k, k, swap_frame, _REVERT_TAG))
                                live_renames[best_track] = cid
                                _tagged_events.append((best_track, cid, swap_frame, _SWAP_TAG))

                                if live_renames.get(cid, cid) == cid:
                                    did = next_displacement_id
                                    live_renames[cid] = did
                                    _tagged_events.append((cid, did, swap_frame, _DISPLACEMENT_TAG))
                                    next_displacement_id += 1
                            else:
                                completed.append(SamWindow(
                                    before_id=spec.raw_id, canonical_id=cid,
                                    seed_frame=spec.seed_frame, entry_frame=spec.entry_frame,
                                    exit_frame=frame_idx, outcome=WindowOutcome.CLEAN,
                                    after_id=None, trigger=spec.trigger,
                                ))

                            to_remove.append(cid)
                    else:
                        exit_streaks[cid] = 0
                        streak_tracks[cid] = None
                else:
                    exit_streaks[cid] = 0
                    streak_tracks[cid] = None

            for cid in to_remove:
                predictor.remove_object(state, cid)
                del active_specs[cid]
                del exit_streaks[cid]
                del streak_tracks[cid]
                del prev_border[cid]

    for cid, spec in active_specs.items():
        completed.append(SamWindow(
            before_id=spec.raw_id, canonical_id=cid,
            seed_frame=spec.seed_frame, entry_frame=spec.entry_frame,
            exit_frame=total_frames - 1, outcome=WindowOutcome.END,
            after_id=None, trigger=spec.trigger,
        ))

    swaps = [w for w in completed if w.outcome == WindowOutcome.SWAP]
    clean = [w for w in completed if w.outcome == WindowOutcome.CLEAN]
    edge = [w for w in completed if w.outcome == WindowOutcome.EDGE]
    end = [w for w in completed if w.outcome == WindowOutcome.END]
    degraded = [w for w in completed if w.outcome == WindowOutcome.DEGRADED]
    stale = [w for w in completed if w.outcome == WindowOutcome.STALE]
    print(f"SAM: {len(swaps)} swap, {len(clean)} clean, {len(edge)} edge, {len(end)} end, {len(degraded)} degraded, {len(stale)} stale")
    for w in swaps:
        print(f"  T{w.canonical_id}: SWAP exit={w.exit_frame} after_id={w.after_id}")
    for w in edge:
        print(f"  T{w.canonical_id}: edge exit at frame {w.exit_frame}")
    for w in degraded:
        print(f"  T{w.canonical_id}: DEGRADED at frame {w.exit_frame}")
    for w in stale:
        print(f"  T{w.canonical_id}: STALE at entry {w.entry_frame} (seed person ≠ entry person)")

    # Prune displacement events superseded by swap events.
    # A displacement at frame F_d for raw_id X is unnecessary if a swap
    # at frame F_s <= F_d already moved X away from its default canonical ID.
    swap_frames: Dict[int, int] = {}
    for raw_id, _, frame, tag in _tagged_events:
        if tag == _SWAP_TAG:
            if raw_id not in swap_frames or frame < swap_frames[raw_id]:
                swap_frames[raw_id] = frame

    rename_events: List[RenameEvent] = []
    for raw_id, canonical_id, frame, tag in _tagged_events:
        if tag == _DISPLACEMENT_TAG and raw_id in swap_frames and swap_frames[raw_id] <= frame:
            continue
        rename_events.append((raw_id, canonical_id, frame))

    total_mask_frames = sum(len(v) for v in sam_masks.values())
    print(f"SAM: {total_mask_frames} track-frame masks across {len(sam_masks)} frames")
    print(f"SAM: {len(rename_events)} rename events")
    for raw_id, canonical_id, frame in rename_events:
        print(f"  T{raw_id} → {canonical_id} from frame {frame}")
    return sam_masks, completed, rename_events, match_history
