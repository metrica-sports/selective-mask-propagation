"""Shared window-machine driver for SAM propagation with runtime exit.

run_sam_windows() owns the complete control flow — seeding, deferred
re-seeds, convergence kill, stale/edge/degraded checks, exit streaks,
and swap/displacement renames. It is the single authority on canonical
ID assignment: a live rename map (raw track ID → canonical ID) handles
both swaps and displacement, and every rename is recorded as a
RenameEvent that the merge step consumes directly.

The VOS model is abstracted behind a session object:

    add_objects(frame_idx, [(obj_id, bbox_px), ...])   seed new objects
    propagate_one_frame(frame_idx) -> (obj_ids, video_res_masks)
    remove_object(obj_id)

``sam2.Sam2Session`` and ``sam3.Sam3Session`` implement it; bboxes are
absolute pixels and any model-specific conversion happens inside the
session.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..config import (
    AREA_DEGRADATION,
    EXIT_CONSECUTIVE,
    IOMA_EXIT,
    MARGIN_EXIT,
    MASK_OVERLAP_EXIT,
    SEED_CLEAN_IOU,
)
from .sam2 import (
    RenameEvent,
    SamWindow,
    WindowOutcome,
    WindowSpec,
    _DISPLACEMENT_TAG,
    _REVERT_TAG,
    _SWAP_TAG,
    _bbox_from_mask,
    _box_iou,
    _deferred_seed_frame_is_valid,
    _mask_at_border,
    _mask_in_box,
)


def run_sam_windows(
    session_factory: Callable[[], object],
    specs: List[WindowSpec],
    tracks: Dict[int, Dict[int, np.ndarray]],
    margins: Dict[int, Dict[int, float]],
) -> Tuple[
    Dict[int, Dict[int, np.ndarray]],
    List[SamWindow],
    List[RenameEvent],
    Dict[int, Dict[int, int]],
]:
    """Run windowed propagation with runtime exit determination.

    Returns (sam_masks, windows, rename_events, match_history).
    """
    if not specs:
        print("SAM: no windows")
        return {}, [], [], {}

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
        session = session_factory()

        for frame_idx in tqdm(range(start_frame, total_frames), desc="SAM propagation"):
            new_objects: List[Tuple[int, np.ndarray]] = []
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

                    new_objects.append((canonical, bbox))
                    active_specs[canonical] = spec
                    exit_streaks[canonical] = 0
                    streak_tracks[canonical] = None
                    prev_border[canonical] = False

            session.add_objects(frame_idx, new_objects)

            if not active_specs:
                if frame_idx >= last_seed:
                    break
                continue

            obj_ids_out, video_res_masks = session.propagate_one_frame(frame_idx)
            frame_masks = {}
            if video_res_masks is not None:
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
                # If not, the base tracker reassigned the raw track — seed and
                # entry are different people.
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

                # Match mask to best base-tracker box
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
                    # Base-tracker bbox isolation
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
                session.remove_object(cid)
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
