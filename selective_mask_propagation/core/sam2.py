"""SAM step: margin-triggered SAM2 propagation with runtime exit.

Scans base-tracker margins to find ambiguous windows (margin <
MARGIN_ENTRY), seeds SAM2 from clean pre-ambiguity frames (margin >=
SEED_MARGIN), and propagates through the window. Exit is determined at
runtime: each frame, IoMA identifies which track bbox contains the
mask, and that track's margin is checked. Single forward pass with
dynamic object add/remove.
"""

import functools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import (
    MARGIN_ENTRY,
    MARGIN_EXIT,
    EXIT_CONSECUTIVE,
    SEED_MARGIN,
    SEED_CONSECUTIVE,
    SEED_CLEAN_IOU,
    GAP_TRIGGER,
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


@functools.cache
def build_predictor():
    """Build SAM2 video predictor. Cached — one build per process."""
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
    """Run SAM2 windowed propagation. See propagation.run_sam_windows."""
    from .propagation import run_sam_windows  # function-level: avoids circular import

    img_dir = str(Path(source_path) / "img1")
    return run_sam_windows(lambda: Sam2Session(predictor, img_dir), specs, tracks, margins)


class Sam2Session:
    """SAM2 session for the propagation driver — one shared inference state."""

    def __init__(self, predictor, img_dir: str):
        self.predictor = predictor
        self.state = predictor.init_state(img_dir)
        predictor.non_overlap_masks = False

    def add_objects(self, frame_idx: int, objects: List[Tuple[int, np.ndarray]]) -> None:
        """Seed new objects with absolute-pixel bbox prompts."""
        for obj_id, bbox in objects:
            self.predictor.add_new_points_or_box(
                self.state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=bbox.tolist(),
            )

    def propagate_one_frame(self, frame_idx: int):
        """Propagate one frame. Returns (obj_ids, video_res_masks)."""
        obj_ids_out: List[int] = []
        video_res_masks = None
        for _, obj_ids, masks in self.predictor.propagate_in_video(
            self.state,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=0,
        ):
            obj_ids_out, video_res_masks = obj_ids, masks
        return obj_ids_out, video_res_masks

    def remove_object(self, obj_id: int) -> None:
        self.predictor.remove_object(self.state, obj_id)
