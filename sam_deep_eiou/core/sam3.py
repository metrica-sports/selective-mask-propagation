"""SAM step: margin-triggered SAM3 propagation with runtime exit.

Drop-in replacement for sam2.py. Same API, same data types, same runtime
exit logic. Only the predictor construction and the predictor interaction
(init_state, add_new_points_or_box, propagate_in_video, remove_object) differ
between SAM2 and SAM3.

To swap: change `from .core.sam2 import step_sam` to
`from .core.sam3 import step_sam` in cli.py.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .sam2 import (
    WindowOutcome,
    RenameEvent,
    SamWindow,
    WindowSpec,
    find_windows,
    MARGIN_ENTRY,
    MARGIN_EXIT,
    EXIT_CONSECUTIVE,
    IOMA_EXIT,
    SEED_CLEAN_IOU,
    MASK_OVERLAP_EXIT,
    AREA_DEGRADATION,
    _SWAP_TAG,
    _DISPLACEMENT_TAG,
    _REVERT_TAG,
    _mask_in_box,
    _bbox_from_mask,
    _box_iou,
    _mask_at_border,
    _deferred_seed_frame_is_valid,
)


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


def build_predictor():
    """Build SAM3 tracker with detector backbone attached."""
    from sam3.model_builder import build_sam3_video_model

    print("Building SAM3 model...")
    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    print("SAM3 ready.")
    return predictor


class LazyFrameLoader:
    """Memory-efficient frame loader for SAM3.

    Drop-in replacement for the preloaded frame tensor. Loads frames on demand
    using the same transforms as SAM3's bulk loader. Evicts frames more than
    ``lookback`` positions behind the most recent access, bounding memory to
    O(lookback) instead of O(total_frames).
    """

    def __init__(self, img_dir: str, image_size: int, lookback: int = 0):
        from sam3.model.io_utils import _load_img_as_tensor

        self._load_fn = _load_img_as_tensor
        self._image_size = image_size
        self._lookback = lookback

        frame_names = [
            p for p in Path(img_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        ]
        frame_names.sort(key=lambda p: int(p.stem))
        self._img_paths = [str(p) for p in frame_names]
        self._num_frames = len(self._img_paths)
        if self._num_frames == 0:
            raise RuntimeError(f"no images found in {img_dir}")

        self._img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16).view(3, 1, 1)
        self._img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16).view(3, 1, 1)

        _, self.video_height, self.video_width = self._load_fn(self._img_paths[0], image_size)
        self._buffer: Dict[int, torch.Tensor] = {}
        self._latest_idx = -1

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self._buffer:
            self._latest_idx = max(self._latest_idx, idx)
            return self._buffer[idx]

        img, _, _ = self._load_fn(self._img_paths[idx], self._image_size)
        img = img.to(dtype=torch.float16)
        img -= self._img_mean
        img /= self._img_std

        self._buffer[idx] = img
        self._latest_idx = max(self._latest_idx, idx)

        evict_before = self._latest_idx - self._lookback
        to_evict = [k for k in self._buffer if k < evict_before]
        for k in to_evict:
            del self._buffer[k]

        return img


def _load_frames(predictor, img_dir: str):
    """Load video frames lazily — O(1) GPU memory instead of O(total_frames)."""
    loader = LazyFrameLoader(img_dir, predictor.image_size)
    return loader, loader.video_height, loader.video_width


def _normalize_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert absolute pixel bbox [x1,y1,x2,y2] to [0,1] relative coords."""
    return np.array([
        bbox[0] / width, bbox[1] / height,
        bbox[2] / width, bbox[3] / height,
    ], dtype=np.float32)


class Sam3Session:
    """SAM3 dynamic-object session using multiple tracker states.

    SAM3 doesn't support dynamic add/remove natively. This works around it:
    - Add objects by creating new inference states
    - Propagate each state per frame and concatenate outputs
    - Remove objects by slicing each state and dropping empty states
    """

    def __init__(self, predictor, images, video_height: int, video_width: int):
        self.predictor = predictor
        self.images = images
        self.video_height = video_height
        self.video_width = video_width
        self.num_frames = len(images)
        self.tracker_states: List[Dict[str, Any]] = []
        # Cache one frame's backbone output and share it across states.
        self._cached_frame_idx: Optional[int] = None
        self._cached_feature: Optional[Tuple[torch.Tensor, Dict[str, Any]]] = None

    def _prepare_frame_feature(self, frame_idx: int) -> None:
        if self._cached_frame_idx == frame_idx and self._cached_feature is not None:
            return
        image = self.images[frame_idx].cuda().float().unsqueeze(0)
        backbone_out = self.predictor.forward_image(image)
        self._cached_frame_idx = frame_idx
        self._cached_feature = (image, backbone_out)

    def _attach_frame_feature(self, state: Dict[str, Any], frame_idx: int) -> None:
        self._prepare_frame_feature(frame_idx)
        assert self._cached_feature is not None
        state["cached_features"] = {frame_idx: self._cached_feature}

    def _new_tracker_state(self, frame_idx: int) -> Dict[str, Any]:
        state = self.predictor.init_state(
            video_height=self.video_height,
            video_width=self.video_width,
            num_frames=self.num_frames,
        )
        state["images"] = self.images
        self._attach_frame_feature(state, frame_idx)
        return state

    def add_objects(self, frame_idx: int, objects: List[Tuple[int, np.ndarray]]) -> None:
        """Add objects that first appear on this frame as one new tracker state."""
        if not objects:
            return
        state = self._new_tracker_state(frame_idx)
        for obj_id, bbox_norm in objects:
            self.predictor.add_new_points_or_box(
                state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=bbox_norm,
                rel_coordinates=True,
            )
        self.predictor.propagate_in_video_preflight(state, run_mem_encoder=True)
        self.tracker_states.append(state)

    def propagate_one_frame(
        self, frame_idx: int
    ) -> Tuple[List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Propagate all states by one frame and concatenate outputs."""
        obj_ids_all: List[int] = []
        video_res_masks_all: List[torch.Tensor] = []
        obj_scores_all: List[torch.Tensor] = []

        for state in self.tracker_states:
            if len(state["obj_ids"]) == 0:
                continue

            self._attach_frame_feature(state, frame_idx)

            num_frames_propagated = 0
            out_frame_idx = None
            out_obj_ids: List[int] = []
            out_video_res_masks = None
            out_obj_scores = None
            for out in self.predictor.propagate_in_video(
                state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=0,
                reverse=False,
                tqdm_disable=True,
                run_mem_encoder=True,
            ):
                (
                    out_frame_idx,
                    out_obj_ids,
                    _out_low_res_masks,
                    out_video_res_masks,
                    out_obj_scores,
                ) = out
                num_frames_propagated += 1

            assert (
                num_frames_propagated == 1 and out_frame_idx == frame_idx
            ), (
                f"num_frames_propagated={num_frames_propagated}, "
                f"out_frame_idx={out_frame_idx}, frame_idx={frame_idx}"
            )
            obj_ids_all.extend(int(obj_id) for obj_id in out_obj_ids)
            video_res_masks_all.append(out_video_res_masks)
            obj_scores_all.append(out_obj_scores)

        if not obj_ids_all:
            return [], None, None

        video_res_masks = (
            video_res_masks_all[0]
            if len(video_res_masks_all) == 1
            else torch.cat(video_res_masks_all, dim=0)
        )
        obj_scores = (
            obj_scores_all[0]
            if len(obj_scores_all) == 1
            else torch.cat(obj_scores_all, dim=0)
        )
        return obj_ids_all, video_res_masks, obj_scores

    def remove_object(self, obj_id: int) -> None:
        """Remove object from all states and drop empty states."""
        tracker_states_before_removal = self.tracker_states.copy()
        self.tracker_states.clear()
        for state in tracker_states_before_removal:
            new_obj_ids, _ = self.predictor.remove_object(
                state, obj_id, strict=False, need_output=False
            )
            if len(new_obj_ids) > 0:
                self.tracker_states.append(state)


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
    """Run SAM3 propagation with runtime exit determination.

    Same algorithm as sam2.run_sam — single authority on canonical ID
    assignment, live rename map, displacement pruning. Only the predictor
    interaction differs (SAM3 API: rel_coordinates, 5-tuple yield,
    propagate_preflight).

    Returns (sam_masks, windows, rename_events, match_history).
    """
    if not specs:
        print("SAM: no windows")
        return {}, [], [], {}

    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(Path(source_path) / "seqinfo.ini")
    video_width = int(cfg["Sequence"]["imWidth"])
    video_height = int(cfg["Sequence"]["imHeight"])

    img_dir = str(Path(source_path) / "img1")

    live_renames: Dict[int, int] = {}
    _tagged_events: list = []  # (raw_id, canonical_id, effective_frame, tag)
    all_ids = {tid for frame in tracks.values() for tid in frame}
    next_displacement_id = max(all_ids) + 1 if all_ids else 1

    specs_by_seed: Dict[int, List[WindowSpec]] = {}
    for s in specs:
        specs_by_seed.setdefault(s.seed_frame, []).append(s)

    images, vid_h, vid_w = _load_frames(predictor, img_dir)

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
        predictor.non_overlap_masks_for_output = False
        session = Sam3Session(predictor, images, vid_h, vid_w)

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
                    bbox_norm = _normalize_bbox(bbox, video_width, video_height)

                    new_objects.append((canonical, bbox_norm))
                    active_specs[canonical] = spec
                    exit_streaks[canonical] = 0
                    streak_tracks[canonical] = None
                    prev_border[canonical] = False

            session.add_objects(frame_idx, new_objects)

            if not active_specs:
                if frame_idx >= last_seed:
                    break
                continue

            obj_ids_out, video_res_masks, _ = session.propagate_one_frame(frame_idx)
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


def run_sam_uniform(
    predictor,
    tracks: Dict[int, Dict[int, np.ndarray]],
    source_path: str,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Run SAM3 uniformly on every Deep-EIoU track from inception to track death.

    Sibling of ``run_sam`` for the §4.7 ``de_uniform_sam3`` baseline: same
    Sam3Session machinery, but no margin signal, no windows, no exit logic,
    no rename events. SAM3 is added at each track's first appearance and
    removed when the track disappears. While alive, every frame runs the
    SAM3 forward pass for that track.

    Algorithm:
        1. Compute per-track inception frame and death frame from the
           ``tracks`` dict (first / last frame each track ID appears).
        2. Iterate frames from earliest inception to last death.
        3. At each frame: add SAM3 box prompts for tracks born this frame
           (using DE's bbox at that frame as the prompt), propagate one
           frame, collect non-empty masks. After collecting, remove SAM3
           objects for tracks that died on this frame so they free state
           before the next frame's add/propagate cycle.
        4. SAM3 obj_id = DE track ID. No canonical/raw ID distinction.

    Returns:
        sam_masks: {frame_idx: {track_id: bool mask (H, W)}} — only
        non-empty masks are recorded.
    """
    if not tracks:
        print("SAM (uniform): no tracks")
        return {}

    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(Path(source_path) / "seqinfo.ini")
    video_width = int(cfg["Sequence"]["imWidth"])
    video_height = int(cfg["Sequence"]["imHeight"])

    img_dir = str(Path(source_path) / "img1")

    track_births: Dict[int, int] = {}
    track_deaths: Dict[int, int] = {}
    for frame_idx in sorted(tracks.keys()):
        for track_id in tracks[frame_idx]:
            if track_id not in track_births:
                track_births[track_id] = frame_idx
            track_deaths[track_id] = frame_idx

    births_at: Dict[int, List[int]] = {}
    deaths_at: Dict[int, List[int]] = {}
    for tid, f in track_births.items():
        births_at.setdefault(f, []).append(tid)
    for tid, f in track_deaths.items():
        deaths_at.setdefault(f, []).append(tid)

    sam_masks: Dict[int, Dict[int, np.ndarray]] = {}

    start_frame = min(track_births.values())
    end_frame = max(track_deaths.values())

    images, vid_h, vid_w = _load_frames(predictor, img_dir)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.non_overlap_masks_for_output = False
        session = Sam3Session(predictor, images, vid_h, vid_w)

        for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="SAM uniform"):
            new_objects: List[Tuple[int, np.ndarray]] = []
            for track_id in births_at.get(frame_idx, []):
                bbox = tracks[frame_idx][track_id]
                bbox_norm = _normalize_bbox(bbox, video_width, video_height)
                new_objects.append((track_id, bbox_norm))
            session.add_objects(frame_idx, new_objects)

            obj_ids_out, video_res_masks, _ = session.propagate_one_frame(frame_idx)
            frame_masks: Dict[int, np.ndarray] = {}
            if video_res_masks is not None:
                for idx, obj_id in enumerate(obj_ids_out):
                    cid = int(obj_id)
                    mask = (video_res_masks[idx][0] > 0.0).cpu().numpy()
                    if mask.any():
                        frame_masks[cid] = mask
            if frame_masks:
                sam_masks[frame_idx] = frame_masks

            for track_id in deaths_at.get(frame_idx, []):
                session.remove_object(track_id)

    total_mask_frames = sum(len(v) for v in sam_masks.values())
    n_tracks = len(track_births)
    print(f"SAM (uniform): {total_mask_frames} track-frame masks across "
          f"{len(sam_masks)} frames; {n_tracks} tracks")
    return sam_masks
