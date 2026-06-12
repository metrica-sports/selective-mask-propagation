"""SAM step: margin-triggered SAM3 propagation with runtime exit.

Same API and exit logic as sam2.py — both delegate to
propagation.run_sam_windows; only the session (predictor construction
and interaction) differs. Sam3Session works around SAM3's lack of
dynamic object add/remove by holding one tracker state per birth
cohort. Also hosts run_sam_uniform, the uniform-dispatch baseline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .sam2 import (
    MARGIN_ENTRY,
    RenameEvent,
    SamWindow,
    WindowSpec,
    find_windows,
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

    def __init__(self, predictor, images, video_height: int, video_width: int,
                 prompt_width: int, prompt_height: int):
        self.predictor = predictor
        self.images = images
        self.video_height = video_height
        self.video_width = video_width
        self.prompt_width = prompt_width
        self.prompt_height = prompt_height
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
        """Add objects (absolute-pixel bboxes) that first appear on this frame
        as one new tracker state."""
        if not objects:
            return
        state = self._new_tracker_state(frame_idx)
        for obj_id, bbox in objects:
            bbox_norm = _normalize_bbox(bbox, self.prompt_width, self.prompt_height)
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
    ) -> Tuple[List[int], Optional[torch.Tensor]]:
        """Propagate all states by one frame and concatenate outputs."""
        obj_ids_all: List[int] = []
        video_res_masks_all: List[torch.Tensor] = []

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

        if not obj_ids_all:
            return [], None

        video_res_masks = (
            video_res_masks_all[0]
            if len(video_res_masks_all) == 1
            else torch.cat(video_res_masks_all, dim=0)
        )
        return obj_ids_all, video_res_masks

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
    """Run SAM3 windowed propagation. See propagation.run_sam_windows."""
    from .propagation import run_sam_windows  # function-level: avoids circular import

    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(Path(source_path) / "seqinfo.ini")
    video_width = int(cfg["Sequence"]["imWidth"])
    video_height = int(cfg["Sequence"]["imHeight"])

    img_dir = str(Path(source_path) / "img1")

    def make_session():
        images, vid_h, vid_w = _load_frames(predictor, img_dir)
        predictor.non_overlap_masks_for_output = False
        return Sam3Session(predictor, images, vid_h, vid_w, video_width, video_height)

    return run_sam_windows(make_session, specs, tracks, margins)


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
        session = Sam3Session(predictor, images, vid_h, vid_w, video_width, video_height)

        for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="SAM uniform"):
            new_objects: List[Tuple[int, np.ndarray]] = []
            for track_id in births_at.get(frame_idx, []):
                bbox = tracks[frame_idx][track_id]
                new_objects.append((track_id, bbox))
            session.add_objects(frame_idx, new_objects)

            obj_ids_out, video_res_masks = session.propagate_one_frame(frame_idx)
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
