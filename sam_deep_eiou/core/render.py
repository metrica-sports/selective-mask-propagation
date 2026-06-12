"""Tracker video rendering with GT overlay."""

import configparser
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .merge import TrackData
from ..utils.shared_constants import BORDER_MARGIN, SAM_BORDER_MARGIN


def _tracks_to_merged(tracks: Dict[int, Dict[int, np.ndarray]]) -> Dict[int, Dict[int, TrackData]]:
    """Convert raw Deep-EIoU tracks to TrackData dicts (bbox only, no masks)."""
    return {
        frame: {tid: TrackData(bbox=bbox) for tid, bbox in frame_tracks.items()}
        for frame, frame_tracks in tracks.items()
    }


def step_render(
    source_path: str,
    tracks: Dict[int, Dict[int, np.ndarray]],
    merged: Dict[int, Dict[int, TrackData]],
    output_dir: str,
    margins: Dict[int, Dict[int, float]],
    renamed_margins: Dict[int, Dict[int, float]],
    de_frame_errors: dict = None,
    sde_frame_errors: dict = None,
    de_gta_tracks: Dict[int, Dict[int, np.ndarray]] = None,
    de_gta_frame_errors: dict = None,
    sde_gta_tracks: Dict[int, Dict[int, np.ndarray]] = None,
    sde_gta_frame_errors: dict = None,
    no_gt: bool = False,
) -> None:
    render_tracker_video(
        source_path, str(Path(output_dir) / "deep_eiou.mp4"),
        _tracks_to_merged(tracks), margins=margins,
        frame_errors=de_frame_errors,
        no_gt=no_gt,
    )
    render_tracker_video(
        source_path, str(Path(output_dir) / "sam_deep_eiou.mp4"),
        merged, margins=renamed_margins,
        frame_errors=sde_frame_errors,
        baseline_errors=de_frame_errors,
        no_gt=no_gt,
    )
    if de_gta_tracks is not None:
        render_tracker_video(
            source_path, str(Path(output_dir) / "de_gta.mp4"),
            _tracks_to_merged(de_gta_tracks),
            frame_errors=de_gta_frame_errors,
            baseline_errors=de_frame_errors,
            no_gt=no_gt,
        )
    if sde_gta_tracks is not None:
        render_tracker_video(
            source_path, str(Path(output_dir) / "sde_gta.mp4"),
            _tracks_to_merged(sde_gta_tracks),
            frame_errors=sde_gta_frame_errors,
            baseline_errors=de_frame_errors,
            no_gt=no_gt,
        )


FFMPEG_PATH = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")

TRACK_COLORS = [
    (255, 144, 30), (50, 205, 50), (205, 0, 205), (255, 191, 0),
    (0, 215, 255), (180, 105, 255), (235, 206, 135), (255, 255, 0),
    (0, 165, 255), (147, 20, 255), (0, 255, 127), (238, 130, 238),
    (60, 180, 75), (230, 25, 75), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (128, 0, 0), (0, 128, 128),
    (128, 128, 0), (0, 0, 200), (170, 110, 40), (100, 200, 200),
    (80, 70, 180),
]


def track_color(track_id: int) -> tuple:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def _load_gt(source_path: str) -> Dict[int, list]:
    """Load GT annotations, filtering to active pedestrians (flag=1, class=1).

    Returns {frame_idx: [(track_id, x1, y1, x2, y2), ...]}.
    """
    gt_path = Path(source_path) / "gt" / "gt.txt"
    if not gt_path.exists():
        return {}
    gt = defaultdict(list)
    for line in gt_path.read_text().strip().split("\n"):
        parts = line.split(",")
        frame_idx = int(parts[0]) - 1
        track_id = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        flag = int(parts[6])
        cls = int(parts[7])
        if flag != 1 or cls != 1:
            continue
        gt[frame_idx].append((track_id, int(x), int(y), int(x + w), int(y + h)))
    return dict(gt)


def _draw_label(frame, text, cx, y, bg_color, *, above=True):
    """Draw a centered label pill above or below y coordinate."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    if above:
        cv2.rectangle(frame, (cx - tw // 2 - 2, y - th - 6),
                      (cx + tw // 2 + 2, y), bg_color, -1)
        cv2.putText(frame, text, (cx - tw // 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        cv2.rectangle(frame, (cx - tw // 2 - 2, y),
                      (cx + tw // 2 + 2, y + th + 6), bg_color, -1)
        cv2.putText(frame, text, (cx - tw // 2, y + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)



def _process_source(source_path: str, target_path: str, callback):
    """Process frames from an image sequence directory and write video."""
    frame_dir = Path(source_path) / "img1"
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No jpg frames found in {frame_dir}")
    cfg = configparser.ConfigParser()
    cfg.read(Path(source_path) / "seqinfo.ini")
    fps = int(cfg["Sequence"]["frameRate"])
    w = int(cfg["Sequence"]["imWidth"])
    h = int(cfg["Sequence"]["imHeight"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(target_path, fourcc, fps, (w, h))
    for idx, f in enumerate(tqdm(frame_files, desc="Rendering")):
        frame = cv2.imread(str(f))
        frame = callback(frame, idx)
        writer.write(frame)
    writer.release()


def _convert_video(source_path: str, target_path: str):
    """Convert to H.264 MP4 using ffmpeg."""
    if FFMPEG_PATH is None:
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install ffmpeg or set the FFMPEG_PATH "
            "environment variable to the ffmpeg binary."
        )
    subprocess.run([
        FFMPEG_PATH, "-y", "-loglevel", "error",
        "-i", source_path, "-vcodec", "libx264", "-crf", "28", target_path,
    ], check=True)


MARGIN_THRESH = 0.01
MARGIN_COLOR = (0, 255, 255)  # Yellow in BGR
FP_COLOR = (0, 0, 255)        # Red in BGR
FN_COLOR = (255, 100, 50)     # Blue in BGR
EXCESS_COLOR = (255, 0, 255)  # Hot magenta BGR — errors SAM introduced
EVENT_DISPLAY_FRAMES = 30      # How long events stay in the feed


def _xyxy_iou(a: tuple, b: tuple) -> float:
    """IoU between two (x1, y1, x2, y2) tuples."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _compute_excess_errors(
    frame_errors: Optional[dict],
    baseline_errors: Optional[dict],
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Compute per-frame excess FP/FN (present in SDE but not DE).

    Excess FN: GT IDs that are FN in SDE but not in DE (set diff).
    Excess FP: SDE FP bboxes with no spatially matching DE FP (IoU > 0.5).

    Returns (excess_fp, excess_fn) — each {frame_idx: set_of_ids}.
    """
    excess_fp: Dict[int, Set[int]] = {}
    excess_fn: Dict[int, Set[int]] = {}
    if not frame_errors or not baseline_errors:
        return excess_fp, excess_fn

    for fidx, errs in frame_errors.items():
        base = baseline_errors.get(fidx, {})

        sde_fn_ids = {e[0] for e in errs.get("fn", [])}
        de_fn_ids = {e[0] for e in base.get("fn", [])}
        efn = sde_fn_ids - de_fn_ids
        if efn:
            excess_fn[fidx] = efn

        sde_fps = errs.get("fp", [])
        de_fps = base.get("fp", [])
        if sde_fps:
            efp = set()
            for pred_id, x1, y1, x2, y2 in sde_fps:
                matched = any(
                    _xyxy_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2)) > 0.5
                    for _, dx1, dy1, dx2, dy2 in de_fps
                )
                if not matched:
                    efp.add(pred_id)
            if efp:
                excess_fp[fidx] = efp

    return excess_fp, excess_fn


def _draw_event_feed(frame: np.ndarray, events: list, x_offset: int, y_offset: int) -> np.ndarray:
    """Draw a scrolling event feed in the top-right corner.

    events: [(text, color_bgr), ...] — most recent last.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    line_h = 18
    w = frame.shape[1]

    for i, (text, color) in enumerate(events):
        y = y_offset + i * line_h
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = w - tw - x_offset
        cv2.rectangle(frame, (x - 3, y - th - 2), (w - x_offset + 3, y + 3), color, -1)
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


def render_tracker_video(
    source_path: str,
    output_path: str,
    merged: Dict[int, Dict[int, TrackData]],
    margins: Optional[Dict[int, Dict[int, float]]] = None,
    margin_thresh: float = MARGIN_THRESH,
    frame_errors: Optional[dict] = None,
    baseline_errors: Optional[dict] = None,
    no_gt: bool = False,
) -> None:
    """Render merged tracks with GT overlay and optional FP/FN highlighting."""
    gt = {} if no_gt else _load_gt(source_path)
    excess_fp, excess_fn = _compute_excess_errors(frame_errors, baseline_errors)

    # Build IDSW lookups: {frame: {gt_id: (old_tid, new_tid)}}
    all_idsw: Dict[int, Dict[int, tuple]] = {}
    excess_idsw: Dict[int, Set[int]] = {}
    if frame_errors:
        for fidx, errs in frame_errors.items():
            for gt_id, old_tid, new_tid, *_ in errs.get("idsw", []):
                all_idsw.setdefault(fidx, {})[gt_id] = (old_tid, new_tid)
        if baseline_errors:
            for fidx, idsw_map in all_idsw.items():
                base_idsw_gt = {e[0] for e in baseline_errors.get(fidx, {}).get("idsw", [])}
                excess = set(idsw_map.keys()) - base_idsw_gt
                if excess:
                    excess_idsw[fidx] = excess

    # Pre-build event feed: only excess errors (SAM-caused)
    event_log = []
    if frame_errors:
        for fidx in sorted(frame_errors):
            errs = frame_errors[fidx]
            efp = excess_fp.get(fidx, set())
            efn = excess_fn.get(fidx, set())
            for pred_id, *_ in errs.get("fp", []):
                if pred_id in efp:
                    event_log.append((fidx, f"+FP T{pred_id} @ {fidx}", EXCESS_COLOR))
            for gt_id, *_ in errs.get("fn", []):
                if gt_id in efn:
                    event_log.append((fidx, f"+FN GT{gt_id} @ {fidx}", EXCESS_COLOR))
            for gt_id, old_tid, new_tid, *_ in errs.get("idsw", []):
                if gt_id in excess_idsw.get(fidx, set()):
                    event_log.append((fidx, f"+SW GT{gt_id} T{old_tid}>T{new_tid} @ {fidx}", EXCESS_COLOR))

    def _fn_gt_ids(index):
        if not frame_errors or index not in frame_errors:
            return set()
        return {e[0] for e in frame_errors[index].get("fn", [])}

    def _fp_pred_ids(index):
        if not frame_errors or index not in frame_errors:
            return set()
        return {e[0] for e in frame_errors[index].get("fp", [])}

    def _excess_fp_ids(index):
        return excess_fp.get(index, set())

    def _excess_fn_ids(index):
        return excess_fn.get(index, set())

    def _get_color(track_id: int, margin: float | None) -> tuple:
        return track_color(track_id)

    def _is_edge_kill(track_id, bbox, index, w, h):
        if merged.get(index + 1, {}).get(track_id) is not None:
            return False
        x1, y1, x2, y2 = bbox
        m = BORDER_MARGIN
        return x1 <= m or y1 <= m or x2 >= w - 1 - m or y2 >= h - 1 - m

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        frame = frame.copy()
        h, w = frame.shape[:2]
        # SAM border (outer, darker)
        s = SAM_BORDER_MARGIN
        frame[:s, :] //= 5
        frame[-s:, :] //= 5
        frame[s:-s, :s] //= 5
        frame[s:-s, -s:] //= 5
        # DE border (inner, lighter) — draw line at DE edge-kill boundary
        b = BORDER_MARGIN
        frame[b, b:-b] = frame[b, b:-b] // 2 + np.array([60, 60, 60], dtype=np.uint8)
        frame[-b-1, b:-b] = frame[-b-1, b:-b] // 2 + np.array([60, 60, 60], dtype=np.uint8)
        frame[b:-b, b] = frame[b:-b, b] // 2 + np.array([60, 60, 60], dtype=np.uint8)
        frame[b:-b, -b-1] = frame[b:-b, -b-1] // 2 + np.array([60, 60, 60], dtype=np.uint8)
        frame_data = merged.get(index, {})
        frame_margins = margins.get(index, {}) if margins else {}
        fp_ids = _fp_pred_ids(index)
        exc_fp = _excess_fp_ids(index)

        for track_id, td in frame_data.items():
            if td.mask is not None:
                is_excess = track_id in exc_fp
                is_fp = track_id in fp_ids
                if is_excess:
                    color = EXCESS_COLOR
                elif is_fp:
                    color = FP_COLOR
                else:
                    color = track_color(track_id)
                overlay = frame.copy()
                overlay[td.mask] = color
                alpha = 0.5 if is_excess else 0.4
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
                contours, _ = cv2.findContours(
                    td.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(frame, contours, -1, color, 4 if is_excess else 2)

        h, w = frame.shape[:2]
        for track_id, td in frame_data.items():
            if td.bbox is not None:
                x1, y1, x2, y2 = int(td.bbox[0]), int(td.bbox[1]), int(td.bbox[2]), int(td.bbox[3])
                edge_kill = _is_edge_kill(track_id, td.bbox, index, w, h)
                is_excess = track_id in exc_fp
                is_fp = track_id in fp_ids
                margin = frame_margins.get(track_id)
                if edge_kill:
                    color = (0, 0, 0)
                elif is_excess:
                    color = EXCESS_COLOR
                elif is_fp:
                    color = FP_COLOR
                else:
                    color = _get_color(track_id, margin)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if is_excess else 2)
                margin_str = f" {margin:.2f}" if margin is not None and margin != float('inf') else ""
                if is_excess:
                    label = f"+FP T{track_id}"
                elif is_fp:
                    label = f"FP T{track_id}"
                else:
                    label = f"T{track_id}{margin_str}"
                _draw_label(frame, label, (x1 + x2) // 2, y1, color)
            elif td.mask is not None and td.mask.any():
                ys, xs = np.where(td.mask)
                x1, y1 = int(xs.min()), int(ys.min())
                x2 = int(xs.max())
                is_excess = track_id in exc_fp
                is_fp = track_id in fp_ids
                if is_excess:
                    color = EXCESS_COLOR
                elif is_fp:
                    color = FP_COLOR
                else:
                    color = track_color(track_id)
                if is_excess:
                    label = f"+FP T{track_id}"
                elif is_fp:
                    label = f"FP T{track_id}"
                else:
                    label = f"T{track_id}"
                _draw_label(frame, label, (x1 + x2) // 2, y1, color)

        # GT boxes: magenta for excess IDSW/FN, blue for shared FN, white otherwise
        fn_ids = _fn_gt_ids(index)
        exc_fn = _excess_fn_ids(index)
        frame_idsw = all_idsw.get(index, {})
        frame_exc_idsw = excess_idsw.get(index, set())
        for gt_id, gx1, gy1, gx2, gy2 in gt.get(index, []):
            is_excess_idsw = gt_id in frame_exc_idsw
            idsw_info = frame_idsw.get(gt_id)
            is_excess_fn = gt_id in exc_fn
            is_fn = gt_id in fn_ids
            if is_excess_idsw:
                old_tid, new_tid = idsw_info
                color = EXCESS_COLOR
                thickness = 4
                label = f"+SW GT{gt_id} T{old_tid}>T{new_tid}"
            elif idsw_info:
                old_tid, new_tid = idsw_info
                color = (255, 255, 255)
                thickness = 2
                label = f"SW GT{gt_id} T{old_tid}>T{new_tid}"
            elif is_excess_fn:
                color = EXCESS_COLOR
                thickness = 4
                label = f"+FN GT{gt_id}"
            elif is_fn:
                color = FN_COLOR
                thickness = 2
                label = f"FN GT{gt_id}"
            else:
                color = (255, 255, 255)
                thickness = 1
                label = f"GT {gt_id}"
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), color, thickness)
            _draw_label(frame, label, (gx1 + gx2) // 2, gy2, color, above=False)

        # Event feed (top-right)
        visible = [(text, color) for (fidx, text, color) in event_log
                    if fidx <= index < fidx + EVENT_DISPLAY_FRAMES]
        if visible:
            # Show most recent 8 events max
            _draw_event_feed(frame, visible[-8:], x_offset=10, y_offset=20)

        cv2.putText(frame, f"Frame: {index}", (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    temp_path = str(Path(output_path).parent / f"temp_{Path(output_path).name}")
    _process_source(source_path, temp_path, callback)
    _convert_video(temp_path, output_path)
    Path(temp_path).unlink()
    print(f"Rendered: {output_path}")
