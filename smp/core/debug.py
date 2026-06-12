"""Debug outputs for SAM-Deep-EIoU windows and merge."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .sam2 import WindowOutcome, SamWindow
from .merge import TrackData


WHITE = (255, 255, 255)
MARGIN_COLOR = (0, 0, 255)  # Red in BGR
HIGHLIGHT = (0, 255, 0)  # Green in BGR

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


def _read_frames(source_path: str, frame_indices) -> Dict[int, np.ndarray]:
    frame_dir = Path(source_path) / "img1"
    frame_files = sorted(frame_dir.glob("*.jpg"))
    cache = {}
    for fidx in sorted(frame_indices):
        if 0 <= fidx < len(frame_files):
            frame = cv2.imread(str(frame_files[fidx]))
            if frame is not None:
                cache[fidx] = frame
    return cache


def _crop_around_bbox(frame: np.ndarray, bbox: np.ndarray, pad_fraction: float = 0.5,
                      mask=None) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        x1 = min(x1, int(xs.min()))
        y1 = min(y1, int(ys.min()))
        x2 = max(x2, int(xs.max()))
        y2 = max(y2, int(ys.max()))

    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * pad_fraction), int(bh * pad_fraction)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    return frame[cy1:cy2, cx1:cx2].copy()


def _draw_bbox(frame: np.ndarray, bbox: np.ndarray, color: tuple, label: str = "") -> None:
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4) -> None:
    overlay = frame.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 2)


def _make_black_panel(size: tuple, text: str) -> np.ndarray:
    """Create a black panel with centered white text."""
    h, w = size
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cx, cy = (w - tw) // 2, (h + th) // 2
    cv2.putText(panel, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return panel


def save_windows_debug(
    windows: List[SamWindow],
    tracks: Dict[int, Dict[int, np.ndarray]],
    margins: Dict[int, Dict[int, float]],
    sam_masks: Dict[int, Dict[int, np.ndarray]],
    source_path: str,
    output_dir,
) -> None:
    output_dir = Path(output_dir)
    debug_dir = output_dir / "debug" / "windows"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not windows:
        (debug_dir / "windows_log.json").write_text("[]")
        print(f"Windows debug saved to: {debug_dir}")
        return

    frames_needed = set()
    for w in windows:
        frames_needed.add(w.seed_frame)
        frames_needed.add(w.seed_frame + 1)
        frames_needed.add(w.entry_frame)
        frames_needed.add(w.exit_frame)
    frame_cache = _read_frames(source_path, frames_needed)

    log = []
    for i, w in enumerate(windows):
        _save_window_panels(i, w, tracks, margins, sam_masks, frame_cache, debug_dir)
        _save_window_mask_grid(i, w, tracks, sam_masks, source_path, debug_dir)

        mask_frames = 0
        for fidx in range(w.seed_frame, w.exit_frame + 1):
            if w.canonical_id in sam_masks.get(fidx, {}):
                mask_frames += 1

        log.append({
            "window_idx": i,
            "before_id": w.before_id,
            "canonical_id": w.canonical_id,
            "seed_frame": w.seed_frame,
            "entry_frame": w.entry_frame,
            "exit_frame": w.exit_frame,
            "outcome": w.outcome.value,
            "after_id": w.after_id,
            "swap": w.after_id is not None and w.after_id != w.canonical_id,
            "mask_frames": mask_frames,
        })

    (debug_dir / "windows_log.json").write_text(json.dumps(log, indent=2))
    print(f"Windows debug saved to: {debug_dir}")


def _save_window_panels(
    idx: int,
    window: SamWindow,
    tracks: Dict[int, Dict[int, np.ndarray]],
    margins: Dict[int, Dict[int, float]],
    sam_masks: Dict[int, Dict[int, np.ndarray]],
    frame_cache: Dict[int, np.ndarray],
    debug_dir: Path,
) -> None:
    """Save 4-panel image: seed | seed+1 | entry | exit."""
    raw_id = window.before_id
    cid = window.canonical_id
    color = track_color(cid)
    panels = []

    ref_bbox = tracks.get(window.entry_frame, {}).get(raw_id)
    if ref_bbox is None:
        return

    seed_img = frame_cache.get(window.seed_frame)
    if seed_img is not None:
        seed_img = seed_img.copy()
        seed_bbox = tracks.get(window.seed_frame, {}).get(raw_id)
        if seed_bbox is not None:
            _draw_bbox(seed_img, seed_bbox, color)
            seed_margin = margins.get(window.seed_frame, {}).get(raw_id)
            panel = _crop_around_bbox(seed_img, seed_bbox)
            margin_str = f" m={seed_margin:.3f}" if seed_margin is not None else ""
            cv2.putText(panel, f"SEED f{window.seed_frame}{margin_str}", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT, 1, cv2.LINE_AA)
            panels.append(panel)

    seed_next = window.seed_frame + 1
    next_img = frame_cache.get(seed_next)
    if next_img is not None:
        next_img = next_img.copy()
        next_bbox = tracks.get(seed_next, {}).get(raw_id, ref_bbox)
        mask = sam_masks.get(seed_next, {}).get(cid)
        if mask is not None:
            _overlay_mask(next_img, mask, color)
        _draw_bbox(next_img, next_bbox, color)
        next_margin = margins.get(seed_next, {}).get(raw_id)
        panel = _crop_around_bbox(next_img, next_bbox)
        margin_str = f" m={next_margin:.3f}" if next_margin is not None else ""
        cv2.putText(panel, f"SEED+1 f{seed_next}{margin_str}", (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT, 1, cv2.LINE_AA)
        panels.append(panel)

    entry_img = frame_cache.get(window.entry_frame)
    if entry_img is not None:
        entry_img = entry_img.copy()
        entry_bbox = tracks.get(window.entry_frame, {}).get(raw_id, ref_bbox)
        entry_mask = sam_masks.get(window.entry_frame, {}).get(cid)
        if entry_mask is not None:
            _overlay_mask(entry_img, entry_mask, color)
        _draw_bbox(entry_img, entry_bbox, MARGIN_COLOR)
        entry_margin = margins.get(window.entry_frame, {}).get(raw_id)
        panel = _crop_around_bbox(entry_img, entry_bbox, mask=entry_mask)
        margin_str = f" m={entry_margin:.3f}" if entry_margin is not None else ""
        cv2.putText(panel, f"ENTRY f{window.entry_frame}{margin_str}", (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MARGIN_COLOR, 1, cv2.LINE_AA)
        panels.append(panel)

    exit_img = frame_cache.get(window.exit_frame)
    if exit_img is not None:
        exit_img = exit_img.copy()
        exit_bbox = tracks.get(window.exit_frame, {}).get(raw_id, ref_bbox)
        exit_mask = sam_masks.get(window.exit_frame, {}).get(cid)
        if exit_mask is not None:
            _overlay_mask(exit_img, exit_mask, color)
        _draw_bbox(exit_img, exit_bbox, color)
        exit_margin = margins.get(window.exit_frame, {}).get(raw_id)
        panel = _crop_around_bbox(exit_img, exit_bbox, mask=exit_mask)
        margin_str = f" m={exit_margin:.3f}" if exit_margin is not None else ""
        cv2.putText(panel, f"EXIT f{window.exit_frame} {window.outcome.value}{margin_str}", (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT, 1, cv2.LINE_AA)
        panels.append(panel)

    if not panels:
        return

    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    if window.outcome == WindowOutcome.SWAP:
        status = f"SWAP → T{window.after_id}"
    else:
        status = window.outcome.value.upper()
    header_text = f"T{cid} | {status}"
    grid = np.concatenate(padded, axis=1)
    cv2.putText(grid, header_text, (4, grid.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"window_{idx:03d}_t{cid}.jpg"), grid)


def _save_window_mask_grid(
    idx: int,
    window: SamWindow,
    tracks: Dict[int, Dict[int, np.ndarray]],
    sam_masks: Dict[int, Dict[int, np.ndarray]],
    source_path: str,
    debug_dir: Path,
    grid_cols: int = 10,
    cell_size: tuple = (100, 140),
    min_samples: int = 10,
) -> None:
    """Save per-window mask grid: sampled crops across the window."""
    raw_id = window.before_id
    cid = window.canonical_id
    color = track_color(cid)

    mask_frames = []
    for fidx in range(window.seed_frame, window.exit_frame + 1):
        if cid in sam_masks.get(fidx, {}):
            mask_frames.append(fidx)

    if not mask_frames:
        return

    n = len(mask_frames)
    if n <= min_samples:
        sampled = mask_frames
    else:
        step = max(1, n // min_samples)
        sampled = [mask_frames[i] for i in range(0, n, step)]
        if mask_frames[-1] not in sampled:
            sampled.append(mask_frames[-1])

    frame_cache = _read_frames(source_path, sampled)

    cell_w, cell_h = cell_size
    n_cells = len(sampled)
    n_rows = (n_cells + grid_cols - 1) // grid_cols
    grid = np.zeros((n_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8)

    for ci, fidx in enumerate(sampled):
        frame = frame_cache.get(fidx)
        if frame is None:
            continue

        frame = frame.copy()
        mask = sam_masks.get(fidx, {}).get(cid)
        if mask is not None:
            _overlay_mask(frame, mask, color)

        bbox = tracks.get(fidx, {}).get(raw_id)
        if bbox is not None:
            _draw_bbox(frame, bbox, color)
            crop = _crop_around_bbox(frame, bbox, pad_fraction=0.4)
        elif mask is not None and mask.any():
            ys, xs = np.where(mask)
            mask_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            crop = _crop_around_bbox(frame, mask_bbox, pad_fraction=0.4)
        else:
            continue

        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (cell_w, cell_h))

        cv2.putText(crop, f"f{fidx}", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)

        row, col = ci // grid_cols, ci % grid_cols
        grid[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = crop

    cv2.imwrite(str(debug_dir / f"window_{idx:03d}_t{cid}_masks.jpg"), grid)


def save_merge_debug(
    windows: List[SamWindow],
    merged: Dict[int, Dict[int, TrackData]],
    tracks: Dict[int, Dict[int, np.ndarray]],
    renamed_margins: Dict[int, Dict[int, float]],
    sam_masks: Dict[int, Dict[int, np.ndarray]],
    rename_events: List[Tuple[int, int, int]],
    source_path: str,
    output_dir,
) -> None:
    """Save merge debug: post-merge state with canonical IDs and renames applied."""
    output_dir = Path(output_dir)
    debug_dir = output_dir / "debug" / "merge"
    debug_dir.mkdir(parents=True, exist_ok=True)

    swap_windows = [w for w in windows if w.outcome == WindowOutcome.SWAP]

    log = []
    for raw_id, canonical_id, effective_frame in rename_events:
        log.append({
            "raw_track": raw_id,
            "canonical_id": canonical_id,
            "effective_frame": effective_frame,
        })
    log_data = {"renames": log, "swaps": []}

    if not swap_windows:
        (debug_dir / "merge_log.json").write_text(json.dumps(log_data, indent=2))
        print(f"Merge debug saved to: {debug_dir}")
        return

    frames_needed = set()
    for w in swap_windows:
        frames_needed.add(w.seed_frame)
        frames_needed.add(w.entry_frame)
        mid = (w.entry_frame + w.exit_frame) // 2
        frames_needed.add(mid)
        frames_needed.add(w.exit_frame)
        frames_needed.add(w.exit_frame + 1)
    frame_cache = _read_frames(source_path, frames_needed)

    for i, w in enumerate(swap_windows):
        _save_merge_panels(i, w, merged, tracks, renamed_margins, sam_masks,
                           frame_cache, debug_dir)
        _save_merge_mask_grid(i, w, merged, tracks, sam_masks,
                              source_path, debug_dir)

        log_data["swaps"].append({
            "window_idx": i,
            "canonical_id": w.canonical_id,
            "after_id": w.after_id,
            "entry_frame": w.entry_frame,
            "exit_frame": w.exit_frame,
            "rename": f"T{w.after_id} → canonical {w.canonical_id} from f{w.entry_frame}",
        })

    (debug_dir / "merge_log.json").write_text(json.dumps(log_data, indent=2))
    print(f"Merge debug saved to: {debug_dir}")


def _merged_bbox(merged, frame_idx, canonical_id):
    """Get the bbox for a canonical ID from the merged dict."""
    td = merged.get(frame_idx, {}).get(canonical_id)
    if td is not None and td.bbox is not None:
        return td.bbox
    return None


def _save_merge_panels(
    idx: int,
    window: SamWindow,
    merged: Dict[int, Dict[int, TrackData]],
    tracks: Dict[int, Dict[int, np.ndarray]],
    renamed_margins: Dict[int, Dict[int, float]],
    sam_masks: Dict[int, Dict[int, np.ndarray]],
    frame_cache: Dict[int, np.ndarray],
    debug_dir: Path,
) -> None:
    """Save 5-panel image: seed | entry | mid-auth | exit | post-exit.

    All labels use canonical IDs. Shows the merged state: renamed bboxes,
    SAM masks, and the after_id track that gets renamed.
    """
    cid = window.canonical_id
    after_id = window.after_id
    color = track_color(cid)
    after_color = track_color(after_id) if after_id is not None else WHITE
    panels = []

    key_frames = [
        (window.seed_frame, "SEED"),
        (window.entry_frame, "ENTRY"),
        ((window.entry_frame + window.exit_frame) // 2, "MID-AUTH"),
        (window.exit_frame, "EXIT"),
        (window.exit_frame + 1, "POST-EXIT"),
    ]

    for fidx, label in key_frames:
        img = frame_cache.get(fidx)
        if img is None:
            panels.append(_make_black_panel((200, 150), f"{label} f{fidx}"))
            continue

        img = img.copy()
        is_seed = (label == "SEED")

        # Draw SAM mask for this canonical ID
        mask = sam_masks.get(fidx, {}).get(cid)
        if mask is not None and mask.any():
            _overlay_mask(img, mask, color)

        if is_seed:
            # Seed panel: show the raw bbox that SAM was actually seeded on
            raw_bbox = tracks.get(fidx, {}).get(window.before_id)
            if raw_bbox is not None:
                _draw_bbox(img, raw_bbox, color, f"T{cid}(seed)")
            crop_bbox = raw_bbox
        else:
            # Draw merged bboxes: canonical ID and after_id
            for draw_id, draw_color, draw_label in [
                (cid, color, f"T{cid}"),
                (after_id, after_color, f"T{after_id}(raw)"),
            ]:
                if draw_id is None:
                    continue
                td = merged.get(fidx, {}).get(draw_id)
                if td is not None and td.bbox is not None:
                    _draw_bbox(img, td.bbox, draw_color, draw_label)

            # Crop around the area of interest
            crop_bbox = _merged_bbox(merged, fidx, cid)
            if crop_bbox is None:
                crop_bbox = _merged_bbox(merged, fidx, after_id)

        if crop_bbox is None and mask is not None and mask.any():
            ys, xs = np.where(mask)
            crop_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        if crop_bbox is None:
            panels.append(_make_black_panel((200, 150), f"{label} f{fidx}"))
            continue

        panel = _crop_around_bbox(img, crop_bbox, pad_fraction=0.5, mask=mask)

        margin = renamed_margins.get(fidx, {}).get(cid)
        margin_str = f" m={margin:.3f}" if margin is not None and margin != float('inf') else ""
        cv2.putText(panel, f"{label} f{fidx}{margin_str}", (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT, 1, cv2.LINE_AA)
        panels.append(panel)

    if not panels:
        return

    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    header = f"T{cid} | SWAP T{after_id} → T{cid} from f{window.entry_frame}"
    grid = np.concatenate(padded, axis=1)
    cv2.putText(grid, header, (4, grid.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"swap_{idx:03d}_t{cid}.jpg"), grid)


def _save_merge_mask_grid(
    idx: int,
    window: SamWindow,
    merged: Dict[int, Dict[int, TrackData]],
    tracks: Dict[int, Dict[int, np.ndarray]],
    sam_masks: Dict[int, Dict[int, np.ndarray]],
    source_path: str,
    debug_dir: Path,
    grid_cols: int = 10,
    cell_size: tuple = (100, 140),
    min_samples: int = 10,
) -> None:
    """Save per-swap mask grid showing merged state across the window."""
    cid = window.canonical_id
    after_id = window.after_id
    color = track_color(cid)
    after_color = track_color(after_id) if after_id is not None else WHITE

    # Sample frames from entry through exit+5
    all_frames = list(range(window.entry_frame, min(window.exit_frame + 6,
                            max(merged.keys()) + 1)))
    if not all_frames:
        return

    n = len(all_frames)
    if n <= min_samples:
        sampled = all_frames
    else:
        step = max(1, n // min_samples)
        sampled = [all_frames[i] for i in range(0, n, step)]
        if all_frames[-1] not in sampled:
            sampled.append(all_frames[-1])

    frame_cache = _read_frames(source_path, sampled)

    cell_w, cell_h = cell_size
    n_cells = len(sampled)
    n_rows = (n_cells + grid_cols - 1) // grid_cols
    grid = np.zeros((n_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8)

    for ci, fidx in enumerate(sampled):
        frame = frame_cache.get(fidx)
        if frame is None:
            continue
        frame = frame.copy()

        # Draw mask
        mask = sam_masks.get(fidx, {}).get(cid)
        if mask is not None and mask.any():
            _overlay_mask(frame, mask, color)

        # Draw merged bboxes
        crop_bbox = None
        for draw_id, draw_color in [(cid, color), (after_id, after_color)]:
            if draw_id is None:
                continue
            td = merged.get(fidx, {}).get(draw_id)
            if td is not None and td.bbox is not None:
                _draw_bbox(frame, td.bbox, draw_color, f"T{draw_id}")
                if draw_id == cid:
                    crop_bbox = td.bbox

        if crop_bbox is None and after_id is not None:
            crop_bbox = _merged_bbox(merged, fidx, after_id)
        if crop_bbox is None and mask is not None and mask.any():
            ys, xs = np.where(mask)
            crop_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        if crop_bbox is None:
            continue

        crop = _crop_around_bbox(frame, crop_bbox, pad_fraction=0.4, mask=mask)
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (cell_w, cell_h))

        zone = "AUTH" if window.entry_frame <= fidx <= window.exit_frame else "POST"
        cv2.putText(crop, f"f{fidx} {zone}", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)

        row, col = ci // grid_cols, ci % grid_cols
        grid[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = crop

    cv2.imwrite(str(debug_dir / f"swap_{idx:03d}_t{cid}_grid.jpg"), grid)
