"""VLM-based track classification.

Two-phase approach:
1. Establish team colors from sampled frames (one VLM call).
2. Classify each track from a crop montage (one VLM call per track, async parallel).

Produces track_teams: {track_id: {"team_id": 0|1|None}}.
Tracks classified as "other" (goalkeeper, referee, etc.) get team_id=None —
they stay in the MOT but can't participate in jersey merge or team veto.
"""

import asyncio
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ..utils.export import parse_mot_tracks

MODEL = "gemini-3-flash-preview"

N_COLOR_FRAMES = 5
SAMPLE_INTERVAL = 15
GRID_COLS = 10
CELL_SIZE = (80, 120)


class TeamColors(BaseModel):
    team_a_color: str = Field(description="Primary jersey color of team A")
    team_b_color: str = Field(description="Primary jersey color of team B")


class TrackClassification(BaseModel):
    classification: str = Field(
        description="Track classification: team_a, team_b, or other"
    )


ESTABLISH_PROMPT = """Look at these sports game frames. Identify the primary jersey color for each of the two teams playing.

Team A and Team B are arbitrary labels - just pick one team for each."""


def _get_classification_prompt(colors: TeamColors) -> str:
    return f"""Classify this sports player track montage.

Team A wears {colors.team_a_color}. Team B wears {colors.team_b_color}.

- team_a: Player wearing {colors.team_a_color}
- team_b: Player wearing {colors.team_b_color}
- other: Goalkeeper, referee, coach, sideline, or anyone not on team A/B"""


def _sample_frames(source_path: str) -> List[np.ndarray]:
    img_dir = Path(source_path) / "img1"
    frame_files = sorted(img_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No jpg frames found in {img_dir}")
    indices = np.linspace(0, len(frame_files) - 1, N_COLOR_FRAMES, dtype=int)
    return [cv2.imread(str(frame_files[i])) for i in indices]


OCCLUSION_THRESHOLD = 0.30


def _bbox_occluded(bbox_xyxy, all_bboxes_xyxy, threshold=OCCLUSION_THRESHOLD):
    """Check if another bbox is in front and overlapping ours.

    Uses IoA (intersection / our area) and y2 depth ordering: higher y2
    = closer to camera in elevated sports broadcasts = in front.
    """
    x1, y1, x2, y2 = bbox_xyxy
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return True
    for ox1, oy1, ox2, oy2 in all_bboxes_xyxy:
        if oy2 <= y2:
            continue
        ix1 = max(x1, ox1); iy1 = max(y1, oy1)
        ix2 = min(x2, ox2); iy2 = min(y2, oy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter / area >= threshold:
            return True
    return False


def _generate_all_grids(
    tracks: Dict[int, list],
    source_path: str,
) -> Dict[int, np.ndarray]:
    """Generate crop montage grids for all tracks. Reads each frame once.

    Skips frames where another track's bbox is in front and overlapping,
    so the LLM only sees unoccluded crops for team classification.
    """
    cell_w, cell_h = CELL_SIZE

    # Build frame-level index of all track bboxes (for occlusion check)
    frame_all_bboxes: Dict[int, Dict[int, Tuple[float, float, float, float]]] = defaultdict(dict)
    for tid, entries in tracks.items():
        for frame_idx, bbox in entries:
            x, y, w, h = bbox
            frame_all_bboxes[frame_idx][tid] = (x, y, x + w, y + h)

    track_samples: Dict[int, list] = {}
    for tid, entries in tracks.items():
        sampled = [e for i, e in enumerate(entries) if i % SAMPLE_INTERVAL == 0]
        if not sampled:
            sampled = entries[:1]
        track_samples[tid] = sampled

    # Filter occluded samples; fall back to unfiltered if all are occluded
    filtered_samples: Dict[int, list] = {}
    for tid, samples in track_samples.items():
        clean = []
        for sample_idx, (frame_idx, bbox) in enumerate(samples):
            x, y, w, h = bbox
            our_xyxy = (x, y, x + w, y + h)
            others = [b for t, b in frame_all_bboxes[frame_idx].items() if t != tid]
            if not _bbox_occluded(our_xyxy, others):
                clean.append((sample_idx, frame_idx, x, y, w, h))
        if not clean:
            clean = [(i, f, *b) for i, (f, b) in enumerate(samples)]
        filtered_samples[tid] = clean

    frame_to_crops: Dict[int, list] = defaultdict(list)
    for tid, entries in filtered_samples.items():
        for sample_idx, frame_idx, x, y, w, h in entries:
            frame_to_crops[frame_idx].append((tid, sample_idx, x, y, w, h))

    track_crops: Dict[int, list] = {
        tid: [None] * len(samples) for tid, samples in track_samples.items()
    }

    img_dir = Path(source_path) / "img1"
    frame_files = sorted(img_dir.glob("*.jpg"))

    for frame_idx in sorted(frame_to_crops.keys()):
        if frame_idx >= len(frame_files):
            continue
        frame = cv2.imread(str(frame_files[frame_idx]))
        if frame is None:
            continue

        for tid, sample_idx, x, y, w, h in frame_to_crops[frame_idx]:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2 = min(frame.shape[1], int(x + w))
            y2 = min(frame.shape[0], int(y + h))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                track_crops[tid][sample_idx] = cv2.resize(crop, (cell_w, cell_h))

    grids: Dict[int, np.ndarray] = {}
    for tid, crops in track_crops.items():
        valid = [c for c in crops if c is not None]
        if not valid:
            continue
        n = len(valid)
        n_rows = (n + GRID_COLS - 1) // GRID_COLS
        grid = np.zeros((n_rows * cell_h, GRID_COLS * cell_w, 3), dtype=np.uint8)
        for idx, crop in enumerate(valid):
            row, col = idx // GRID_COLS, idx % GRID_COLS
            gy, gx = row * cell_h, col * cell_w
            grid[gy : gy + cell_h, gx : gx + cell_w] = crop
        grids[tid] = grid

    return grids


def _establish_colors(frames: List[np.ndarray], client: Any) -> TeamColors:
    from google.genai import types

    image_parts = []
    for frame in frames:
        _, jpeg_bytes = cv2.imencode(".jpg", frame)
        image_parts.append(
            types.Part.from_bytes(data=jpeg_bytes.tobytes(), mime_type="image/jpeg")
        )

    response = client.models.generate_content(
        model=MODEL,
        contents=[ESTABLISH_PROMPT] + image_parts,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": TeamColors.model_json_schema(),
        },
    )

    return TeamColors.model_validate_json(response.text)


async def _classify_track(
    track_id: int,
    grid_image: np.ndarray,
    prompt: str,
    client: Any,
    max_retries: int = 3,
) -> Tuple[int, str]:
    from google.genai import types

    _, jpeg_bytes = cv2.imencode(".jpg", grid_image)
    image_part = types.Part.from_bytes(
        data=jpeg_bytes.tobytes(), mime_type="image/jpeg"
    )

    for attempt in range(max_retries):
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=[prompt, image_part],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": TrackClassification.model_json_schema(),
            },
        )
        if response.text is not None:
            result = TrackClassification.model_validate_json(response.text)
            return track_id, result.classification
        if attempt < max_retries - 1:
            await asyncio.sleep(1)

    print(f"  Warning: Gemini returned no response for track {track_id}, defaulting to 'other'")
    return track_id, "other"


def classify_tracks(
    source_path: str,
    mot_path: str,
) -> Tuple[Dict[int, dict], dict]:
    """Classify tracks by team via VLM.

    Returns:
        track_teams: {track_id: {"team_id": 0|1|None}}
        debug_info: {colors, color_frames, classifications, grids}
    """
    return asyncio.run(_classify_tracks_async(source_path, mot_path))


async def _classify_tracks_async(
    source_path: str,
    mot_path: str,
) -> Tuple[Dict[int, dict], dict]:
    load_dotenv()
    from google import genai

    keys = [v for k, v in os.environ.items()
            if k.startswith("GEMINI_API_KEY") and v]
    if not keys:
        raise RuntimeError("GEMINI_API_KEY not set")
    clients = [genai.Client(api_key=k) for k in keys]
    print(f"  Using {len(clients)} Gemini API key(s)")

    print("Sampling frames for color establishment...")
    frames = _sample_frames(source_path)

    print(f"Establishing team colors with {MODEL}...")
    colors = _establish_colors(frames, clients[0])
    print(f"  Team A: {colors.team_a_color}, Team B: {colors.team_b_color}")

    tracks = parse_mot_tracks(mot_path)
    print(f"Generating montage grids for {len(tracks)} tracks...")
    grids = _generate_all_grids(tracks, source_path)

    prompt = _get_classification_prompt(colors)

    print(f"Classifying {len(grids)} tracks with {MODEL}...")
    sorted_grids = sorted(grids.items())
    tasks = []
    for i, (tid, grid) in enumerate(sorted_grids):
        key_idx = i % len(clients)
        print(f"  T{tid} -> key {key_idx}")
        tasks.append(_classify_track(tid, grid, prompt, clients[key_idx]))
    results = await asyncio.gather(*tasks)

    track_teams: Dict[int, dict] = {}
    for tid, classification in results:
        print(f"  T{tid:03d}: {classification}")
        if classification == "team_a":
            track_teams[tid] = {"team_id": 0}
        elif classification == "team_b":
            track_teams[tid] = {"team_id": 1}
        else:
            track_teams[tid] = {"team_id": None}

    classifications = {tid: cls for tid, cls in results}

    n_a = sum(1 for t in track_teams.values() if t["team_id"] == 0)
    n_b = sum(1 for t in track_teams.values() if t["team_id"] == 1)
    n_other = sum(1 for t in track_teams.values() if t["team_id"] is None)
    print(f"Classification: {n_a} team_a, {n_b} team_b, {n_other} other")

    debug_info = {
        "colors": colors.model_dump(),
        "color_frames": frames,
        "classifications": classifications,
        "grids": grids,
    }

    return track_teams, debug_info
