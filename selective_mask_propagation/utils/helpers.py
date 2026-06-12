"""Pipeline helpers: step registry, cleanup, FPS reading, input expansion."""

import configparser
import shutil
from pathlib import Path
from typing import List

from .artifacts import get_output_dir, get_artifacts_dir

STEPS = [
    "detect", "track", "sam", "merge",
    "pose", "jersey", "classify", "embed", "match", "interp",
    "eval", "render",
]

GTA_STEPS = {"pose", "jersey", "classify", "embed", "match", "interp"}

STEP_ARTIFACTS = {
    "detect": ["detections.pkl", "embeddings.pkl"],
    "track": ["tracks.pkl", "margins.pkl", "mot_deep_eiou.txt"],
    "sam": ["sam_masks.pkl", "windows.pkl", "rename_events.pkl", "match_history.pkl"],
    "merge": ["merged.pkl", "renamed_margins.pkl", "mot_sam_deep_eiou.txt"],
    "pose": ["pose_data.pkl"],
    "jersey": ["ocr_results.pkl",
               "jersey_map_de.pkl", "jersey_map_sde.pkl"],
    "classify": ["track_teams_de.pkl", "track_teams_sde.pkl"],
    "embed": ["track_embeddings_de.pkl", "track_embeddings_sde.pkl"],
    "match": ["canonical_to_gta_de.pkl", "canonical_to_gta_sde.pkl",
              "mot_de_gta.txt", "mot_sde_gta.txt"],
    "interp": ["mot_de_gta_interp.txt", "mot_sde_gta_interp.txt"],
    "eval": ["deep_eiou_frame_errors.pkl", "sam_deep_eiou_frame_errors.pkl",
             "de_gta_frame_errors.pkl", "sde_gta_frame_errors.pkl"],
}

STEP_OUTPUTS = {
    "eval": ["eval.md", "eval.json"],
    "render": ["deep_eiou.mp4", "sam_deep_eiou.mp4", "de_gta.mp4", "sde_gta.mp4"],
}

STEP_DEBUG_DIRS = {
    "sam": ["windows"],
    "merge": ["merge"],
    "pose": ["jersey/pose"],
    "jersey": ["jersey/crops", "de/jersey/track", "sde/jersey/track"],
    "classify": ["de/classify", "sde/classify"],
    "embed": ["de/embed", "sde/embed"],
    "match": ["de/gta", "sde/gta"],
}


def read_sequence_info(source_path: str) -> dict:
    seqinfo_path = Path(source_path) / "seqinfo.ini"
    if not seqinfo_path.exists():
        raise FileNotFoundError(
            f"seqinfo.ini not found at {seqinfo_path}. "
            f"Make sure the dataset is downloaded and --input points at a "
            f"sequence directory (see README, Data section)."
        )
    cfg = configparser.ConfigParser()
    cfg.read(seqinfo_path)
    seq = cfg["Sequence"]
    info = {
        "fps": int(seq["frameRate"]),
        "width": int(seq["imWidth"]),
        "height": int(seq["imHeight"]),
        "length": int(seq["seqLength"]),
    }
    print(f"Video: {info['fps']} FPS, {info['width']}x{info['height']}, {info['length']} frames")
    return info


def clean_steps(steps: list, source_path: str, suffix: str = "") -> None:
    artifacts_dir = get_artifacts_dir(source_path, suffix)
    output_dir = get_output_dir(source_path, suffix)
    cleaned = []
    for step in steps:
        for filename in STEP_ARTIFACTS.get(step, []):
            path = artifacts_dir / filename
            if path.exists():
                path.unlink()
                cleaned.append(filename)
        for filename in STEP_OUTPUTS.get(step, []):
            path = output_dir / filename
            if path.exists():
                path.unlink()
                cleaned.append(filename)
        for dirname in STEP_DEBUG_DIRS.get(step, []):
            path = output_dir / "debug" / dirname
            if path.exists():
                shutil.rmtree(path)
                cleaned.append(f"debug/{dirname}/")
    if cleaned:
        print(f"Cleaned: {', '.join(cleaned)}")


def print_step(name: str):
    print(f"\n{'='*60}")
    print(f"Step: {name}")
    print(f"{'='*60}")


def expand_input(pattern: str) -> List[str]:
    if any(c in pattern for c in "*?["):
        paths = sorted(Path(".").glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No inputs found matching: {pattern}")
        if len(paths) > 1:
            print(f"Found {len(paths)} sequences")
        return [str(p) for p in paths]
    return [str(Path(pattern))]
