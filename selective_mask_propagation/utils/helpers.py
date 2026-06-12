"""Pipeline helpers: step registry, cleanup, FPS reading, input expansion."""

import configparser
import json
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


def _artifact_variants(artifacts_dir: Path, filename: str) -> List[Path]:
    """An artifact may be stored compressed (name.pkl.zst) or plain."""
    paths = [artifacts_dir / filename]
    if filename.endswith(".pkl"):
        paths.append(artifacts_dir / (filename + ".zst"))
    return paths


def step_done(step: str, source_path: str, suffix: str = "", gta: bool = True) -> bool:
    """True if every artifact and output of a step already exists on disk."""
    artifacts_dir = get_artifacts_dir(source_path, suffix)
    output_dir = get_output_dir(source_path, suffix)
    artifacts = STEP_ARTIFACTS.get(step, [])
    outputs = STEP_OUTPUTS.get(step, [])
    if not gta:
        artifacts = [f for f in artifacts if "gta" not in f]
        outputs = [f for f in outputs if "gta" not in f]
    if not artifacts and not outputs:
        return False
    for filename in artifacts:
        if not any(p.exists() for p in _artifact_variants(artifacts_dir, filename)):
            return False
    return all((output_dir / filename).exists() for filename in outputs)


def clean_steps(steps: list, source_path: str, suffix: str = "") -> None:
    artifacts_dir = get_artifacts_dir(source_path, suffix)
    output_dir = get_output_dir(source_path, suffix)
    cleaned = []
    for step in steps:
        for filename in STEP_ARTIFACTS.get(step, []):
            for path in _artifact_variants(artifacts_dir, filename):
                if path.exists():
                    path.unlink()
                    cleaned.append(path.name)
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


def record_timing(step: str, seconds: float, source_path: str, suffix: str = "", **extra) -> None:
    """Record a step's wall-clock (and optional extras) into the clip's timing.json."""
    path = get_artifacts_dir(source_path, suffix) / "timing.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[step] = round(seconds, 2)
    data.update(extra)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{step} wall clock: {seconds:.1f}s")


def expand_input(pattern: str) -> List[str]:
    if any(c in pattern for c in "*?["):
        paths = sorted(Path(".").glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No inputs found matching: {pattern}")
        if len(paths) > 1:
            print(f"Found {len(paths)} sequences")
        return [str(p) for p in paths]
    return [str(Path(pattern))]
