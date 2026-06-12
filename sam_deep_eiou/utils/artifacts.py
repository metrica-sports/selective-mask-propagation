"""Artifact persistence — pickle save/load for pipeline intermediates."""

import pickle
from pathlib import Path
from typing import Any

COMPRESSED_ARTIFACTS = {"sam_masks", "merged"}


def get_output_dir(source_path: str, suffix: str = "") -> Path:
    """Returns results/{parent_name}/{stem}{suffix}/, creates if needed."""
    source = Path(source_path)
    output_dir = Path("results") / source.parent.name / (source.name + suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_artifacts_dir(source_path: str, suffix: str = "") -> Path:
    """Returns results/{parent_name}/{stem}{suffix}/artifacts/, creates if needed."""
    artifacts_dir = get_output_dir(source_path, suffix) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def save_artifact(name: str, data: Any, source_path: str, suffix: str = "") -> Path:
    """Pickle data to artifacts dir. Uses zstd compression for mask artifacts."""
    artifacts_dir = get_artifacts_dir(source_path, suffix)
    if name in COMPRESSED_ARTIFACTS:
        import zstandard as zstd
        path = artifacts_dir / f"{name}.pkl.zst"
        cctx = zstd.ZstdCompressor(level=3)
        with open(path, "wb") as f:
            with cctx.stream_writer(f) as compressor:
                pickle.dump(data, compressor)
    else:
        path = artifacts_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
    print(f"  Saved: {path}")
    return path


def load_artifact(name: str, source_path: str, suffix: str = "", allow_missing: bool = False) -> Any:
    """Load pickled artifact. Supports both .pkl.zst and .pkl formats."""
    artifacts_dir = get_artifacts_dir(source_path, suffix)
    zst_path = artifacts_dir / f"{name}.pkl.zst"
    pkl_path = artifacts_dir / f"{name}.pkl"
    if zst_path.exists():
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(zst_path, "rb") as f:
            with dctx.stream_reader(f) as reader:
                data = pickle.load(reader)
        print(f"  Loaded: {zst_path}")
        return data
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded: {pkl_path}")
        return data
    if allow_missing:
        return None
    raise FileNotFoundError(f"Artifact not found: {pkl_path}")
