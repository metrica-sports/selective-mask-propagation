"""TrackEval driver: per-MOT-file evaluation against ground truth.

Walks the per-clip output directories produced by the runners
(``pure_sam3``, ``sam3_deep_eiou*``), runs TrackEval on each MOT file,
and persists the resulting metrics next to the file as ``eval.json``.

Idempotent: a MOT file with an existing ``eval.json`` is skipped unless
``--force`` is passed. Eval failure on one file is logged and the next is
attempted.

CLI:
    uv run python scripts/eval_clip.py <clip_dir>... [--variants pure_sam3,sam3_deep_eiou,...] [--force]

Per (clip, variant) outputs (next to the MOT file):
    eval.json   {"hota": ..., "deta": ..., "assa": ..., "mota": ..., "idf1": ..., "idsw": ..., ...}
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam_deep_eiou.core.eval import _eval_mot  # noqa: E402


VARIANT_MOT_FILES = {
    "deep_eiou":                  ("sam3_deep_eiou",            "mot_deep_eiou.txt"),
    "sam3_deep_eiou":             ("sam3_deep_eiou",            "mot_sam3_deep_eiou.txt"),
    "sam3_deep_eiou-margin":      ("sam3_deep_eiou-margin",     "mot_sam3_deep_eiou.txt"),
    "sam3_deep_eiou-margin_gap":  ("sam3_deep_eiou-margin_gap", "mot_sam3_deep_eiou.txt"),
    "de_uniform_sam3":            ("de_uniform_sam3",           "mot_de_uniform_sam3.txt"),
}


def _eval_one(clip_path: str, variant: str, mot_path: Path, force: bool) -> bool:
    eval_path = mot_path.parent / f"{mot_path.stem}.eval.json"
    seq_name = Path(clip_path).name
    if eval_path.exists() and not force:
        print(f"[{seq_name}/{variant}] skip (eval exists)")
        return True
    if not mot_path.exists():
        print(f"[{seq_name}/{variant}] missing MOT file: {mot_path}")
        return False
    try:
        metrics, _ = _eval_mot(seq_name, clip_path, str(mot_path))
        eval_path.write_text(json.dumps(metrics, indent=2))
        print(f"[{seq_name}/{variant}] HOTA={metrics['hota']:.1f} "
              f"AssA={metrics['assa']:.1f} IDF1={metrics['idf1']:.1f} "
              f"MOTA={metrics['mota']:.1f} IDSW={metrics['idsw']}")
        return True
    except Exception as e:
        print(f"[{seq_name}/{variant}] EVAL FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("clips", nargs="+", help="Clip directories (each contains gt/gt.txt).")
    parser.add_argument("--variants", default=",".join(VARIANT_MOT_FILES.keys()),
                        help="Comma-separated variant names to evaluate (default: all known).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run eval even if eval.json already exists")
    parser.add_argument("--output-root", type=Path, default=Path("results"),
                        help="Root dir to find per-clip outputs (default: results/). "
                             "Per-variant path is <root>/<parent>/<clip>/<subdir>/<mot>.")
    args = parser.parse_args(args_list)

    variants: List[str] = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANT_MOT_FILES]
    if unknown:
        print(f"Unknown variant(s): {unknown}; known: {list(VARIANT_MOT_FILES)}")
        return 1

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for clip_path in args.clips:
        clip = Path(clip_path).name
        parent = Path(clip_path).parent.name
        for variant in variants:
            subdir, mot_name = VARIANT_MOT_FILES[variant]
            mot_path = args.output_root / parent / clip / subdir / mot_name
            eval_path = mot_path.parent / f"{mot_path.stem}.eval.json"
            if eval_path.exists() and not args.force:
                n_skipped += 1
                print(f"[{clip}/{variant}] skip (eval exists)")
                continue
            if _eval_one(clip_path, variant, mot_path, args.force):
                n_done += 1
            else:
                n_failed += 1

    print(f"\nSummary: {n_done} evaluated, {n_skipped} skipped, {n_failed} failed")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
