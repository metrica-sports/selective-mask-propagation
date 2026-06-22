"""Build a SportsMOT submission zip from pipeline artifacts.

Usage:
    cd SAM-Deep-EIoU
    uv run python scripts/reproduce/build_submission.py
    uv run python scripts/reproduce/build_submission.py --variant de_gta
"""

import argparse
import zipfile
from pathlib import Path

RESULTS_DIR = Path("results") / "test"

VARIANTS = {
    "sde_gta": "mot_sde_gta_interp.txt",
    "de_gta": "mot_de_gta_interp.txt",
    "sde": "mot_sam_deep_eiou.txt",
    "de": "mot_deep_eiou.txt",
}


def main():
    parser = argparse.ArgumentParser(description="Build SportsMOT submission zip")
    parser.add_argument("--variant", choices=VARIANTS.keys(), default="sde_gta",
                        help="Which MOT variant to submit (default: sde_gta)")
    parser.add_argument("--sam3", action="store_true",
                        help="Use SAM3 results (dirs ending in -sam3)")
    args = parser.parse_args()

    suffix = "-sam3" if args.sam3 else ""
    mot_filename = VARIANTS[args.variant]
    clip_dirs = sorted(p for p in RESULTS_DIR.iterdir()
                       if p.is_dir() and p.name.endswith(suffix) == bool(suffix))
    print(f"Building submission for {args.variant}{suffix} ({mot_filename})")
    print(f"Found {len(clip_dirs)} clip directories in {RESULTS_DIR}")

    tag = f"{args.variant}{suffix}"
    zip_path = RESULTS_DIR / f"submission-{tag}.zip"
    missing = []
    count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for clip_dir in clip_dirs:
            mot_path = clip_dir / "artifacts" / mot_filename
            if not mot_path.exists():
                missing.append(clip_dir.name)
                continue
            clip_name = clip_dir.name.removesuffix("-sam3")
            zf.write(mot_path, f"{clip_name}.txt")
            count += 1

    if missing:
        print(f"WARNING: {len(missing)} clips missing {mot_filename}:")
        for name in missing[:10]:
            print(f"  {name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    print(f"\n{count}/150 sequences -> {zip_path}")


if __name__ == "__main__":
    main()
