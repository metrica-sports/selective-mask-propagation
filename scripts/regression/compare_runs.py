"""Compare a results tree against a frozen reference tree.

For every clip present in both trees, reports per-variant metric deltas
(from eval.json), MOT byte-diff status, and window-outcome shifts.
The one-command verdict for "did my change degrade anything, and where".

    uv run python scripts/regression/compare_runs.py refs/val-<sha> results/val
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import selective_mask_propagation.core.sam2  # noqa: E402,F401 — needed for the sam_deep_eiou pickle alias below

sys.modules.setdefault("sam_deep_eiou", sys.modules["selective_mask_propagation"])
sys.modules.setdefault("sam_deep_eiou.core", sys.modules["selective_mask_propagation.core"])
sys.modules.setdefault("sam_deep_eiou.core.sam2", sys.modules["selective_mask_propagation.core.sam2"])

VARIANTS = ["deep_eiou", "sam_deep_eiou", "de_gta", "sde_gta"]
MOT_FILES = ["mot_deep_eiou.txt", "mot_sam_deep_eiou.txt", "mot_sde_gta_interp.txt"]
METRICS = ["hota", "assa", "idf1"]


def _outcomes(artifacts: Path) -> Counter:
    p = artifacts / "windows.pkl"
    if not p.exists():
        return Counter()
    windows = pickle.load(open(p, "rb"))
    return Counter(w.outcome.value for w in windows)


def compare_clip(ref: Path, new: Path) -> dict:
    out = {"mot": {}, "deltas": {}, "outcomes": None}

    for f in MOT_FILES:
        a, b = ref / "artifacts" / f, new / "artifacts" / f
        if a.exists() and b.exists():
            out["mot"][f] = "same" if a.read_bytes() == b.read_bytes() else "DIFF"
        elif a.exists() != b.exists():
            out["mot"][f] = "MISSING"

    re_, ne = ref / "eval.json", new / "eval.json"
    if re_.exists() and ne.exists():
        rj, nj = json.load(open(re_)), json.load(open(ne))
        for v in VARIANTS:
            if v in rj and v in nj:
                out["deltas"][v] = {m: nj[v][m] - rj[v][m] for m in METRICS}
                out["deltas"][v]["idsw"] = nj[v]["idsw"] - rj[v]["idsw"]

    ro, no = _outcomes(ref / "artifacts"), _outcomes(new / "artifacts")
    if ro != no:
        out["outcomes"] = {k: (ro.get(k, 0), no.get(k, 0)) for k in sorted(set(ro) | set(no))
                           if ro.get(k, 0) != no.get(k, 0)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ref", help="Frozen reference tree (e.g. refs/val-<sha>)")
    parser.add_argument("new", help="New results tree (e.g. results/val)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Flag clips with |ΔHOTA| above this (default 0.05)")
    args = parser.parse_args()

    ref_clips = {p.name for p in Path(args.ref).iterdir() if (p / "artifacts").is_dir()}
    new_clips = {p.name for p in Path(args.new).iterdir() if (p / "artifacts").is_dir()}
    both = sorted(ref_clips & new_clips)
    print(f"clips: {len(both)} compared "
          f"({len(ref_clips - new_clips)} only in ref, {len(new_clips - ref_clips)} only in new)\n")

    sums = {v: {m: 0.0 for m in METRICS + ["idsw"]} for v in VARIANTS}
    counts = {v: 0 for v in VARIANTS}
    flagged = []
    mot_diff = Counter()

    for clip in both:
        r = compare_clip(Path(args.ref) / clip, Path(args.new) / clip)
        for f, status in r["mot"].items():
            if status != "same":
                mot_diff[f] += 1
        for v, d in r["deltas"].items():
            counts[v] += 1
            for m in d:
                sums[v][m] += d[m]
        worst = max((abs(d["hota"]), v) for v, d in r["deltas"].items()) if r["deltas"] else (0, "")
        if worst[0] > args.threshold or any(s != "same" for s in r["mot"].values()) or r["outcomes"]:
            flagged.append((clip, r, worst))

    print(f"{'variant':14s} {'ΔHOTA':>8s} {'ΔAssA':>8s} {'ΔIDF1':>8s} {'ΔIDSW':>7s}   (mean over clips)")
    for v in VARIANTS:
        if counts[v]:
            s, n = sums[v], counts[v]
            print(f"{v:14s} {s['hota']/n:+8.3f} {s['assa']/n:+8.3f} {s['idf1']/n:+8.3f} {s['idsw']/n:+7.2f}")

    if mot_diff:
        print("\nMOT diffs:", dict(mot_diff))
    if flagged:
        print(f"\n{len(flagged)} clip(s) flagged:")
        for clip, r, worst in flagged:
            parts = [f"|ΔHOTA|max={worst[0]:.3f} ({worst[1]})"]
            mots = [f for f, s in r["mot"].items() if s != "same"]
            if mots:
                parts.append("mot: " + ",".join(mots))
            if r["outcomes"]:
                parts.append(f"outcomes: {r['outcomes']}")
            print(f"  {clip}: " + " | ".join(parts))
    else:
        print("\nno clips flagged — identical within threshold")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
