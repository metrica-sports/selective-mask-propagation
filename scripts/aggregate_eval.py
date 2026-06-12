"""Aggregate eval.json files from a SportsMOT val set run.

Usage:
    uv run python scripts/aggregate_eval.py results/val
    uv run python scripts/aggregate_eval.py results/val --phase sam_deep_eiou
    uv run python scripts/aggregate_eval.py results/val --sports data/sportsmot/splits_txt
"""

import argparse
import json
import sys
from pathlib import Path


ALL_PHASES = ["deep_eiou", "sam_deep_eiou", "de_gta", "sde_gta"]
PHASE_LABELS = {
    "deep_eiou": "Deep-EIoU",
    "sam_deep_eiou": "SAM-DE",
    "de_gta": "DE+GTA",
    "sde_gta": "SDE+GTA",
}
SUMMARY_METRICS = ["hota", "deta", "assa", "mota", "idf1"]
COUNT_METRICS = ["idsw", "clr_tp", "clr_fp", "clr_fn", "frag"]


def load_evals(val_dir: str, suffix: str = "") -> list:
    results = []
    for p in sorted(Path(val_dir).glob("*/eval.json")):
        dirname = p.parent.name
        if suffix and not dirname.endswith(suffix):
            continue
        results.append(json.loads(p.read_text()))
    return results


def load_sport_map(splits_dir: str) -> dict:
    clip_to_sport = {}
    for sport in ["basketball", "football", "volleyball"]:
        path = Path(splits_dir) / f"{sport}.txt"
        if not path.exists():
            continue
        for line in path.read_text().strip().split("\n"):
            clip_to_sport[line.strip()] = sport
    return clip_to_sport


def get_available_phases(results: list) -> list:
    return [p for p in ALL_PHASES if any(p in r for r in results)]


def aggregate_summary(results: list, phase: str) -> dict:
    available = [r for r in results if phase in r]
    if not available:
        return {}
    n = len(available)
    means = {k: sum(r[phase][k] for r in available) / n for k in SUMMARY_METRICS}
    totals = {k: sum(r[phase][k] for r in available) for k in COUNT_METRICS}
    return {"n": n, **means, **totals}


def print_comparison(results: list, phases: list, label: str = "") -> None:
    if label:
        print(label)
    print(f"{'Metric':<8}", end="")
    for p in phases:
        print(f" {PHASE_LABELS[p]:>12}", end="")
    print()
    print("-" * (8 + 13 * len(phases)))

    summaries = {p: aggregate_summary(results, p) for p in phases}

    for metric in SUMMARY_METRICS:
        print(f"{metric.upper():<8}", end="")
        for p in phases:
            s = summaries[p]
            if s:
                print(f" {s[metric]:12.1f}", end="")
            else:
                print(f" {'—':>12}", end="")
        print()

    print(f"{'IDSW':<8}", end="")
    for p in phases:
        s = summaries[p]
        if s:
            print(f" {s['idsw']:12d}", end="")
        else:
            print(f" {'—':>12}", end="")
    print()

    print(f"{'FN':<8}", end="")
    for p in phases:
        s = summaries[p]
        if s:
            print(f" {s['clr_fn']:12d}", end="")
        else:
            print(f" {'—':>12}", end="")
    print()

    print(f"{'FP':<8}", end="")
    for p in phases:
        s = summaries[p]
        if s:
            print(f" {s['clr_fp']:12d}", end="")
        else:
            print(f" {'—':>12}", end="")
    print()

    print()
    for p in phases:
        s = summaries[p]
        if s:
            print(f"  {PHASE_LABELS[p]}: {s['n']}/{len(results)} clips")


def print_detail(results: list, phases: list) -> None:
    available = [r for r in results if all(p in r for p in phases)]
    if not available:
        print("No results found.")
        return

    de_phase = phases[0]
    sde_phase = phases[1] if len(phases) > 1 else None

    clips = []
    for r in available:
        de = r[de_phase]
        row = {"name": r["seq_name"], "de": de}
        if sde_phase:
            sde = r[sde_phase]
            row["sde"] = sde
            row["d_hota"] = sde["hota"] - de["hota"]
            row["d_assa"] = sde["assa"] - de["assa"]
            row["d_fn"] = sde["clr_fn"] - de["clr_fn"]
            row["d_idsw"] = sde["idsw"] - de["idsw"]
        clips.append(row)

    if sde_phase:
        clips.sort(key=lambda x: x["d_hota"])
    else:
        clips.sort(key=lambda x: x["de"]["hota"])

    if sde_phase:
        print(f"{'Clip':<30} {'DE':>6} {'SDE':>6} {'dHOTA':>6} {'dAssA':>6} {'dFN':>6} {'dIDSW':>6}")
        print("-" * 72)
        for c in clips:
            print(f"{c['name']:<30} {c['de']['hota']:6.1f} {c['sde']['hota']:6.1f} {c['d_hota']:>+6.1f} {c['d_assa']:>+6.1f} {c['d_fn']:>+6d} {c['d_idsw']:>+6d}")
    else:
        print(f"{'Clip':<30} {'HOTA':>6} {'DetA':>6} {'AssA':>6} {'MOTA':>6} {'IDF1':>6} {'IDSW':>6}")
        print("-" * 72)
        for c in clips:
            m = c["de"]
            print(f"{c['name']:<30} {m['hota']:6.1f} {m['deta']:6.1f} {m['assa']:6.1f} {m['mota']:6.1f} {m['idf1']:6.1f} {m['idsw']:6d}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate SportsMOT eval results.")
    parser.add_argument("val_dir", help="Directory containing per-clip result dirs with eval.json")
    parser.add_argument("--phase", choices=ALL_PHASES,
                        help="Show per-clip detail for a specific phase")
    parser.add_argument("--sports", metavar="SPLITS_DIR",
                        help="Path to splits_txt dir for per-sport breakdown")
    parser.add_argument("--suffix", default="",
                        help="Filter result dirs by suffix (e.g. -sam3, -sam3-noreid)")
    args = parser.parse_args()

    results = load_evals(args.val_dir, args.suffix)
    if not results:
        print(f"No eval.json files found in {args.val_dir}/*/")
        sys.exit(1)

    print(f"Found {len(results)} eval.json files\n")

    phases = get_available_phases(results)

    if args.phase:
        print(f"── {PHASE_LABELS[args.phase]} per-clip ──\n")
        print_detail(results, [args.phase])
    else:
        print_comparison(results, phases)
        print("\n\n── Deep-EIoU vs SAM-Deep-EIoU per-clip ──\n")
        print_detail(results, ["deep_eiou", "sam_deep_eiou"])

    if args.sports:
        clip_to_sport = load_sport_map(args.sports)

        sport_results = {}
        unmapped = []
        for r in results:
            sport = clip_to_sport.get(r["seq_name"])
            if sport:
                sport_results.setdefault(sport, []).append(r)
            else:
                unmapped.append(r["seq_name"])

        if unmapped:
            print(f"\nWarning: {len(unmapped)} clips not in any sport split: {', '.join(unmapped[:5])}")

        print(f"\n{'='*60}")
        print("Per-Sport Breakdown")
        print(f"{'='*60}")

        for sport in ["basketball", "football", "volleyball"]:
            if sport not in sport_results:
                continue
            print(f"\n── {sport.title()} ({len(sport_results[sport])} clips) ──\n")
            print_comparison(sport_results[sport], phases)

        print(f"\n── All Sports ({len(results)} clips) ──\n")
        print_comparison(results, phases)


if __name__ == "__main__":
    main()
