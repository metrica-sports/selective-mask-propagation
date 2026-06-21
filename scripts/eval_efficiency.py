"""Per-sport efficiency + dispatch breakdown over a SportsMOT run.

Computes source metadata, processing throughput (overall / per-step / per-sport),
SAM fps under three denominators, peak VRAM, and the selective-dispatch outcome
distribution, from the artifacts a completed run leaves on disk. Used to produce
the efficiency numbers reported for SAM-Deep-EIoU on SportsMOT.

Reads, for each test clip:
  - data/sportsmot/dataset/test/<clip>/seqinfo.ini    (frameRate, seqLength, res)
  - results/test/<clip>-sam3/artifacts/timing.json     (per-step wall-clock, VRAM)
  - results/test/<clip>-sam3/artifacts/windows.pkl      (dispatch outcomes)
  - data/sportsmot/splits_txt/{basketball,football,volleyball}.txt  (sport label)

Usage (from repo root, after a `run_sportsmot.sh test sam3` run):
    uv run python scripts/eval_efficiency.py

Notes:
  - Per-step timing for the FIRST clip of each step's batch includes one-time
    model-load warmup; medians (used here) suppress it.
  - The `classify` step is Gemini-API latency, not local compute.
  - SAM-active% is the fraction of frames where >=1 window was live; it is NOT the
    same as the window-outcome "base-unchanged %".
"""

import configparser
import json
import pickle
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make selective_mask_propagation importable for unpickling windows
DATA = ROOT / "data" / "sportsmot" / "dataset" / "test"
RES = ROOT / "results" / "test"
SPLITS = ROOT / "data" / "sportsmot" / "splits_txt"

GPU_STEPS = ["detect", "track", "sam", "merge", "pose", "embed"]  # local compute
ALL_STEPS = ["detect", "track", "sam", "merge", "pose", "jersey",
             "classify", "embed", "match", "interp"]


def load_sport_map():
    m = {}
    for sport in ("basketball", "football", "volleyball"):
        for line in (SPLITS / f"{sport}.txt").read_text().split():
            m[line.strip()] = sport
    return m


def load_seqinfo(clip):
    cp = configparser.ConfigParser()
    cp.read(DATA / clip / "seqinfo.ini")
    s = cp["Sequence"]
    return {
        "fps": int(s["frameRate"]),
        "frames": int(s["seqLength"]),
        "w": int(s["imWidth"]),
        "h": int(s["imHeight"]),
    }


def main():
    sport_map = load_sport_map()
    clip_dirs = sorted(p for p in RES.iterdir() if p.is_dir() and p.name.endswith("-sam3"))

    rows = []
    for d in clip_dirs:
        clip = d.name.removesuffix("-sam3")
        art = d / "artifacts"
        timing = json.loads((art / "timing.json").read_text())
        info = load_seqinfo(clip)
        with open(art / "windows.pkl", "rb") as f:
            windows = pickle.load(f)
        rows.append({
            "clip": clip,
            "sport": sport_map.get(clip, "unknown"),
            **info,
            "timing": timing,
            "windows": windows,
        })

    def agg(rs, key):
        return [r[key] for r in rs]

    def fmt_stats(vals):
        return f"median {st.median(vals):.1f}  mean {st.mean(vals):.1f}  min {min(vals):.1f}  max {max(vals):.1f}"

    # ---- Source metadata ----
    print("=" * 70)
    print(f"SOURCE METADATA  ({len(rows)} test clips)")
    print("=" * 70)
    fps_dist = Counter(r["fps"] for r in rows)
    res_dist = Counter((r["w"], r["h"]) for r in rows)
    total_frames = sum(r["frames"] for r in rows)
    print(f"  source frameRate (fps): {dict(fps_dist)}")
    print(f"  resolution: {{{', '.join(f'{w}x{h}: {n}' for (w,h),n in res_dist.items())}}}")
    print(f"  total frames: {total_frames:,}   clip length frames: {fmt_stats(agg(rows,'frames'))}")

    by_sport = defaultdict(list)
    for r in rows:
        by_sport[r["sport"]].append(r)
    print("\n  per sport:")
    print(f"    {'sport':<12}{'clips':>6}{'frames':>10}{'src_fps':>9}{'med_len':>9}")
    for sport in ("basketball", "football", "volleyball"):
        rs = by_sport[sport]
        tf = sum(r["frames"] for r in rs)
        fps_set = sorted({r["fps"] for r in rs})
        print(f"    {sport:<12}{len(rs):>6}{tf:>10,}{str(fps_set):>9}{st.median(agg(rs,'frames')):>9.0f}")

    # ---- Processing throughput ----
    print("\n" + "=" * 70)
    print("PROCESSING THROUGHPUT  (frames / wall-clock-sec)")
    print("=" * 70)

    def step_total(r, steps):
        return sum(r["timing"].get(s, 0.0) for s in steps)

    for r in rows:
        r["t_gpu"] = step_total(r, GPU_STEPS)
        r["t_all"] = step_total(r, ALL_STEPS)
        r["t_sam"] = r["timing"].get("sam", 0.0)
        r["fps_gpu"] = r["frames"] / r["t_gpu"] if r["t_gpu"] else 0
        r["fps_all"] = r["frames"] / r["t_all"] if r["t_all"] else 0
        r["sam_s_per_frame"] = r["t_sam"] / r["frames"] if r["frames"] else 0

    print("  Aggregate (sum frames / sum time) — the honest end-to-end number:")
    sum_f = sum(r["frames"] for r in rows)
    sum_gpu = sum(r["t_gpu"] for r in rows)
    sum_all = sum(r["t_all"] for r in rows)
    print(f"    local-compute steps {GPU_STEPS}: {sum_f/sum_gpu:.1f} fps  ({sum_gpu/60:.1f} min total)")
    print(f"    full pipeline incl. Gemini classify: {sum_f/sum_all:.1f} fps  ({sum_all/60:.1f} min total)")
    print(f"    classify (Gemini API) share of wall-clock: "
          f"{sum(r['timing'].get('classify',0) for r in rows)/sum_all*100:.0f}%")

    print("\n  Per-step median seconds/clip + s/frame (median over clips):")
    print(f"    {'step':<10}{'med_s/clip':>12}{'med_s/frame':>13}")
    for s in ALL_STEPS:
        per_clip = [r["timing"].get(s, 0.0) for r in rows]
        per_frame = [r["timing"].get(s, 0.0) / r["frames"] for r in rows]
        print(f"    {s:<10}{st.median(per_clip):>12.2f}{st.median(per_frame):>13.4f}")

    # ---- SAM cost per sport (paper-relevant: analog of Tab.2 SAM s/frame) ----
    print("\n" + "=" * 70)
    print("SAM s/frame & VRAM per sport")
    print("=" * 70)
    print(f"  {'sport':<12}{'sam_s/frame':>13}{'fps_gpu':>9}{'peak_vram_mb':>14}")
    for sport in ("basketball", "football", "volleyball"):
        rs = by_sport[sport]
        spf = st.median(agg(rs, "sam_s_per_frame"))
        fgpu = st.median(agg(rs, "fps_gpu"))
        vram = [r["timing"].get("sam_peak_vram_mb", 0) for r in rs]
        print(f"  {sport:<12}{spf:>13.4f}{fgpu:>9.1f}{f'{st.median(vram):.0f} (max {max(vram)})':>14}")
    all_vram = [r["timing"].get("sam_peak_vram_mb", 0) for r in rows]
    print(f"  {'ALL':<12}{st.median(agg(rows,'sam_s_per_frame')):>13.4f}"
          f"{st.median(agg(rows,'fps_gpu')):>9.1f}{f'{st.median(all_vram):.0f} (max {max(all_vram)})':>14}")

    # ---- SAM throughput, honest denominators ----
    # amortized: sam_time / all clip frames (what the pipeline pays per video frame)
    # active:    sam_time / frames where >=1 window was live (SAM's real per-pass rate)
    # object:    sam_time / sum(window live-lengths) (per propagated object-frame)
    def sam_detail(rs):
        sam_t = sum(r["t_sam"] for r in rs)
        all_f = sum(r["frames"] for r in rs)
        active_f = 0
        obj_f = 0
        for r in rs:
            iv = sorted((w.seed_frame, w.exit_frame) for w in r["windows"])
            obj_f += sum(e - s + 1 for s, e in iv)
            # union length of [s,e] intervals = frames SAM ran at least one pass
            cur_s = cur_e = None
            for s, e in iv:
                if cur_e is None or s > cur_e + 1:
                    if cur_e is not None:
                        active_f += cur_e - cur_s + 1
                    cur_s, cur_e = s, e
                else:
                    cur_e = max(cur_e, e)
            if cur_e is not None:
                active_f += cur_e - cur_s + 1
        return sam_t, all_f, active_f, obj_f

    print("\n" + "=" * 70)
    print("SAM THROUGHPUT (aggregate sum/sum, warmup-diluted over 150 clips)")
    print("=" * 70)
    print(f"  {'scope':<12}{'amort_fps':>11}{'active_fps':>12}{'obj_fps':>10}{'SAM-active%':>12}")
    for sport in ("basketball", "football", "volleyball", "ALL"):
        rs = rows if sport == "ALL" else by_sport[sport]
        sam_t, all_f, active_f, obj_f = sam_detail(rs)
        print(f"  {sport:<12}{all_f/sam_t:>11.1f}{active_f/sam_t:>12.1f}"
              f"{obj_f/sam_t:>10.1f}{active_f/all_f*100:>11.1f}%")
    print("  amort = sam_time/all-frames (paper Tab.2 basis; DanceTrack=5.8)")
    print("  active = sam_time/frames-SAM-ran; obj = sam_time/object-frames")

    # ---- Selective dispatch outcomes per sport ----
    print("\n" + "=" * 70)
    print("SELECTIVE DISPATCH  (windows.pkl) per sport")
    print("=" * 70)

    def window_stats(rs):
        triggers = Counter()
        outcomes = Counter()
        nwin = 0
        for r in rs:
            for w in r["windows"]:
                nwin += 1
                triggers[w.trigger] += 1
                outcomes[w.outcome.value] += 1
        return nwin, triggers, outcomes

    for sport in ("basketball", "football", "volleyball", "ALL"):
        rs = rows if sport == "ALL" else by_sport[sport]
        nwin, triggers, outcomes = window_stats(rs)
        nclip = len(rs)
        tf = sum(r["frames"] for r in rs)
        print(f"\n  {sport}  ({nclip} clips, {tf:,} frames)")
        print(f"    windows: {nwin}  ({nwin/nclip:.1f}/clip, {nwin/tf*1000:.2f}/1k-frames)")
        print(f"    triggers: {dict(triggers)}")
        print(f"    outcomes: {dict(outcomes)}")
        swaps = outcomes.get("swap", 0)
        unchanged = sum(outcomes.get(o, 0) for o in ("clean", "stale", "edge", "end"))
        print(f"    SWAP (relabels): {swaps} ({swaps/max(nwin,1)*100:.1f}%)   "
              f"base-unchanged: {unchanged} ({unchanged/max(nwin,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
