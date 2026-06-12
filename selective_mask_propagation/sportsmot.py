"""SportsMOT pipeline: detect -> track -> sam -> merge -> [GTA] -> eval -> render

CLI:
    uv run python -m selective_mask_propagation.sportsmot --input "data/sportsmot/dataset/val/v_00HRwkvvjtQ_c005" --precomputed -d --sam3
    uv run python -m selective_mask_propagation.sportsmot --input "data/sportsmot/dataset/val/v_0kUtTtmLaJA_c006" --step sam -c -d
    uv run python -m selective_mask_propagation.sportsmot --input "data/sportsmot/dataset/val/v_2QhNRucNC7E_c017" --step eval

Steps: detect, track, sam, merge, pose, jersey, classify, embed, match, interp, eval, render
       --gta enables: pose, jersey, classify, embed, match, interp
"""

import argparse
from typing import List, Optional

from .utils.artifacts import get_output_dir, get_artifacts_dir, save_artifact, load_artifact
from .utils.helpers import STEPS, GTA_STEPS, read_sequence_info, clean_steps, step_done, print_step, expand_input


def run_pipeline(
    source_path: str,
    start_step: str = "detect",
    continue_to_end: bool = True,
    precomputed: bool = False,
    dev: bool = False,
    sam3: bool = False,
    gta: bool = False,
    with_reid: bool = True,
    no_gt: bool = False,
    skip_existing: bool = False,
):
    seq_info = read_sequence_info(source_path)
    fps = seq_info["fps"]
    parts = []
    if sam3:
        parts.append("sam3")
    if not with_reid:
        parts.append("noreid")
    suffix = "-" + "-".join(parts) if parts else ""
    output_dir = str(get_output_dir(source_path, suffix))
    artifacts_dir = get_artifacts_dir(source_path, suffix)

    start_idx = STEPS.index(start_step)
    steps = STEPS[start_idx:] if continue_to_end else [start_step]
    if not gta:
        steps = [s for s in steps if s not in GTA_STEPS]
    if skip_existing:
        done = [s for s in steps if step_done(s, source_path, suffix, gta=gta)]
        if done:
            print(f"Skipping (already done): {', '.join(done)}")
        steps = [s for s in steps if s not in done]
        if not steps:
            return
    clean_steps(steps, source_path, suffix)

    detections = embeddings = tracks = margins = sam_masks = windows = rename_events = match_history = None
    merged = renamed_margins = rename_map = None
    frame_errors = {}

    if "detect" in steps:
        print_step("detect")
        from .core.detection import step_detect
        detections, embeddings = step_detect(source_path, precomputed)
        save_artifact("detections", detections, source_path, suffix)
        save_artifact("embeddings", embeddings, source_path, suffix)

    if "track" in steps:
        print_step("track")
        from .deep_eiou.tracker import step_track
        from .utils.export import export_mot
        if detections is None:
            detections = load_artifact("detections", source_path, suffix)
            embeddings = load_artifact("embeddings", source_path, suffix)
        tracks, margins = step_track(detections, embeddings, fps,
                                      seq_info["width"], seq_info["height"],
                                      with_reid=with_reid)
        save_artifact("tracks", tracks, source_path, suffix)
        save_artifact("margins", margins, source_path, suffix)
        export_mot(tracks, str(artifacts_dir / "mot_deep_eiou.txt"))

    if "sam" in steps:
        print_step("sam")
        if sam3:
            from .core.sam3 import build_predictor, step_sam
        else:
            from .core.sam2 import build_predictor, step_sam
        if tracks is None:
            tracks = load_artifact("tracks", source_path, suffix)
        if margins is None:
            margins = load_artifact("margins", source_path, suffix)
        predictor = build_predictor()
        sam_masks, windows, rename_events, match_history = step_sam(
            predictor, tracks, margins, source_path,
        )
        save_artifact("sam_masks", sam_masks, source_path, suffix)
        save_artifact("windows", windows, source_path, suffix)
        save_artifact("rename_events", rename_events, source_path, suffix)
        save_artifact("match_history", match_history, source_path, suffix)

        if dev:
            from .core.debug import save_windows_debug
            save_windows_debug(windows, tracks, margins, sam_masks, source_path, output_dir)

    if "merge" in steps:
        print_step("merge")
        from .core.merge import step_merge, extract_bboxes, TrackData, canon_key
        from .utils.export import export_mot
        if tracks is None:
            tracks = load_artifact("tracks", source_path, suffix)
        if sam_masks is None:
            sam_masks = load_artifact("sam_masks", source_path, suffix)
        if windows is None:
            windows = load_artifact("windows", source_path, suffix)
        if margins is None:
            margins = load_artifact("margins", source_path, suffix)
        if rename_events is None:
            rename_events = load_artifact("rename_events", source_path, suffix)
        if match_history is None:
            match_history = load_artifact("match_history", source_path, suffix, allow_missing=True) or {}
        merged, renamed_margins, rename_map = step_merge(tracks, sam_masks, windows, margins, rename_events)
        save_artifact("renamed_margins", renamed_margins, source_path, suffix)

        sde_bboxes = extract_bboxes(merged, tracks, rename_map, match_history, windows)
        export_mot(sde_bboxes, str(artifacts_dir / "mot_sam_deep_eiou.txt"))

        # Keep merged state strictly aligned with exported MOT.
        # Preserve masks only for tracks that survive into sde_bboxes.
        materialized = {}
        for f, ft in sde_bboxes.items():
            frame_data = {}
            merged_frame = merged.get(f, {})
            for tid, bbox in ft.items():
                entry = merged_frame.get(canon_key(tid))
                if entry is None:
                    entry = merged_frame.get(tid)
                mask = entry.mask if entry is not None else None
                frame_data[tid] = TrackData(bbox=bbox, mask=mask)
            materialized[f] = frame_data
        merged = materialized

        save_artifact("merged", merged, source_path, suffix)

        if dev:
            from .core.debug import save_merge_debug
            save_merge_debug(windows, merged, tracks, renamed_margins,
                             sam_masks, rename_events, source_path, output_dir)

    if "pose" in steps:
        print_step("pose")
        from .gta.models import get_vitpose
        from .gta.pose import estimate_all_poses
        if detections is None:
            detections = load_artifact("detections", source_path, suffix)
        processor, model = get_vitpose()
        pose_data = estimate_all_poses(source_path, detections, processor, model)
        save_artifact("pose_data", pose_data, source_path, suffix)

        if dev:
            from .gta.debug import save_pose_debug
            save_pose_debug(detections, pose_data, source_path, output_dir)

    if "jersey" in steps:
        print_step("jersey")
        from .gta.models import get_parseq
        from .gta.jersey import run_jersey_ocr, aggregate_jersey_numbers
        from .gta.assignments import build_assignments
        from .utils.export import parse_mot
        if detections is None:
            detections = load_artifact("detections", source_path, suffix)
        if tracks is None:
            tracks = load_artifact("tracks", source_path, suffix)
        pose_data = load_artifact("pose_data", source_path, suffix)

        ocr_results, crops = run_jersey_ocr(source_path, detections, pose_data, get_parseq())
        save_artifact("ocr_results", ocr_results, source_path, suffix)

        de_assignments = build_assignments(tracks, detections)
        jersey_map_de = aggregate_jersey_numbers(ocr_results, de_assignments)
        save_artifact("jersey_map_de", jersey_map_de, source_path, suffix)

        sde_tracks = parse_mot(str(artifacts_dir / "mot_sam_deep_eiou.txt"))
        sde_assignments = build_assignments(sde_tracks, detections)
        jersey_map_sde = aggregate_jersey_numbers(ocr_results, sde_assignments)
        save_artifact("jersey_map_sde", jersey_map_sde, source_path, suffix)

        if dev:
            from .gta.debug import save_crop_debug, save_jersey_track_debug
            save_crop_debug(detections, pose_data, source_path, output_dir, ocr_results)
            save_jersey_track_debug(ocr_results, crops, de_assignments, jersey_map_de, output_dir, variant="de")
            save_jersey_track_debug(ocr_results, crops, sde_assignments, jersey_map_sde, output_dir, variant="sde")

    if "classify" in steps:
        print_step("classify")
        from .gta.classify import classify_tracks

        mot_de = str(artifacts_dir / "mot_deep_eiou.txt")
        track_teams_de, debug_info_de = classify_tracks(source_path, mot_de)
        save_artifact("track_teams_de", track_teams_de, source_path, suffix)

        mot_sde = str(artifacts_dir / "mot_sam_deep_eiou.txt")
        track_teams_sde, debug_info_sde = classify_tracks(source_path, mot_sde)
        save_artifact("track_teams_sde", track_teams_sde, source_path, suffix)

        if dev:
            from .gta.debug import save_classify_debug
            for variant, info in [("de", debug_info_de), ("sde", debug_info_sde)]:
                save_classify_debug(
                    info["colors"], info["color_frames"],
                    info["classifications"], info["grids"],
                    output_dir, variant=variant,
                )

    if "embed" in steps:
        print_step("embed")
        from .gta.embed import aggregate_embeddings
        from .gta.assignments import build_assignments
        if embeddings is None:
            embeddings = load_artifact("embeddings", source_path, suffix)
        if detections is None:
            detections = load_artifact("detections", source_path, suffix)
        if tracks is None:
            tracks = load_artifact("tracks", source_path, suffix)

        de_assignments = build_assignments(tracks, detections)
        track_embeddings_de = aggregate_embeddings(embeddings, de_assignments)
        save_artifact("track_embeddings_de", track_embeddings_de, source_path, suffix)

        from .utils.export import parse_mot
        sde_tracks = parse_mot(str(artifacts_dir / "mot_sam_deep_eiou.txt"))
        sde_assignments = build_assignments(sde_tracks, detections)
        track_embeddings_sde = aggregate_embeddings(embeddings, sde_assignments)
        save_artifact("track_embeddings_sde", track_embeddings_sde, source_path, suffix)

        if dev:
            from .gta.debug import save_embed_debug
            save_embed_debug(track_embeddings_de, output_dir, variant="de")
            save_embed_debug(track_embeddings_sde, output_dir, variant="sde")

    if "match" in steps:
        print_step("match")
        from .gta.match import match_tracklets, save_mot_remapped

        mot_de = str(artifacts_dir / "mot_deep_eiou.txt")
        track_embeddings_de = load_artifact("track_embeddings_de", source_path, suffix)
        jersey_map_de = load_artifact("jersey_map_de", source_path, suffix)
        track_teams_de = load_artifact("track_teams_de", source_path, suffix)
        canonical_to_gta_de, merge_events_de = match_tracklets(
            mot_de, track_embeddings_de, jersey_map_de, track_teams_de,
            seq_info["width"], seq_info["height"],
        )
        save_artifact("canonical_to_gta_de", canonical_to_gta_de, source_path, suffix)
        save_mot_remapped(str(artifacts_dir / "mot_de_gta.txt"), canonical_to_gta_de, mot_de)

        mot_sde = str(artifacts_dir / "mot_sam_deep_eiou.txt")
        track_embeddings_sde = load_artifact("track_embeddings_sde", source_path, suffix)
        jersey_map_sde = load_artifact("jersey_map_sde", source_path, suffix)
        track_teams_sde = load_artifact("track_teams_sde", source_path, suffix)
        canonical_to_gta_sde, merge_events_sde = match_tracklets(
            mot_sde, track_embeddings_sde, jersey_map_sde, track_teams_sde,
            seq_info["width"], seq_info["height"],
        )
        save_artifact("canonical_to_gta_sde", canonical_to_gta_sde, source_path, suffix)
        save_mot_remapped(str(artifacts_dir / "mot_sde_gta.txt"), canonical_to_gta_sde, mot_sde)

        if dev:
            from .gta.debug import save_gta_debug
            save_gta_debug(merge_events_de, canonical_to_gta_de, mot_de, source_path, output_dir, variant="de")
            save_gta_debug(merge_events_sde, canonical_to_gta_sde, mot_sde, source_path, output_dir, variant="sde")

    if "interp" in steps:
        print_step("interp")
        from .core.interp import interpolate_tracks
        from .utils.export import export_mot, parse_mot

        track_teams_de = load_artifact("track_teams_de", source_path, suffix)
        other_ids_de = {tid for tid, t in track_teams_de.items() if t["team_id"] is None}
        de_gta_tracks = parse_mot(str(artifacts_dir / "mot_de_gta.txt"))
        de_gta_interp = interpolate_tracks(de_gta_tracks, skip_ids=other_ids_de)
        de_gta_combined = {f: dict(ft) for f, ft in de_gta_tracks.items()}
        for f, ft in de_gta_interp.items():
            de_gta_combined.setdefault(f, {}).update(ft)
        export_mot(de_gta_combined, str(artifacts_dir / "mot_de_gta_interp.txt"))

        track_teams_sde = load_artifact("track_teams_sde", source_path, suffix)
        other_ids_sde = {tid for tid, t in track_teams_sde.items() if t["team_id"] is None}
        sde_gta_tracks = parse_mot(str(artifacts_dir / "mot_sde_gta.txt"))
        sde_gta_interp = interpolate_tracks(sde_gta_tracks, skip_ids=other_ids_sde)
        sde_gta_combined = {f: dict(ft) for f, ft in sde_gta_tracks.items()}
        for f, ft in sde_gta_interp.items():
            sde_gta_combined.setdefault(f, {}).update(ft)
        export_mot(sde_gta_combined, str(artifacts_dir / "mot_sde_gta_interp.txt"))

    if "eval" in steps:
        print_step("eval")
        from .core.eval import step_eval
        frame_errors = step_eval(source_path, output_dir, suffix, no_gta=not gta)
        for key, errors in frame_errors.items():
            save_artifact(f"{key}_frame_errors", errors, source_path, suffix)

    if "render" in steps:
        print_step("render")
        from .core.render import step_render
        if tracks is None:
            tracks = load_artifact("tracks", source_path, suffix)
        if margins is None:
            margins = load_artifact("margins", source_path, suffix)
        if merged is None:
            merged = load_artifact("merged", source_path, suffix)
        if renamed_margins is None:
            renamed_margins = load_artifact("renamed_margins", source_path, suffix)
        def _load_errors(key):
            if key not in frame_errors:
                frame_errors[key] = load_artifact(f"{key}_frame_errors", source_path, suffix, allow_missing=True)
            return frame_errors.get(key)
        de_fe = _load_errors("deep_eiou")
        sde_fe = _load_errors("sam_deep_eiou")

        de_gta_tracks = sde_gta_tracks = None
        de_gta_fe = sde_gta_fe = None
        if gta:
            from .utils.export import parse_mot
            de_gta_mot = artifacts_dir / "mot_de_gta_interp.txt"
            if de_gta_mot.exists():
                de_gta_tracks = parse_mot(str(de_gta_mot))
                de_gta_fe = _load_errors("de_gta")
            sde_gta_mot = artifacts_dir / "mot_sde_gta_interp.txt"
            if sde_gta_mot.exists():
                sde_gta_tracks = parse_mot(str(sde_gta_mot))
                sde_gta_fe = _load_errors("sde_gta")

        step_render(source_path, tracks, merged, output_dir,
                    margins=margins, renamed_margins=renamed_margins,
                    de_frame_errors=de_fe, sde_frame_errors=sde_fe,
                    de_gta_tracks=de_gta_tracks, de_gta_frame_errors=de_gta_fe,
                    sde_gta_tracks=sde_gta_tracks, sde_gta_frame_errors=sde_gta_fe,
                    no_gt=no_gt)

    print("\nDone.")


def main(args_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="SAM-Deep-EIoU pipeline")
    parser.add_argument("--input", required=True, nargs="+", help="Sequence directories or glob pattern.")
    parser.add_argument("--step", choices=STEPS, help="Run a specific step.")
    parser.add_argument("-c", "--continue", dest="continue_pipeline",
                        action="store_true", help="Continue from --step to end.")
    parser.add_argument("--precomputed", action="store_true",
                        help="Use precomputed det.txt + emb.npy (SportsMOT only).")
    parser.add_argument("-d", "--dev", action="store_true",
                        help="Save debug outputs.")
    parser.add_argument("--sam3", action="store_true",
                        help="Use SAM3 instead of SAM2. Outputs to separate results dir.")
    parser.add_argument("--gta", action="store_true",
                        help="Run GTA steps (pose, jersey, classify, embed, match, interp). Requires GEMINI_API_KEY.")
    parser.add_argument("--no-reid", dest="no_reid", action="store_true",
                        help="Disable ReID (OSNet). Use pure EIoU association.")
    parser.add_argument("--no-gt", dest="no_gt", action="store_true",
                        help="Skip GT overlay in rendered videos.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip steps whose artifacts already exist, and keep going "
                             "past per-clip failures. Makes long multi-clip runs resumable: "
                             "re-run the same command after a crash to pick up where it stopped.")
    args = parser.parse_args(args_list)

    source_paths = []
    for inp in args.input:
        source_paths.extend(expand_input(inp))

    failed = []
    for source_path in source_paths:
        try:
            if args.step:
                run_pipeline(source_path, args.step, args.continue_pipeline,
                             precomputed=args.precomputed, dev=args.dev, sam3=args.sam3,
                             gta=args.gta, with_reid=not args.no_reid, no_gt=args.no_gt,
                             skip_existing=args.skip_existing)
            else:
                run_pipeline(source_path, "detect", continue_to_end=True,
                             precomputed=args.precomputed, dev=args.dev, sam3=args.sam3,
                             gta=args.gta, with_reid=not args.no_reid, no_gt=args.no_gt,
                             skip_existing=args.skip_existing)
        except Exception:
            if not args.skip_existing:
                raise
            import traceback
            traceback.print_exc()
            failed.append(source_path)

    if failed:
        print(f"\n{len(failed)} clip(s) failed (re-run with --skip-existing to retry only these):")
        for p in failed:
            print(f"  {p}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
