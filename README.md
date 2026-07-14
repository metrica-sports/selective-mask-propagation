# Selective Mask Propagation

Selective mask propagation (SMP) for multi-object tracking: a lightweight base tracker runs every frame, and SAM is dispatched only where the tracker is at risk of an identity switch. Training-free and tracker-agnostic; the flagship instantiation, **SAM3-Deep-EIoU**, combines Deep-EIoU with SAM 3.

**87.2 HOTA on SportsMOT**, [#1 on the official leaderboard](https://www.codabench.org/competitions/13077/#/results-tab) (June 2026). Consistent improvements across three base trackers on DanceTrack.

[[Paper]](https://arxiv.org/abs/2606.13033)

![SportsMOT test leaderboard: SAM3-Deep-EIoU (holma91) ranked #1 at 87.16 HOTA](https://github.com/user-attachments/assets/34037fc5-9fe9-4cb9-a8bc-472efdd12635)

The same occlusion through plain Deep-EIoU (identities swap in the pile-up) and through SAM3-Deep-EIoU (masks dispatched, swaps corrected):

https://github.com/user-attachments/assets/1655577c-ede3-4fb0-bc71-860e771df183

https://github.com/user-attachments/assets/e1e6c7ce-27f5-4fec-89cd-0581b4c199bf

## How it works

A lightweight tracker (Deep-EIoU) runs on every frame. The assignment margin — the gap between the best and second-best match in its Hungarian cost matrix — tells us, for free, when the tracker is about to guess between near-tied identities. When the margin collapses, a **window** opens: SAM is seeded from a recent frame where the identity was unambiguous and propagates masks through the ambiguity. At exit, the output is modified only if the mask confidently settled into a *different* track than it was seeded on (a SWAP); every other outcome — including the mask degrading or drifting — leaves the base tracker's output untouched. So the only way SMP can do worse than its base tracker is a false-positive swap, and the windowing is built to make that rare.

For sports, an optional [global track association](#global-track-association-gta) stage re-links players who leave and re-enter the camera view. Full details in the [paper](https://arxiv.org/abs/2606.13033).

## Quickstart

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), CUDA GPU.

```bash
git clone https://github.com/holma91/selective-mask-propagation.git
cd selective-mask-propagation
uv sync
uv run python scripts/demo.py
```

No dataset needed: the demo runs SMP on a bundled SportsMOT basketball clip, prints before/after tracking metrics against its bundled ground truth (+9.4 HOTA over Deep-EIoU), and writes the mask-overlay video to `results/demo/`. First run downloads the SAM 3 checkpoint. `--clip football|volleyball` for the other bundled clips, `--no-render` to skip the video.

## Use it on your own tracker

The algorithm is exposed as a single call and treats the tracker as a black box. If your tracker solves a Hungarian assignment, extract the margin (`second_best - best` per matched column) and pass it in:

```python
from selective_mask_propagation.augment import augment

# tracks:  {frame: {track_id: [x1, y1, x2, y2]}}
# margins: {frame: {track_id: float}}
tracks, margins = my_tracker.run(detections)

corrected_tracks = augment(tracks, margins, "path/to/sequence", sam3=True)
```

`augment()` takes `mode="benchmark"` (default) or `mode="prod"`. Benchmark keeps your tracker's box geometry and uses SAM only to correct identities — what benchmark evaluators reward, and what all paper numbers use. Prod makes the mask authoritative while a window is open, carrying the correct player *through* occlusions — better for real applications (pose, event attribution), but it scores lower against box annotations, because tight mask-derived boxes get IoU-punished by full-extent ground truth. The pipelines below take the same `--mode` flag; the demo prints benchmark metrics and renders the prod view.

## Results

### SportsMOT test set

Scored by the official remote evaluator; listed as `holma91` on the [leaderboard](https://www.codabench.org/competitions/13077/#/results-tab).

| Method | HOTA | AssA | IDF1 | MOTA |
|--------|------|------|------|------|
| Deep-EIoU | 77.2 | 67.7 | 79.8 | 96.3 |
| SAM2-Deep-EIoU (with GTA) | 85.5 | 81.7 | 91.2 | 97.3 |
| SAM3-Deep-EIoU (with GTA) | **87.2** | **84.2** | **93.6** | **98.1** |

The SAM2 → SAM3 gap comes from swapping the VOS component while the dispatch signal, window logic, and merge stayed frozen: a better mask propagator improves the system for free.

### DanceTrack validation set

The identical recipe applied to three base trackers — same YOLOX detections, same thresholds, no per-tracker tuning, no GTA:

| Base tracker | Baseline | + SAM 2 | + SAM 3 |
|--------------|----------|---------|---------|
| SORT | 39.8 | 45.1 | **46.1** |
| ByteTrack | 54.6 | 60.3 | **61.2** |
| Deep-EIoU | 51.7 | 57.8 | **59.7** |

All numbers HOTA. The gains concentrate in AssA (+7 to +11), the association component — exactly where a method that only fixes identity assignment should show up.

## Efficiency

Throughput of the SAM step, the cost added on top of the base tracker:

> **fps = video frames / wall-clock(SAM + merge)**

On the full SportsMOT test set (150 clips, 94.8k frames), RTX PRO 6000:

| Sport | fps |
|---|---|
| Basketball | 11 |
| Football | 21 |
| Volleyball | 11 |
| **Overall** | **13** |

Peak VRAM: 5.4 GB median across clips, 6.1 GB worst case. Each SAM pass tracks only the few ambiguous objects, not every player. Measure it on your GPU with `scripts/show_fps.py` (no dataset needed).

### Selective vs uniform dispatch

The same SAM 3, two dispatch policies: selective windows versus a mask for every player on every frame, on identical YOLOX detections and Deep-EIoU tracks. Four bundled broadcast clips (`bench/`: two NBA, two international football), RTX 5090:

| clip | frames | tracks | selective | GB | uniform | GB |
|---|---|---|---|---|---|---|
| basketball-1 | 600 | 12 | 8.7 fps | 5.7 | 6.0 fps | 11.3 |
| basketball-2 | 360 | 10 | 9.5 fps | 5.5 | 6.1 fps | 8.5 |
| soccer-1 | 600 | 30 | 37.7 fps | 5.0 | 5.0 fps | 10.1 |
| soccer-2 | 740 | 42 | 32.4 fps | 5.0 | 5.4 fps | 9.5 |
| **aggregate** | **2300** | | **15.8 fps** | **5.7** | **5.5 fps** | **11.3** |

Reproduce (YOLOX + OSNet checkpoints needed once; `--render` writes the overlay videos to `results/bench/`):

```bash
uv run python scripts/setup/download_checkpoints.py
uv run python scripts/fps_benchmark.py
```

## Reproduce the paper

### Data

Download [SportsMOT](https://github.com/MCG-NJU/SportsMOT) and/or [DanceTrack](https://github.com/DanceTrack/DanceTrack) from their official repos and extract (or symlink) into `data/`:

```
data/sportsmot/dataset/{train,val,test}/<sequence>/{img1,gt,seqinfo.ini}
data/dancetrack/{val,test}/<sequence>/{img1,gt,seqinfo.ini}
```

Then download the precomputed YOLOX detections and OSNet embeddings:

```bash
uv run python scripts/setup/download_precomputed.py sportsmot --split val       # 1.4 GB
uv run python scripts/setup/download_precomputed.py sportsmot                   # all splits, 7.5 GB
uv run python scripts/setup/download_precomputed.py dancetrack                  # val, 0.6 GB
```

For SAM2 runs only, download the SAM2 checkpoint (SAM 3 downloads automatically):

```bash
cd vendor/sam2/checkpoints && bash download_ckpts.sh && cd ../../..
```

### Run

Single sequence through the full pipeline (detect → track → sam → merge → eval → render):

```bash
uv run python -m selective_mask_propagation.sportsmot \
  --input data/sportsmot/dataset/val/v_00HRwkvvjtQ_c005 \
  --precomputed --sam3

uv run python -m selective_mask_propagation.dancetrack \
  --input data/dancetrack/val/dancetrack0007 --precomputed --sam3
```

Full benchmark runs:

```bash
# SportsMOT (paper Table 3; --gta variant needs the GTA extras below)
bash scripts/reproduce/run_sportsmot.sh val sam3
bash scripts/reproduce/run_sportsmot.sh test sam3
uv run python scripts/reproduce/build_submission.py --sam3

# DanceTrack (paper Table 1)
bash scripts/reproduce/run_dancetrack.sh val sam3
bash scripts/reproduce/run_dancetrack.sh val sam3 --tracker bytetrack
bash scripts/reproduce/run_dancetrack.sh val sam3 --tracker sort
```

## Global Track Association (GTA)

GTA links tracklets across frame-boundary exits using jersey recognition, team classification, and appearance embeddings. It operates on finished tracklets and does not modify the tracking or SAM steps.

It needs extra dependencies and a Gemini API key for team classification:

```bash
uv sync --extra gta
cp .env.example .env    # then add your GEMINI_API_KEY
```

Enable with `--gta`:

```bash
uv run python -m selective_mask_propagation.sportsmot \
  --input data/sportsmot/dataset/val/v_00HRwkvvjtQ_c001 \
  --precomputed --gta --sam3
```

- **Pose estimation** uses ViTPose+ (base) via a standalone implementation (`selective_mask_propagation/vitpose/`); only torso keypoints (shoulders + hips) are used.
- **Jersey OCR** uses PARSeq via a standalone implementation (`selective_mask_propagation/parseq/`), no torch.hub.
- **Team classification** uses Gemini Flash to label each player crop (team A, B, or other). It is the only proprietary dependency, swappable for an open VLM.

## Postscript: SAM 3.1

[SAM 3.1 (Object Multiplex)](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) was released after this project was completed. It tracks objects jointly in shared-memory buckets instead of SAM 3's per-object passes — a direct attack on the cost axis measured above. The same benchmark for its native dense-tracking pipeline (`scripts/fps_sam31_native.py`: text prompt `"player"`, its own detector), same clips, same RTX 5090: **7.3 fps aggregate at 27.2 GB peak**. Substantially faster than uniform SAM 3, but selective dispatch is still ~2× faster at ~5× less memory — it wins by not tracking everything, not by tracking everything faster, so the two approaches compose rather than compete.

```bash
uv run python scripts/fps_sam31_native.py     # --render for videos
```

SAM 3.1 lives in `vendor/sam3-with-3.1`, separate from the frozen `vendor/sam3` snapshot that all paper results run on, keeping every reported number reproducible.

## Acknowledgments

This repository builds on and vendors code from the following projects:

- [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU): base tracker (`selective_mask_propagation/deep_eiou/`)
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) (MIT) and [SORT](https://github.com/abewley/sort) (GPL-3.0): alternative base trackers for the cross-tracker comparison
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0): detector (`selective_mask_propagation/yolox/`)
- [OSNet / deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) (MIT): appearance embeddings (`selective_mask_propagation/osnet/`)
- [SAM 2](https://github.com/facebookresearch/sam2) (Apache-2.0) and [SAM 3](https://github.com/facebookresearch/sam3) (SAM License): VOS models, vendored in `vendor/` with their original licenses
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) (MIT): evaluation (`TrackEval/`)
- [GTA](https://github.com/sjc042/gta-link): global tracklet association formulation
- Jersey OCR follows [Koshkina & Elder](https://github.com/mkoshkina/jersey-number-pipeline), using [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and [PARSeq](https://github.com/baudm/parseq) via standalone inference-only implementations (`selective_mask_propagation/{vitpose,parseq}/`)

Code written for this project is MIT-licensed (see `LICENSE`). Vendored components obviously retain their original licenses.

## Citation

```bibtex
@article{holmberg2026selectivemaskpropagation,
  title={Selective Mask Propagation for Multi-Object Tracking},
  author={Holmberg, Alexander},
  journal={arXiv preprint arXiv:2606.13033},
  year={2026}
}
```
