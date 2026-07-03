# Selective Mask Propagation

Selective mask propagation (SMP) for multi-object tracking. Monitors the assignment margin in the Hungarian cost matrix and selectively invokes SAM to preserve identity through occlusions. Only modifies the base tracker's output when positive evidence of an identity switch is found. The flagship instantiation, **SAM3-Deep-EIoU**, combines Deep-EIoU with SAM 3.

**87.2 HOTA on SportsMOT**, [#1 on the official leaderboard](https://www.codabench.org/competitions/13077/#/results-tab) (June 2026). Consistent improvements across three base trackers on DanceTrack.

[[Paper]](https://arxiv.org/abs/2606.13033)

![SportsMOT test leaderboard: SAM3-Deep-EIoU (holma91) ranked #1 at 87.16 HOTA](https://github.com/user-attachments/assets/34037fc5-9fe9-4cb9-a8bc-472efdd12635)

https://github.com/user-attachments/assets/1655577c-ede3-4fb0-bc71-860e771df183

https://github.com/user-attachments/assets/e1e6c7ce-27f5-4fec-89cd-0581b4c199bf

## Demo

No dataset needed. After `uv sync` (see [Setup](#setup)), run the SAM step on three bundled clips (one per sport) and print throughput (first run downloads the SAM 3 checkpoint):

```bash
uv run python scripts/show_fps.py
```

## Results

### SportsMOT Test

Scored by the official remote evaluator. See the [SportsMOT leaderboard](https://www.codabench.org/competitions/13077/#/results-tab) (listed as `holma91`).

| Method | HOTA | AssA | IDF1 | MOTA |
|--------|------|------|------|------|
| Deep-EIoU | 77.2 | 67.7 | 79.8 | 96.3 |
| SAM3-Deep-EIoU (with GTA) | **87.2** | **84.2** | **93.6** | **98.1** |

## Efficiency

Throughput of the SAM step, the cost added on top of the base tracker:

> **fps = video frames / wall-clock(SAM + merge)**

On SportsMOT test (150 clips, 94.8k frames), RTX PRO 6000:

| Sport | fps |
|---|---|
| Basketball | 11 |
| Football | 21 |
| Volleyball | 11 |
| **Overall** | **13** |

Peak VRAM 5.4 GB (max 6.1). Each SAM pass tracks only the few ambiguous objects, not every player. Measure it on your GPU with [`scripts/show_fps.py`](#demo).

### Selective vs uniform

`scripts/fps_benchmark.py` compares selective mask propagation against uniform SAM3 (a mask for every track on every frame) on four bundled broadcast clips (`bench/`: two NBA, two international football, 360–740 frames each). Both sides share the same YOLOX detections and Deep-EIoU tracks; only the dispatch differs. On an RTX 5090:

| clip | frames | tracks | selective | GB | uniform | GB |
|---|---|---|---|---|---|---|
| basketball-1 | 600 | 12 | 8.7 fps | 5.7 | 6.0 fps | 11.3 |
| basketball-2 | 360 | 10 | 9.5 fps | 5.5 | 6.1 fps | 8.5 |
| soccer-1 | 600 | 30 | 37.7 fps | 5.0 | 5.0 fps | 10.1 |
| soccer-2 | 740 | 42 | 32.4 fps | 5.0 | 5.4 fps | 9.5 |
| **aggregate** | **2300** | | **15.8 fps** | **5.7** | **5.5 fps** | **11.3** |

Reproduce (YOLOX + OSNet checkpoints needed once, then one command; `--render` writes the overlay videos to `results/bench/`):

```bash
uv run python scripts/setup/download_checkpoints.py
uv run python scripts/fps_benchmark.py
```

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), CUDA GPU.

```bash
git clone https://github.com/holma91/selective-mask-propagation.git
cd selective-mask-propagation
uv sync
```

### SAM Checkpoints

SAM2 (download into `vendor/sam2/checkpoints/`):
```bash
cd vendor/sam2/checkpoints && bash download_ckpts.sh && cd ../../..
```

SAM3 checkpoints are downloaded automatically on first use.

### GTA extras (only for `--gta`)

The global track association module needs extra dependencies and a Gemini API key for team classification:

```bash
uv sync --extra gta
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Data

### SportsMOT

1. Download SportsMOT from the [official repo](https://github.com/MCG-NJU/SportsMOT) and extract into `data/sportsmot/`:

```
data/sportsmot/dataset/
  val/
    v_00HRwkvvjtQ_c001/
      img1/
      gt/
      seqinfo.ini
    ...
  train/
  test/
```

If you already have SportsMOT downloaded elsewhere, symlink it instead:

```bash
mkdir -p data && ln -s /path/to/sportsmot data/sportsmot
```

2. Download precomputed YOLOX detections and OSNet embeddings:

```bash
uv run python scripts/setup/download_precomputed.py sportsmot --split val       # 1.4 GB
uv run python scripts/setup/download_precomputed.py sportsmot                   # all splits, 7.5 GB
```

### DanceTrack

1. Download DanceTrack from the [official repo](https://github.com/DanceTrack/DanceTrack) and extract into `data/dancetrack/`:

```
data/dancetrack/
  val/
    dancetrack0007/
      img1/
      gt/
      seqinfo.ini
    ...
  test/
```

2. Download precomputed detections and embeddings:

```bash
uv run python scripts/setup/download_precomputed.py dancetrack                  # val, 0.6 GB
```

## Quickstart

Run on a single SportsMOT val sequence (basketball, +9.4 HOTA over baseline):

```bash
uv run python -m selective_mask_propagation.sportsmot \
  --input data/sportsmot/dataset/val/v_00HRwkvvjtQ_c005 \
  --precomputed --sam3
```

Run on a single DanceTrack val sequence (+26.7 HOTA over baseline):

```bash
uv run python -m selective_mask_propagation.dancetrack \
  --input data/dancetrack/val/dancetrack0007 --precomputed --sam3
```

### Test the augment API

Runs Deep-EIoU, augments with SAM, and prints before/after metrics:

```bash
uv run python -m selective_mask_propagation.augment data/sportsmot/dataset/val/v_00HRwkvvjtQ_c005 --sam3
```

## API

To augment your own tracker, extract the assignment margin from the cost matrix (`second_best - best` per matched column) and call `augment`:

```python
from selective_mask_propagation.augment import augment

# tracks: {frame: {track_id: [x1, y1, x2, y2]}}
# margins: {frame: {track_id: float}}
tracks, margins = my_tracker.run(detections)

corrected_tracks = augment(tracks, margins, "path/to/sequence", sam3=True)
```

## Reproduce Paper Results

### SportsMOT (Table 3)

```bash
bash scripts/reproduce/run_sportsmot.sh val sam3
bash scripts/reproduce/run_sportsmot.sh test sam3
uv run python scripts/reproduce/build_submission.py --sam3
```

### DanceTrack (Table 1)

```bash
bash scripts/reproduce/run_dancetrack.sh val sam3
bash scripts/reproduce/run_dancetrack.sh val sam3 --tracker bytetrack
bash scripts/reproduce/run_dancetrack.sh val sam3 --tracker sort
```

## Global Track Association (GTA)

GTA links tracklets across frame-boundary exits using jersey recognition, team classification, and appearance embeddings. It operates on finished tracklets and does not modify the tracking or SAM steps. Enable with `--gta` (requires `GEMINI_API_KEY` in `.env`).

```bash
uv run python -m selective_mask_propagation.sportsmot \
  --input data/sportsmot/dataset/val/v_00HRwkvvjtQ_c001 \
  --precomputed --gta --sam3
```

- **Pose estimation** uses ViTPose+ (base) via a standalone implementation (`selective_mask_propagation/vitpose/`); only torso keypoints (shoulders + hips) are used.
- **Jersey OCR** uses PARSeq via a standalone implementation (`selective_mask_propagation/parseq/`), no torch.hub.
- **Team classification** uses Gemini Flash to label each player crop (team A, B, or other). It is the only proprietary dependency, swappable for an open VLM.

## Postscript: SAM 3.1

[SAM 3.1 (Object Multiplex)](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) was released after this project was completed. It tracks objects jointly in shared-memory buckets instead of SAM 3's per-object passes, which directly attacks the cost axis measured above — so here is the same benchmark for its native dense-tracking pipeline (`scripts/fps_sam31_native.py`: text prompt `"player"`, its own detector, `max_num_objects=32`), on the same clips and the same RTX 5090:

| clip | frames | objects | fps | GB |
|---|---|---|---|---|
| basketball-1 | 600 | 18 | 7.2 | 23.7 |
| basketball-2 | 360 | 15 | 9.5 | 19.0 |
| soccer-1 | 600 | 19 | 7.4 | 24.1 |
| soccer-2 | 740 | 24 | 6.5 | 27.2 |
| **aggregate** | **2300** | | **7.3** | **27.2** |

Multiplex makes dense tracking substantially faster than uniform SAM 3 (7.3 vs 5.5 fps aggregate — while also running its own detection, which is inseparable and therefore included; the tables above exclude detection, ~0.001 s/frame with YOLOX). Selective mask propagation is still ~2× faster at ~5× less memory: it wins by not tracking everything, not by tracking everything faster, so the two approaches compose rather than compete. The script makes three memory adjustments to fit a 32 GB card at all (the upstream demo targets 80 GB H100s); each is documented in the script.

```bash
uv run python scripts/fps_sam31_native.py     # --render for videos
```

SAM 3.1 lives in `vendor/sam3-with-3.1`, separate from the frozen `vendor/sam3` snapshot that all paper results run on: SAM 3.1 modifies shared model internals (attention/backbone refactors), and keeping both pinned keeps every reported number reproducible. Both packages import as `sam3`, so the script shadows the installed one via `sys.path` for its own process only.

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

Code written for this project is MIT-licensed (see `LICENSE`). Vendored components retain their original licenses.

## Citation

```bibtex
@article{holmberg2026selectivemaskpropagation,
  title={Selective Mask Propagation for Multi-Object Tracking},
  author={Holmberg, Alexander},
  journal={arXiv preprint arXiv:2606.13033},
  year={2026}
}
```
