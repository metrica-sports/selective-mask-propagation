# SAM-Deep-EIoU

Selective mask propagation for multi-object tracking. Monitors the assignment margin in the Hungarian cost matrix and selectively invokes SAM to preserve identity through occlusions. Only modifies the base tracker's output when positive evidence of an identity switch is found.

**86.8 HOTA on SportsMOT** — [#1 on the official leaderboard](https://www.codabench.org/competitions/13077/#/results-tab) (June 2026). Consistent improvements across three base trackers on DanceTrack.

[[Paper]](TODO)

https://github.com/user-attachments/assets/1655577c-ede3-4fb0-bc71-860e771df183

https://github.com/user-attachments/assets/e1e6c7ce-27f5-4fec-89cd-0581b4c199bf

## Results

### SportsMOT Test

Scored by the official remote evaluator — see the [SportsMOT leaderboard](https://www.codabench.org/competitions/13077/#/results-tab) (listed as `holma91`).

| Method | HOTA | AssA | IDF1 | MOTA |
|--------|------|------|------|------|
| Deep-EIoU | 77.2 | 67.7 | 79.8 | 96.3 |
| SAM3-Deep-EIoU (with GTA) | **86.8** | **84.2** | **93.2** | **97.3** |

### DanceTrack Val

| Base Tracker | HOTA | + SAM2 | + SAM3 |
|-------------|----------|--------|--------|
| SORT | 39.8 | 45.0 (+5.2) | **46.1** (+6.2) |
| ByteTrack | 54.6 | 60.3 (+5.7) | **61.2** (+6.6) |
| Deep-EIoU | 51.7 | 57.7 (+6.0) | **59.7** (+8.0) |

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
uv run python scripts/download_precomputed.py sportsmot --split val       # 1.4 GB
uv run python scripts/download_precomputed.py sportsmot                   # all splits, 7.5 GB
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
uv run python scripts/download_precomputed.py dancetrack                  # val, 0.6 GB
```

## Quickstart

Run on a single SportsMOT val sequence (basketball, +9.4 HOTA over baseline):

```bash
uv run python -m sam_deep_eiou.sportsmot \
  --input data/sportsmot/dataset/val/v_00HRwkvvjtQ_c005 \
  --precomputed --sam3
```

Run on a single DanceTrack val sequence (+26.7 HOTA over baseline):

```bash
uv run python -m sam_deep_eiou.dancetrack \
  --input data/dancetrack/val/dancetrack0007 --precomputed --sam3
```

### Test the augment API

Runs Deep-EIoU, augments with SAM, and prints before/after metrics:

```bash
uv run python -m sam_deep_eiou.augment data/sportsmot/dataset/val/v_00HRwkvvjtQ_c005 --sam3
```

## API

To augment your own tracker, extract the assignment margin from the cost matrix (`second_best - best` per matched column) and call `augment`:

```python
from sam_deep_eiou.augment import augment

# tracks: {frame: {track_id: [x1, y1, x2, y2]}}
# margins: {frame: {track_id: float}}
tracks, margins = my_tracker.run(detections)

corrected_tracks = augment(tracks, margins, "path/to/sequence", sam3=True)
```

## Reproduce Paper Results

### SportsMOT (Table 3)

```bash
bash scripts/run_sportsmot.sh val sam3
bash scripts/run_sportsmot.sh test sam3
uv run python scripts/build_submission.py --sam3
```

### DanceTrack (Table 1)

```bash
bash scripts/run_dancetrack.sh val sam3
bash scripts/run_dancetrack.sh val sam3 --tracker bytetrack
bash scripts/run_dancetrack.sh val sam3 --tracker sort
```

## Global Track Association (GTA)

GTA links tracklets across frame-boundary exits using jersey recognition, team classification, and appearance embeddings. It operates on finished tracklets and does not modify the tracking or SAM steps. Enable with `--gta` (requires `GEMINI_API_KEY` in `.env`).

```bash
uv run python -m sam_deep_eiou.sportsmot \
  --input data/sportsmot/dataset/val/v_00HRwkvvjtQ_c001 \
  --precomputed --gta --sam3
```

The current GTA implementation is not optimized for speed or model efficiency:

- **Pose estimation** uses ViTPose-large via HuggingFace Transformers. Model loading is slow. A smaller ViTPose variant would likely produce equivalent results since only torso keypoints (shoulders + hips) are used.
- **Jersey OCR** uses PARSeq via HuggingFace Transformers. Same slow loading issue.
- **Team classification** uses Gemini Flash, a proprietary API. The task is simple (classify player crop as team A, team B, or other) and could be replaced with a small open-source VLM.

These are straightforward to improve and would make good contributions.

## Acknowledgments

This repository builds on and vendors code from the following projects:

- [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU) — base tracker (`sam_deep_eiou/deep_eiou/`)
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) (MIT) and [SORT](https://github.com/abewley/sort) (GPL-3.0) — alternative base trackers for the cross-tracker comparison
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) — detector (`sam_deep_eiou/yolox/`)
- [OSNet / deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) (MIT) — appearance embeddings (`sam_deep_eiou/osnet/`)
- [SAM 2](https://github.com/facebookresearch/sam2) (Apache-2.0) and [SAM 3](https://github.com/facebookresearch/sam3) (SAM License) — VOS models, vendored in `sam2/` and `sam3/` with their original licenses
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) (MIT) — evaluation (`TrackEval/`)
- [GTA](https://github.com/sjc042/gta-link) — global tracklet association formulation
- Jersey OCR follows [Koshkina & Elder](https://github.com/mkoshkina/jersey-number-pipeline), using [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and [PARSeq](https://github.com/baudm/parseq) via HuggingFace Transformers

Code written for this project is MIT-licensed (see `LICENSE`). Vendored components retain their original licenses.

## Citation

TODO when paper is up. 
