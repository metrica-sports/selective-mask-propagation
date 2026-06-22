# Selective Mask Propagation

Selective mask propagation (SMP) for multi-object tracking. Monitors the assignment margin in the Hungarian cost matrix and selectively invokes SAM to preserve identity through occlusions. Only modifies the base tracker's output when positive evidence of an identity switch is found. The flagship instantiation, **SAM3-Deep-EIoU**, combines Deep-EIoU with SAM 3.

**87.2 HOTA on SportsMOT** — [#1 on the official leaderboard](https://www.codabench.org/competitions/13077/#/results-tab) (June 2026). Consistent improvements across three base trackers on DanceTrack.

[[Paper]](https://arxiv.org/abs/2606.13033)

![SportsMOT test leaderboard — SAM3-Deep-EIoU (holma91) ranked #1 at 87.16 HOTA](https://github.com/user-attachments/assets/34037fc5-9fe9-4cb9-a8bc-472efdd12635)

https://github.com/user-attachments/assets/1655577c-ede3-4fb0-bc71-860e771df183

https://github.com/user-attachments/assets/e1e6c7ce-27f5-4fec-89cd-0581b4c199bf

## Results

### SportsMOT Test

Scored by the official remote evaluator — see the [SportsMOT leaderboard](https://www.codabench.org/competitions/13077/#/results-tab) (listed as `holma91`).

| Method | HOTA | AssA | IDF1 | MOTA |
|--------|------|------|------|------|
| Deep-EIoU | 77.2 | 67.7 | 79.8 | 96.3 |
| SAM3-Deep-EIoU (with GTA) | **87.2** | **84.2** | **93.6** | **98.1** |

### DanceTrack Val

| Base Tracker | HOTA | + SAM2 | + SAM3 |
|-------------|----------|--------|--------|
| SORT | 39.8 | 45.0 (+5.2) | **46.1** (+6.2) |
| ByteTrack | 54.6 | 60.3 (+5.7) | **61.2** (+6.6) |
| Deep-EIoU | 51.7 | 57.7 (+6.0) | **59.7** (+8.0) |

## Efficiency

We report the **amortized throughput of selective mask propagation**, defined as

> **fps = total video frames / wall-clock(SAM propagation + merge)**, over *every* frame of the clip — not just the dispatched ones.

This is the marginal cost the method adds on top of the base tracker. Detection (YOLOX + OSNet) and base tracking are separate, shared stages and are not counted here; the base tracker's association step runs at ~1000 fps, so the SAM step dominates the added cost.

On SportsMOT test (150 clips, 94.8k frames, RTX PRO 6000):

| | fps (amortized) | Peak VRAM |
|---|---|---|
| **SAM + merge** | **~13** | **5.4 GB** (max 6.1) |
| basketball | 11 | |
| football | 21 | |
| volleyball | 11 | |

The cost is low **not** because SAM skips most frames — it runs on ~79% of them — but because each pass tracks only the few ambiguous objects (~4.4 on average), not all 10–22 players. Throughput therefore scales with how many windows fire, which depends on the sport and on `τ_entry`.

See it on your own GPU — runs the SAM step on three bundled clips (one per sport, no dataset download needed) and prints the throughput:

```bash
uv run python scripts/show_fps.py
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

- **Pose estimation** uses ViTPose+ (base) via a standalone implementation (`selective_mask_propagation/vitpose/`) loading local safetensors. A smaller ViTPose variant would likely produce equivalent results since only torso keypoints (shoulders + hips) are used.
- **Jersey OCR** uses PARSeq via a standalone implementation (`selective_mask_propagation/parseq/`), no torch.hub.
- **Team classification** uses Gemini Flash, a proprietary API. The task is simple (classify player crop as team A, team B, or other) and could be replaced with a small open-source VLM — this would make a good contribution.

## Acknowledgments

This repository builds on and vendors code from the following projects:

- [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU) — base tracker (`selective_mask_propagation/deep_eiou/`)
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) (MIT) and [SORT](https://github.com/abewley/sort) (GPL-3.0) — alternative base trackers for the cross-tracker comparison
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) — detector (`selective_mask_propagation/yolox/`)
- [OSNet / deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) (MIT) — appearance embeddings (`selective_mask_propagation/osnet/`)
- [SAM 2](https://github.com/facebookresearch/sam2) (Apache-2.0) and [SAM 3](https://github.com/facebookresearch/sam3) (SAM License) — VOS models, vendored in `vendor/` with their original licenses
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) (MIT) — evaluation (`TrackEval/`)
- [GTA](https://github.com/sjc042/gta-link) — global tracklet association formulation
- Jersey OCR follows [Koshkina & Elder](https://github.com/mkoshkina/jersey-number-pipeline), using [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and [PARSeq](https://github.com/baudm/parseq) via standalone inference-only implementations (`selective_mask_propagation/{vitpose,parseq}/`)

Code written for this project is MIT-licensed (see `LICENSE`). Vendored components retain their original licenses.

## Citation

```bibtex
@article{holmberg2026samdeepeiou,
  title={SAM-Deep-EIoU: Selective Mask Propagation for Multi-Object Tracking},
  author={Holmberg, Alexander},
  journal={arXiv preprint arXiv:2606.13033},
  year={2026}
}
```
