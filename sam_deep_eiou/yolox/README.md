# YOLOX Inference

Minimal YOLOX inference code ported from the [MixSort](https://github.com/MCG-NJU/MixSort) repo (which itself is a ByteTrack fork). Only the inference path is included — all training code, loss functions, data augmentation, and the experiment system have been removed.

## Checkpoints

Under `checkpoints/` (gitignored), from MixSort's pretrained weights:

- `yolox_x_sports_train.pth.tar` — trained on SportsMOT train split. Use for val evaluation.
- `yolox_x_sports_mix.pth.tar` — trained on SportsMOT train+val splits. Use for test evaluation.

Both are YOLOX-X (`depth=1.33, width=1.25, num_classes=1`), input size `(800, 1440)`, conf threshold `0.01`, NMS threshold `0.7`.

## Key detail: BatchNorm eps

The YOLOX experiment system sets `eps=1e-3` on all BatchNorm2d layers via an `init_yolo` function. This is **not** stored in the checkpoint (eps is a module attribute, not a state_dict entry). PyTorch defaults to `eps=1e-5`, which produces different BN outputs and wrong detections. Any code loading a YOLOX checkpoint must set `eps=1e-3` before calling `load_state_dict`.

## Verification

```
uv run python -m sam_deep_eiou.yolox.test_yolox
```

Runs inference on 10 SportsMOT val clips and compares against pre-computed `det.txt` files. Results: exact detection count match on all clips, sub-pixel box differences (mean <1px, p99 <0.1px), score differences <0.005.
