# OSNet Inference

Minimal OSNet-x1.0 embedding extraction ported from the [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU) repo, which uses [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid). Only the inference path is included — classifier head, pretrained weight downloading, training utilities, and all model variants except `osnet_x1_0` have been removed.

## Checkpoint

Under `checkpoints/` (gitignored):

- `sports_model.pth.tar-60` — OSNet-x1.0 fine-tuned on SportsMOT by Deep-EIoU authors. 512-dim embeddings, 2.2M params, ~29MB.

Loads 565/567 keys (2 discarded: `classifier.weight`, `classifier.bias` — we don't use the classification head).

## Model

Input: detection crops resized to 256x128, ImageNet-normalized. Output: 512-dim L2-normalizable embedding vector.

## Verification

```
uv run python -m smp.osnet.test_osnet
```

Extracts embeddings from detection crops on 10 SportsMOT val clips and compares against pre-computed `emb.npy` files via cosine similarity. Results: 100% of pairs > 0.999 cosine similarity across all clips, min similarity 0.999952.
