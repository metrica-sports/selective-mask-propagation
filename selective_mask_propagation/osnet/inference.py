"""OSNet-x1.0 embedding extraction for detection crops."""

import glob
import os
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from .osnet import build_osnet_x1_0

CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "sports_model.pth.tar-60")

TRANSFORM = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(device: torch.device, checkpoint: str = CHECKPOINT) -> torch.nn.Module:
    """Build OSNet-x1.0 and load sports checkpoint."""
    model = build_osnet_x1_0()

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    model_dict = model.state_dict()
    filtered = OrderedDict(
        (k, v) for k, v in state_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    )
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    print(f"OSNet loaded: {checkpoint} ({len(filtered)}/{len(state_dict)} keys)")
    return model


def extract_embeddings(
    model: torch.nn.Module,
    seq_dir: str,
    detections: Dict[int, np.ndarray],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """Extract OSNet embeddings for all detections in a sequence.

    detections: {frame_idx: (N, 5) ndarray [x1,y1,x2,y2,score]}.
    Returns {frame_idx: (N, 512) ndarray}.
    """
    img_dir = os.path.join(seq_dir, "img1")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    result: Dict[int, np.ndarray] = {}
    for frame_idx in tqdm(sorted(detections.keys()), desc="OSNet"):
        dets = detections[frame_idx]
        if len(dets) == 0:
            result[frame_idx] = np.empty((0, 512))
            continue

        img = Image.open(imgs[frame_idx])
        crops = []
        for det in dets:
            x1, y1, x2, y2 = det[:4]
            crop = img.crop((x1, y1, x2, y2)).convert("RGB")
            crops.append(TRANSFORM(crop))

        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            feats = model(batch)
        result[frame_idx] = feats.cpu().numpy()

    total = sum(len(e) for e in result.values())
    print(f"OSNet: {total} embeddings across {len(result)} frames")
    return result
