"""YOLOX-X inference on SportsMOT-style image sequences."""

import glob
import os
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .models.yolox import YOLOX
from .models.yolo_pafpn import YOLOPAFPN
from .models.yolo_head import YOLOXHead
from .utils.boxes import postprocess

NUM_CLASSES = 1
DEPTH = 1.33
WIDTH = 1.25
TEST_SIZE = (800, 1440)
CONF_THRESH = 0.01
NMS_THRESH = 0.7
RGB_MEANS = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "yolox_x_sports_train.pth.tar")


def preproc(image, input_size, mean, std):
    """Aspect-ratio-preserving resize + normalize. From YOLOX data_augment.py."""
    padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    padded_img -= mean
    padded_img /= std
    padded_img = padded_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def build_model(device: torch.device, checkpoint: str = CHECKPOINT) -> torch.nn.Module:
    """Build YOLOX-X with 1 class and load checkpoint."""
    backbone = YOLOPAFPN(depth=DEPTH, width=WIDTH)
    head = YOLOXHead(NUM_CLASSES, WIDTH, in_channels=[256, 512, 1024])
    model = YOLOX(backbone, head)

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"YOLOX loaded: {checkpoint}")
    return model


class _SequenceDataset(Dataset):
    def __init__(self, img_dir):
        self.frames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        img = cv2.imread(frame_path)
        img_t, ratio = preproc(img, TEST_SIZE, RGB_MEANS, RGB_STD)
        return (
            torch.from_numpy(img_t).float(),
            torch.tensor(idx, dtype=torch.int64),
            torch.tensor(ratio, dtype=torch.float32),
        )


def detect_sequence(
    model: torch.nn.Module,
    seq_dir: str,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """Run YOLOX on a sequence.

    Returns {frame_idx (0-indexed): (N, 5) ndarray [x1,y1,x2,y2,score]}.
    Empty frames get an (0, 5) array.
    """
    dataset = _SequenceDataset(os.path.join(seq_dir, "img1"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    result: Dict[int, np.ndarray] = {}
    for img_batch, frame_indices, ratios in tqdm(loader, desc="YOLOX"):
        img_batch = img_batch.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(img_batch)
            outputs = postprocess(outputs, NUM_CLASSES, CONF_THRESH, NMS_THRESH)

        for i in range(len(frame_indices)):
            frame_idx = frame_indices[i].item()
            ratio = ratios[i].item()
            dets = outputs[i]

            if dets is None:
                result[frame_idx] = np.empty((0, 5))
                continue

            dets = dets.cpu().numpy()
            dets[:, :4] /= ratio
            scores = dets[:, 4] * dets[:, 5]
            result[frame_idx] = np.column_stack([dets[:, :4], scores])

    # Fill any missing frames
    for idx in range(len(dataset)):
        if idx not in result:
            result[idx] = np.empty((0, 5))

    total = sum(len(d) for d in result.values())
    print(f"YOLOX: {total} detections across {len(result)} frames")
    return result
