"""
Standalone ViTPose image processor — no transformers dependency.

Handles preprocessing (affine transform + normalize) and post-processing
(heatmaps → keypoint coordinates via DARK unbiased data processing).

Ported verbatim from:
  transformers/models/vitpose/image_processing_vitpose.py
"""

import itertools
import math
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# BatchFeature — minimal dict subclass that supports .to(device)
# ---------------------------------------------------------------------------

class BatchFeature(dict):
    def to(self, device):
        return BatchFeature(
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.items()}
        )


# ---------------------------------------------------------------------------
# Preprocessing helpers (ported from HF image_processing_vitpose.py)
# ---------------------------------------------------------------------------

def box_to_center_and_scale(
    box,
    image_width: int,
    image_height: int,
    normalize_factor: float = 200.0,
    padding_factor: float = 1.25,
):
    top_left_x, top_left_y, width, height = box[:4]
    aspect_ratio = image_width / image_height
    center = np.array([top_left_x + width * 0.5, top_left_y + height * 0.5], dtype=np.float32)

    if width > aspect_ratio * height:
        height = width * 1.0 / aspect_ratio
    elif width < aspect_ratio * height:
        width = height * aspect_ratio

    scale = np.array([width / normalize_factor, height / normalize_factor], dtype=np.float32)
    scale = scale * padding_factor
    return center, scale


def get_warp_matrix(theta: float, size_input, size_dst, size_target):
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * math.cos(theta)
        + 0.5 * size_input[1] * math.sin(theta)
        + 0.5 * size_target[0]
    )
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * math.sin(theta)
        - 0.5 * size_input[1] * math.cos(theta)
        + 0.5 * size_target[1]
    )
    return matrix


def warp_affine(src, M, size):
    """Affine warp using OpenCV (bilinear interpolation, same as scipy order=1).

    Args:
        src: HWC source image
        M: 2x3 forward affine matrix (source -> destination)
        size: (height, width) of output — scipy convention
    """
    dsize = (size[1], size[0])  # cv2 wants (width, height)
    return cv2.warpAffine(src, M, dsize, flags=cv2.INTER_LINEAR, borderValue=0)


def coco_to_pascal_voc(bboxes: np.ndarray) -> np.ndarray:
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0] - 1
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1] - 1
    return bboxes


# ---------------------------------------------------------------------------
# Post-processing helpers (ported from HF image_processing_vitpose.py)
# ---------------------------------------------------------------------------

def get_keypoint_predictions(heatmaps: np.ndarray):
    batch_size, num_keypoints, _, width = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((batch_size, num_keypoints, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((batch_size, num_keypoints, 1))
    scores = np.amax(heatmaps_reshaped, 2).reshape((batch_size, num_keypoints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % width
    preds[:, :, 1] = preds[:, :, 1] // width

    preds = np.where(np.tile(scores, (1, 1, 2)) > 0.0, preds, -1)
    return preds, scores


def post_dark_unbiased_data_processing(coords, batch_heatmaps, kernel=3):
    batch_size, num_keypoints, height, width = batch_heatmaps.shape
    num_coords = coords.shape[0]
    radius = int((kernel - 1) // 2)
    batch_heatmaps = gaussian_filter(
        batch_heatmaps, sigma=(0, 0, 0.8, 0.8), radius=(0, 0, radius, radius)
    )
    batch_heatmaps = np.clip(batch_heatmaps, 0.001, 50)
    batch_heatmaps = np.log(batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge"
    ).flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (width + 2)
    index += (width + 2) * (height + 2) * np.arange(0, batch_size * num_keypoints).reshape(
        -1, num_keypoints
    )
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + width + 2]
    ix1y1 = batch_heatmaps_pad[index + width + 3]
    ix1_y1_ = batch_heatmaps_pad[index - width - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - width]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(num_coords, num_keypoints, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(num_coords, num_keypoints, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze()
    return coords


def transform_preds(coords, center, scale, output_size):
    scale = scale * 200.0
    scale_y = scale[1] / (output_size[0] - 1.0)
    scale_x = scale[0] / (output_size[1] - 1.0)

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target_coords


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class VitPoseProcessor:
    """Drop-in replacement for transformers.AutoProcessor for ViTPose."""

    def __init__(
        self,
        size: Optional[dict] = None,
        image_mean=None,
        image_std=None,
        normalize_factor: float = 200.0,
    ):
        self.size = size or {"height": 256, "width": 192}
        self.image_mean = image_mean if image_mean is not None else IMAGENET_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STD
        self.normalize_factor = normalize_factor

    def __call__(
        self,
        images,
        boxes,
        return_tensors: str = "pt",
    ) -> BatchFeature:
        if not isinstance(images, list):
            images = [images]

        all_crops = []
        for image, image_boxes in zip(images, boxes):
            img_np = np.array(image) if isinstance(image, Image.Image) else image.copy()
            # Ensure HWC uint8
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)

            for box in image_boxes:
                center, scale = box_to_center_and_scale(
                    box,
                    image_width=self.size["width"],
                    image_height=self.size["height"],
                    normalize_factor=self.normalize_factor,
                )
                warp_size = (self.size["width"], self.size["height"])
                transformation = get_warp_matrix(
                    0, center * 2.0, np.array(warp_size) - 1.0, scale * 200.0
                )
                crop = warp_affine(
                    src=img_np, M=transformation, size=(warp_size[1], warp_size[0])
                )

                # Rescale [0,255] → [0,1]
                crop = crop.astype(np.float32) / 255.0
                # Normalize
                crop = (crop - self.image_mean) / self.image_std
                # HWC → CHW
                crop = crop.transpose(2, 0, 1)
                all_crops.append(crop)

        pixel_values = torch.from_numpy(np.stack(all_crops))
        return BatchFeature({"pixel_values": pixel_values})

    def _keypoints_from_heatmaps_gpu(self, heatmaps, centers, scales, kernel=11):
        """DARK keypoint extraction on GPU. Same math as CPU path, runs on CUDA.

        Args:
            heatmaps: (B, K, H, W) float32 tensor on GPU
            centers: (B, 2) numpy array
            scales: (B, 2) numpy array
        Returns:
            preds: (B, K, 2) numpy array, scores: (B, K, 1) numpy array
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device

        # 1. Argmax to find peak locations
        flat = heatmaps.reshape(B, K, -1)
        scores, idx = flat.max(dim=2)
        cx = (idx % W).float()
        cy = (idx // W).float()
        invalid = scores <= 0.0
        cx[invalid] = -1
        cy[invalid] = -1

        # 2. Gaussian blur (separable conv, matches scipy gaussian_filter)
        radius = (kernel - 1) // 2
        ax = torch.arange(kernel, device=device, dtype=torch.float32) - radius
        k1d = torch.exp(-0.5 * (ax / 0.8) ** 2)
        k1d = k1d / k1d.sum()
        x = heatmaps.reshape(B * K, 1, H, W)
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode='reflect'), k1d.reshape(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode='reflect'), k1d.reshape(1, 1, -1, 1))
        blurred = x.reshape(B, K, H, W)

        # 3. DARK sub-pixel refinement
        blurred = blurred.clamp(min=0.001, max=50.0).log()
        padded = F.pad(blurred, (1, 1, 1, 1), mode='replicate').reshape(B, K, -1)
        pw = W + 2
        fi = (cy.long() + 1) * pw + (cx.long() + 1)
        fi = fi.clamp(0, padded.shape[-1] - 1)

        def g(off):
            return padded.gather(2, (fi + off).clamp(0, padded.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)

        i_    = g(0)
        ix1   = g(1)
        ix1_  = g(-1)
        iy1   = g(pw)
        iy1_  = g(-pw)
        ix1y1 = g(pw + 1)
        ix1_y1_ = g(-pw - 1)

        dx  = 0.5 * (ix1 - ix1_)
        dy  = 0.5 * (iy1 - iy1_)
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)

        # Manual 2x2 inverse (matches np.linalg.inv(H + eps*I))
        eps = torch.finfo(torch.float32).eps
        dxx_e = dxx + eps
        dyy_e = dyy + eps
        det = dxx_e * dyy_e - dxy * dxy
        cx = cx - (dyy_e * dx - dxy * dy) / det
        cy = cy - (-dxy * dx + dxx_e * dy) / det

        # 4. Transform to image coordinates
        centers_t = torch.from_numpy(centers).to(device)
        scales_t = torch.from_numpy(scales).to(device) * 200.0
        sx = scales_t[:, 0:1] / (W - 1.0)
        sy = scales_t[:, 1:2] / (H - 1.0)
        cx = cx * sx + centers_t[:, 0:1] - scales_t[:, 0:1] * 0.5
        cy = cy * sy + centers_t[:, 1:2] - scales_t[:, 1:2] * 0.5

        return torch.stack([cx, cy], dim=-1).cpu().numpy(), scores.unsqueeze(-1).cpu().numpy()

    def keypoints_from_heatmaps(self, heatmaps, center, scale, kernel=11):
        batch_size, _, height, width = heatmaps.shape
        coords, scores = get_keypoint_predictions(heatmaps)
        preds = post_dark_unbiased_data_processing(coords, heatmaps, kernel=kernel)
        for i in range(batch_size):
            preds[i] = transform_preds(
                preds[i], center=center[i], scale=scale[i], output_size=[height, width]
            )
        return preds, scores

    def post_process_pose_estimation(
        self,
        outputs,
        boxes,
        kernel_size: int = 11,
        threshold=None,
        target_sizes=None,
    ):
        heatmaps_tensor = outputs.heatmaps
        batch_size, num_keypoints, _, _ = heatmaps_tensor.shape

        centers = np.zeros((batch_size, 2), dtype=np.float32)
        scales = np.zeros((batch_size, 2), dtype=np.float32)
        flattened_boxes = list(itertools.chain(*boxes))
        for i in range(batch_size):
            if target_sizes is not None:
                image_width, image_height = target_sizes[i][0], target_sizes[i][1]
                scale_factor = np.array([image_width, image_height, image_width, image_height])
                flattened_boxes[i] = flattened_boxes[i] * scale_factor
            width, height = self.size["width"], self.size["height"]
            center, scale = box_to_center_and_scale(
                flattened_boxes[i], image_width=width, image_height=height
            )
            centers[i, :] = center
            scales[i, :] = scale

        preds, scores = self._keypoints_from_heatmaps_gpu(
            heatmaps_tensor, centers, scales, kernel=kernel_size
        )

        all_boxes = np.zeros((batch_size, 4), dtype=np.float32)
        all_boxes[:, 0:2] = centers[:, 0:2]
        all_boxes[:, 2:4] = scales[:, 0:2]

        poses = torch.tensor(preds)
        scores = torch.tensor(scores)
        labels = torch.arange(0, num_keypoints)
        bboxes_xyxy = torch.tensor(coco_to_pascal_voc(all_boxes))

        results = []
        pose_bbox_pairs = zip(poses, scores, bboxes_xyxy)

        for image_bboxes in boxes:
            image_results = []
            for _ in image_bboxes:
                pose, score, bbox_xyxy = next(pose_bbox_pairs)
                score = score.squeeze()
                keypoints_labels = labels
                if threshold is not None:
                    keep = score > threshold
                    pose = pose[keep]
                    score = score[keep]
                    keypoints_labels = keypoints_labels[keep]
                pose_result = {
                    "keypoints": pose,
                    "scores": score,
                    "labels": keypoints_labels,
                    "bbox": bbox_xyxy,
                }
                image_results.append(pose_result)
            results.append(image_results)

        return results
