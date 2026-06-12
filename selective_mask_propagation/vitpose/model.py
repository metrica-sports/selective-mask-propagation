"""
Standalone ViTPose model — pure PyTorch, no transformers dependency.

Module hierarchy matches the HuggingFace state dict keys exactly so that
weights can be loaded with model.load_state_dict() directly.

Ported from:
  transformers/models/vitpose_backbone/modeling_vitpose_backbone.py
  transformers/models/vitpose/modeling_vitpose.py
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VitPoseConfig:
    image_size: tuple = (256, 192)
    patch_size: tuple = (16, 16)
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    num_experts: int = 1
    part_features: int = 256
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True
    use_simple_decoder: bool = True
    scale_factor: int = 4
    num_labels: int = 17

    @staticmethod
    def from_hf_dict(d: dict) -> "VitPoseConfig":
        bb = d.get("backbone_config", {})
        image_size = bb.get("image_size", [256, 192])
        patch_size = bb.get("patch_size", [16, 16])
        id2label = d.get("id2label", {})
        return VitPoseConfig(
            image_size=tuple(image_size),
            patch_size=tuple(patch_size),
            num_channels=bb.get("num_channels", 3),
            hidden_size=bb.get("hidden_size", 768),
            num_hidden_layers=bb.get("num_hidden_layers", 12),
            num_attention_heads=bb.get("num_attention_heads", 12),
            mlp_ratio=bb.get("mlp_ratio", 4),
            num_experts=bb.get("num_experts", 1),
            part_features=bb.get("part_features", 256),
            layer_norm_eps=bb.get("layer_norm_eps", 1e-12),
            qkv_bias=bb.get("qkv_bias", True),
            use_simple_decoder=d.get("use_simple_decoder", True),
            scale_factor=d.get("scale_factor", 4),
            num_labels=len(id2label) if id2label else 17,
        )


@dataclass
class VitPoseOutput:
    heatmaps: torch.Tensor


class PatchEmbeddings(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=2,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class Embeddings(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config)
        num_patches = (config.image_size[0] // config.patch_size[0]) * (
            config.image_size[1] // config.patch_size[1]
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(pixel_values)
        x = x + self.position_embeddings[:, 1:] + self.position_embeddings[:, :1]
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, _ = hidden_states.shape
        shape = (B, N, self.num_heads, self.head_dim)

        q = self.query(hidden_states).view(*shape).transpose(1, 2)
        k = self.key(hidden_states).view(*shape).transpose(1, 2)
        v = self.value(hidden_states).view(*shape).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return out


class SelfOutput(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden_states)


class Attention(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.attention = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(hidden_states))


class MoeMLP(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        hidden_features = config.hidden_size * config.mlp_ratio
        self.part_features = config.part_features
        self.num_experts = config.num_experts

        self.fc1 = nn.Linear(config.hidden_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, config.hidden_size - config.part_features)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_features, config.part_features) for _ in range(config.num_experts)]
        )

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        expert_out = torch.zeros_like(x[:, :, -self.part_features:])
        x = F.gelu(self.fc1(x))
        shared = self.fc2(x)
        indices = indices.view(-1, 1, 1)
        for i in range(self.num_experts):
            expert_out = expert_out + self.experts[i](x) * (indices == i)
        return torch.cat([shared, expert_out], dim=-1)


class MLP(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        hidden_features = config.hidden_size * config.mlp_ratio
        self.fc1 = nn.Linear(config.hidden_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Layer(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Attention(config)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MoeMLP(config) if config.num_experts > 1 else MLP(config)

    def forward(
        self, hidden_states: torch.Tensor, dataset_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.attention(self.layernorm_before(hidden_states)) + hidden_states
        normed = self.layernorm_after(hidden_states)
        if self.num_experts > 1:
            mlp_out = self.mlp(normed, dataset_index)
        else:
            mlp_out = self.mlp(normed)
        return mlp_out + hidden_states


class Encoder(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self, hidden_states: torch.Tensor, dataset_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, dataset_index)
        return hidden_states


class Backbone(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, pixel_values: torch.Tensor, dataset_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embeddings(pixel_values)
        x = self.encoder(x, dataset_index)
        x = self.layernorm(x)
        return x


class ClassicDecoder(nn.Module):
    """2 deconv blocks + 1x1 conv -> heatmaps."""

    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(config.hidden_size, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.conv = nn.Conv2d(256, config.num_labels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.batchnorm1(self.deconv1(x)))
        x = self.relu2(self.batchnorm2(self.deconv2(x)))
        return self.conv(x)


class SimpleDecoder(nn.Module):
    """ReLU + 4x bilinear upsample + 3x3 conv -> heatmaps."""

    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.activation = nn.ReLU()
        self.upsampling = nn.Upsample(scale_factor=config.scale_factor, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(config.hidden_size, config.num_labels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsampling(self.activation(x)))


class VitPoseForPoseEstimation(nn.Module):
    def __init__(self, config: VitPoseConfig):
        super().__init__()
        self.config = config
        self.backbone = Backbone(config)
        self.head = SimpleDecoder(config) if config.use_simple_decoder else ClassicDecoder(config)
        self.patch_height = config.image_size[0] // config.patch_size[0]
        self.patch_width = config.image_size[1] // config.patch_size[1]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        pixel_values: torch.Tensor,
        dataset_index: Optional[torch.Tensor] = None,
    ) -> VitPoseOutput:
        x = self.backbone(pixel_values, dataset_index)
        B = x.shape[0]
        x = x.permute(0, 2, 1).reshape(B, -1, self.patch_height, self.patch_width).contiguous()
        heatmaps = self.head(x)
        return VitPoseOutput(heatmaps=heatmaps)
