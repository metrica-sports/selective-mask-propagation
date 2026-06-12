"""Standalone PARSeq scene text recognition model — pure PyTorch.

Inference-only port of strhub.models.parseq from baudm/parseq, stripped
of the pytorch_lightning training wrapper, hydra config machinery, nltk,
and all training-time code (loss, permutation generation, optimizer
config, validation hooks).

The state dict layout matches the original baudm/parseq checkpoint
exactly, so the released `parseq-bb5792a6.pt` weights load directly
with no key remapping. The math is bit-identical to the original since
we use the same nn.Module classes with the same parameters.

Module hierarchy:
    PARSeq
    ├── encoder: Encoder (timm.VisionTransformer, no class token, no head)
    ├── decoder: Decoder
    │   └── layers: N × DecoderLayer (two-stream attention)
    ├── head: nn.Linear (per-token classifier)
    ├── text_embed: TokenEmbedding
    └── pos_queries: nn.Parameter

Ported from:
    strhub/models/parseq/model.py
    strhub/models/parseq/modules.py
"""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from timm.models.vision_transformer import PatchEmbed, VisionTransformer


@dataclass
class PARSeqConfig:
    """Architecture hyperparameters for the base PARSeq model.

    Defaults match strhub/configs/main.yaml + configs/model/parseq.yaml +
    configs/charset/94_full.yaml — the configuration that produces the
    `parseq-bb5792a6.pt` checkpoint.
    """
    charset: str = (
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    )
    max_label_length: int = 25
    img_size: tuple = (32, 128)
    patch_size: tuple = (4, 8)
    embed_dim: int = 384
    enc_num_heads: int = 6
    enc_mlp_ratio: int = 4
    enc_depth: int = 12
    dec_num_heads: int = 12
    dec_mlp_ratio: int = 4
    dec_depth: int = 1
    decode_ar: bool = True
    refine_iters: int = 1

    @staticmethod
    def from_dict(d: dict) -> "PARSeqConfig":
        return PARSeqConfig(
            charset=d["charset"],
            max_label_length=d["max_label_length"],
            img_size=tuple(d["img_size"]),
            patch_size=tuple(d["patch_size"]),
            embed_dim=d["embed_dim"],
            enc_num_heads=d["enc_num_heads"],
            enc_mlp_ratio=d["enc_mlp_ratio"],
            enc_depth=d["enc_depth"],
            dec_num_heads=d["dec_num_heads"],
            dec_mlp_ratio=d["dec_mlp_ratio"],
            dec_depth=d["dec_depth"],
            decode_ar=d["decode_ar"],
            refine_iters=d["refine_iters"],
        )


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class DecoderLayer(nn.Module):
    """Pre-LN transformer decoder layer with two-stream (XLNet) attention.

    Matches strhub.models.parseq.modules.DecoderLayer exactly, including
    parameter names, so that the original checkpoint loads with no key
    remapping. Two attention streams: one over the query positions and
    one over the content positions, sharing the cross-attention to the
    image memory.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward_stream(
        self,
        tgt: Tensor,
        tgt_norm: Tensor,
        tgt_kv: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
    ):
        tgt2, _ = self.self_attn(
            tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt2, _ = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
        update_content: bool = True,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)
        if update_content:
            content = self.forward_stream(
                content, content_norm, content_norm, memory, content_mask, content_key_padding_mask
            )
        return query, content


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer: DecoderLayer, num_layers: int, norm: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
    ):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(
                query, content, memory, query_mask, content_mask, content_key_padding_mask, update_content=not last
            )
        return self.norm(query)


class Encoder(VisionTransformer):
    """ViT encoder with no class token, no global pool, no head.

    Subclasses timm's VisionTransformer to inherit patch embed, position
    embed, and transformer blocks. Returns the full token sequence as
    image memory for the decoder.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            num_classes=0,
            global_pool='',
            class_token=False,
        )

    def forward(self, x):
        return self.forward_features(x)


class PARSeq(nn.Module):
    """Permutation-Autoregressive Sequence model for scene text recognition.

    Inference-only: greedy autoregressive decoding with optional cloze
    refinement passes. Training-time permutation logic, loss computation,
    and Lightning hooks are intentionally omitted — at inference we only
    need encode → decode → head.

    Algorithm:
        1. Encode the input image into a sequence of patch tokens via
           the ViT encoder. The encoder output is the cross-attention
           memory for the decoder.
        2. Autoregressively decode tokens one at a time using the
           canonical forward permutation. The lookahead causal mask
           ensures position i can only attend to positions ≤ i.
        3. After AR decoding completes (or all sequences emit EOS),
           optionally run `refine_iters` cloze passes that re-decode
           with the previous output as context, masked to ignore each
           position's own previous prediction.
    """

    def __init__(
        self,
        num_tokens: int,
        max_label_length: int,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        embed_dim: int,
        enc_num_heads: int,
        enc_mlp_ratio: int,
        enc_depth: int,
        dec_num_heads: int,
        dec_mlp_ratio: int,
        dec_depth: int,
        decode_ar: bool = True,
        refine_iters: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        self.encoder = Encoder(
            img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio
        )
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, num_tokens - 2)
        self.text_embed = TokenEmbedding(num_tokens, embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.zeros(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)

    @property
    def _device(self) -> torch.device:
        return next(self.head.parameters(recurse=False)).device

    def encode(self, img: Tensor) -> Tensor:
        return self.encoder(img)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        tgt_query: Optional[Tensor] = None,
        tgt_query_mask: Optional[Tensor] = None,
    ) -> Tensor:
        N, L = tgt.shape
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, tokenizer, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        num_steps = max_length + 1
        memory = self.encode(images)

        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        tgt_mask = query_mask = torch.triu(
            torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1
        )

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), tokenizer.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = tokenizer.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            tgt_in = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            for _ in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, : tgt_in.shape[1]]
                )
                logits = self.head(tgt_out)

        return logits
