"""Standalone PARSeq loader — drop-in replacement for torch.hub PARSeq.

Usage:
    from selective_mask_propagation.parseq.loader import load_parseq
    model, tokenizer = load_parseq()
"""

import json
from pathlib import Path
from typing import Tuple

from safetensors.torch import load_file

from .model import PARSeq, PARSeqConfig
from .tokenizer import Tokenizer

WEIGHTS_PATH = Path(__file__).parent / "checkpoints" / "parseq.safetensors"
CONFIG_PATH = Path(__file__).parent / "checkpoints" / "parseq-config.json"


def load_parseq() -> Tuple[PARSeq, Tokenizer]:
    """Load the local PARSeq model and tokenizer.

    Returns:
        (model, tokenizer) — model is in eval mode on CUDA. Tokenizer is
        independent of device.
    """
    with open(CONFIG_PATH) as f:
        config = PARSeqConfig.from_dict(json.load(f))

    tokenizer = Tokenizer(config.charset)

    model = PARSeq(
        num_tokens=len(tokenizer),
        max_label_length=config.max_label_length,
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        enc_num_heads=config.enc_num_heads,
        enc_mlp_ratio=config.enc_mlp_ratio,
        enc_depth=config.enc_depth,
        dec_num_heads=config.dec_num_heads,
        dec_mlp_ratio=config.dec_mlp_ratio,
        dec_depth=config.dec_depth,
        decode_ar=config.decode_ar,
        refine_iters=config.refine_iters,
        dropout=0.0,
    )
    model.load_state_dict(load_file(str(WEIGHTS_PATH)))
    model.eval()
    model = model.to("cuda")

    print(f"PARSeq loaded on {next(model.parameters()).device}.")
    return model, tokenizer
