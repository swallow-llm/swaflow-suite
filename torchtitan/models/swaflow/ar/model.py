# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Daisuke Oba
#
# Adapted from torchtitan's Llama 3 model definition (BSD 3-Clause License,
# see LICENSE-torchtitan). Modifications are licensed under the Apache
# License, Version 2.0 (see LICENSE and NOTICE).

"""Autoregressive (AR) model.

A standard llama3-style decoder (pre-norm RMSNorm, SwiGLU, RoPE base 5e5) built
from the same component classes as the flow model:

    Attention(causal=True)  ← torchtitan.models.swaflow.model.Attention
    FeedForward             ← torchtitan.models.swaflow.model.FeedForward
    apply_rotary_emb / precompute_freqs_cis ← llama3

QK-norm, RoPE and initialization thus go through the same code, and the AR and
flow models differ only in the objective: causal attention instead of
bidirectional; token embeddings as input instead of SLERP-noised sphere
embeddings; no time conditioning (no sigma_map / adaLN); position i predicts token
i+1 instead of the clean token at i. Without adaLN to supply scale and shift, the
norms carry a learned weight (standard llama3 `nn.RMSNorm(dim, eps)`).

Shapes (d / layers / heads / FFN) match the flow models; the flow models have
slightly more parameters (sigma_map, adaLN_modulation, final_modulation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.llama3.model.model import (
    apply_rotary_emb,
    precompute_freqs_cis,
)
from torchtitan.models.swaflow.model import (
    Attention,
    FeedForward,
    PROFILE_PRODUCTION,
)
from torchtitan.models.utils import get_dense_model_nparams_and_flops, trunc_normal_
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.tools.logging import logger

__all__ = ["ARBlock", "ARModelArgs", "ARTransformer"]


@dataclass
class ARModelArgs(BaseModelArgs):
    """Model arguments for the AR models.

    Unlike `SwaFlowModelArgs` there are no time-conditioning fields (`cond_dim`,
    `sigma_freq_dim`, `schedule_alpha_max`, `antithetic_t_sampling`,
    `t_sampling_eps`), so flow-only values do not appear in the run config.
    `backbone_profile` is fixed to production; it is a field because
    `Attention` / `FeedForward` read it.
    """

    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 49152  # SmolLM2 tokenizer (same as the flow models)
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000  # same as the flow production profile
    max_seq_len: int = 2048
    depth_init: bool = True

    #: Per-head RMSNorm on q and k; on by default, as for the flow models.
    qk_norm: bool = True

    attn_type: str = "sdpa"  # SDPA only (causal via is_causal=True)

    #: Read by `Attention` / `FeedForward`; fixed to production.
    backbone_profile: str = PROFILE_PRODUCTION

    def __post_init__(self) -> None:
        if self.backbone_profile != PROFILE_PRODUCTION:
            raise ValueError(
                "the AR models support only the production profile "
                f"(got {self.backbone_profile!r}); the sflm profile mirrors the "
                "S-FLM reference model and has no AR counterpart"
            )

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        # Only the `[swaflow.model]` entries that apply to AR are read (qk_norm);
        # flow-only entries such as `schedule_alpha_max` have no effect here.
        overrides = getattr(getattr(job_config, "swaflow", None), "model", None)
        if overrides is None:
            return
        value = getattr(overrides, "qk_norm", None)
        if value is not None:
            logger.info(f"[swaflow.model] qk_norm: {self.qk_norm} -> {value} (config override)")
            self.qk_norm = value

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        return get_dense_model_nparams_and_flops(
            self, model, 2 * (self.dim // self.n_heads), seq_len
        )


class ARBlock(nn.Module):
    """Pre-norm Transformer block (same structure as llama3).

    The flow `SphereDiTBlock` without adaLN modulation and gates; attention and
    FFN use the same classes with the same arguments.
    """

    def __init__(self, layer_id: int, model_args: ARModelArgs):
        super().__init__()
        self.attention = Attention(model_args, layer_id, causal=True)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        # With a learned weight (standard llama3): in the flow blocks adaLN
        # supplies scale and shift, which the AR model lacks.
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # depth-scaled init (same scheme as the flow models and llama3)
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self) -> None:
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class ARTransformer(ModelProtocol):
    """Decoder for next-token prediction.

    Attribute names (`tok_embeddings` / `layers` / `norm` / `output`) match llama3
    and the flow model, so upstream `parallelize_llama` (FSDP, AC) is reused
    unchanged.
    """

    def __init__(self, model_args: ARModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = ARBlock(layer_id, model_args)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        # lm_head, untied from the embedding (as in the flow model)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        # same as llama3 and the flow production profile (normal, std=1)
        nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            layer.init_weights()
        self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """token ids [B, L] -> logits [B, L, V].

        Logits at position i predict the token at position i+1; alignment with
        the labels happens in the loss (`ar/loss.py`). In the flow model position
        i predicts position i.
        """
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        return self.output(self.norm(h))
