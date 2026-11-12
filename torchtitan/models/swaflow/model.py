# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Daisuke Oba
#
# Adapted from torchtitan's Llama 3 model definition (BSD 3-Clause License,
# see LICENSE-torchtitan). Modifications are licensed under the Apache
# License, Version 2.0 (see LICENSE and NOTICE).

"""Sphere-DiT: a bidirectional Transformer with adaLN time conditioning.

Inputs are continuous vectors on the unit sphere x ∈ [B, L, d] and the cleanness
α ∈ [0, 1] (α=1 clean, α=0 pure noise). Relative to llama3: no causal mask; each
block is conditioned on σ = −log(1 − α) via adaLN-Zero; the lm_head is untied;
per-head QK-norm is on by default; MHA with SDPA only.

backbone_profile selects the block internals. "production" (default): RMSNorm
without affine, bias-free SwiGLU, llama3 RoPE (adjacent pairs, base 5e5),
bias-free lm_head. "sflm": the S-FLM reference sphere-DiT, so its released
weights load key-for-key: centered LayerNorm with weight, GELU(tanh) 4d MLP with
bias, rotate-half RoPE (base 1e4), zero-initialized lm_head with bias.
RoPE runs in fp32 in both profiles; SLERP runs in sphere_ops (fp64 by default).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.models.attention import ScaledDotProductAttentionWrapper
from torchtitan.models.llama3.model.model import apply_rotary_emb, precompute_freqs_cis
from torchtitan.models.swaflow.metrics import get_attention_stats
from torchtitan.models.swaflow.schedules import (
    LogLinear,
    NoiseSchedule,
    TruncatedScheduleWrapper,
)
from torchtitan.models.swaflow.sphere_ops import slerp, sphere_normalize
from torchtitan.models.utils import get_dense_model_nparams_and_flops, trunc_normal_
from torchtitan.protocols.model import AttentionMasksType, BaseModelArgs
from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.tools.logging import logger


PROFILE_PRODUCTION = "production"
PROFILE_SFLM = "sflm"
_PROFILES = (PROFILE_PRODUCTION, PROFILE_SFLM)


@dataclass
class SwaFlowModelArgs(BaseModelArgs):
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 49152  # SmolLM tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_seq_len: int = 2048
    depth_init: bool = True

    # ---- sphere-DiT ----
    cond_dim: int = 256  # dimension of the time-conditioning vector
    sigma_freq_dim: int = 256  # dimension of the sinusoidal σ embedding
    # Per-head RMSNorm on q and k. Forced off by __post_init__ for the sflm
    # profile (the reference model has no QK-norm).
    qk_norm: bool = True
    # α_max of the static truncated training window [0, α_max] (None = no truncation).
    schedule_alpha_max: float | None = None

    attn_type: str = "sdpa"  # SDPA only (bidirectional, is_causal=False)

    # ---- backbone profile (see the module docstring) ----
    # "production": default backbone
    # "sflm": the S-FLM reference sphere-DiT
    backbone_profile: str = PROFILE_PRODUCTION
    # Residual dropout (0.1 in the reference sphere-DiT).
    dropout: float = 0.0

    # ---- t sampling ----
    # Stratified (antithetic) t sampling, as in the reference
    # training.antithetic_sampling. Makes the α composition of every batch identical.
    # Per-run override: `[swaflow.model] antithetic_t_sampling`.
    antithetic_t_sampling: bool = True
    # Equivalent of the reference training.sampling_eps (1e-3 there). In our
    # convention it keeps α away from the pure-noise end (α->0).
    t_sampling_eps: float = 0.0

    def __post_init__(self) -> None:
        if self.backbone_profile not in _PROFILES:
            raise ValueError(
                f"backbone_profile must be one of {_PROFILES}, "
                f"got {self.backbone_profile!r}"
            )
        if not 0.0 <= self.t_sampling_eps < 1.0:
            raise ValueError(
                f"t_sampling_eps must be in [0, 1), got {self.t_sampling_eps}"
            )
        if self.backbone_profile == PROFILE_SFLM and self.qk_norm:
            # The reference sphere-DiT has no QK-norm.
            logger.info(
                "backbone_profile='sflm': qk_norm turned off "
                "(the reference model has no QK-norm)"
            )
            self.qk_norm = False

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        # Flavor overrides from [swaflow.model] (torchtitan.models.swaflow.job_config).
        # The section is absent when `custom_config_module` is not set.
        overrides = getattr(getattr(job_config, "swaflow", None), "model", None)
        if overrides is None:
            return
        for name in ("schedule_alpha_max", "qk_norm", "antithetic_t_sampling"):
            value = getattr(overrides, name, None)
            if value is None:
                continue
            before = getattr(self, name)
            setattr(self, name, value)
            logger.info(f"[swaflow.model] {name}: {before} -> {value} (config override)")

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # The FLOP estimate assumes causal attention; MFU is approximate here.
        return get_dense_model_nparams_and_flops(
            self, model, 2 * (self.dim // self.n_heads), seq_len
        )


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation. x: [B, L, d]; shift/scale: [B, d]."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Caps α so that α=1 (clean) does not map to σ=+inf; σ_max = -log(1e-7) ≈ 16.1.
# The truncation window (schedule_alpha_max) normally applies first.
_ALPHA_CLAMP_MAX = 1.0 - 1e-7


def alpha_to_sigma(alpha: torch.Tensor) -> torch.Tensor:
    """σ = −log(1 − α), the time input of the S-FLM TimestepEmbedder (both profiles).

    α=0 (pure noise) -> σ=0; α->1 (clean) -> σ->∞.
    """
    return -torch.log1p(-alpha.float().clamp(max=_ALPHA_CLAMP_MAX))


class CenteredLayerNorm(nn.Module):
    """Mean-subtracted LayerNorm with a learned weight and no bias (sflm profile).

    Normalizes in fp32 as the reference does, then casts back to the input dtype.
    The reference runs under bf16 autocast, while torchtitan casts parameters with
    FSDP2's MixedPrecisionPolicy; casting back here gives the same dtype policy
    (a no-op in fp32). eps is the F.layer_norm default 1e-5.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(x.float(), (self.dim,)) * self.weight.float()
        return out.to(x.dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


def make_norm(model_args: "SwaFlowModelArgs") -> nn.Module:
    """Normalization layer for the profile.

    production: RMSNorm without affine (adaLN supplies scale and shift).
    sflm: mean-subtracted LayerNorm with a learned weight, as in the reference.
    """
    if model_args.backbone_profile == PROFILE_SFLM:
        return CenteredLayerNorm(model_args.dim)
    return nn.RMSNorm(
        model_args.dim, eps=model_args.norm_eps, elementwise_affine=False
    )


class SigmaEmbedder(nn.Module):
    """Time σ -> conditioning vector [B, cond_dim].

    Sinusoidal frequency embedding (fp32, max_period=1e4) -> Linear -> SiLU -> Linear.
    σ is not rescaled (as in the S-FLM TimestepEmbedder); callers pass
    alpha_to_sigma(α).
    """

    def __init__(self, cond_dim: int, freq_dim: int) -> None:
        super().__init__()
        if freq_dim % 2 != 0:
            raise ValueError(f"freq_dim must be even, got {freq_dim}")
        self.freq_dim = freq_dim
        self.w1 = nn.Linear(freq_dim, cond_dim)
        self.w2 = nn.Linear(cond_dim, cond_dim)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=sigma.device)
            / half
        )
        args = sigma.float().reshape(-1, 1) * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.w2(F.silu(self.w1(emb.to(self.w1.weight.dtype))))

    def init_weights(self, profile: str) -> None:
        if profile == PROFILE_SFLM:
            # the reference uses the nn.Linear default init (kaiming_uniform)
            for linear in (self.w1, self.w2):
                linear.reset_parameters()
            return
        for linear in (self.w1, self.w2):
            trunc_normal_(linear.weight, mean=0.0, std=0.02)
            nn.init.zeros_(linear.bias)


def precompute_rope_cos_sin(
    head_dim: int, max_seq_len: int, base: float
) -> torch.Tensor:
    """cos/sin table for rotate-half (NeoX-style) RoPE (sflm profile).

    Returns: [max_seq_len, head_dim // 2, 2] in fp32; the last axis is (cos, sin).
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
    return torch.stack((freqs.cos(), freqs.sin()), dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_half(
    xq: torch.Tensor, xk: torch.Tensor, cos_sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotate-half RoPE to q and k (sflm profile).

    llama3's complex version pairs dims (2i, 2i+1); this one pairs
    (i, i + head_dim/2). They differ by a permutation of head_dim, so they are
    equally expressive but give different values for the same weights.

    Args:
        xq, xk: [B, L, H, head_dim]
        cos_sin: [max_seq_len, head_dim // 2, 2]; the first L rows are used
            (same slicing as llama3's reshape_for_broadcast)

    Computed in fp32 and cast back to the input dtype.
    """
    seqlen = xq.shape[1]
    cos_sin = cos_sin[0:seqlen]
    cos, sin = cos_sin[..., 0], cos_sin[..., 1]
    # [L, hd/2] -> [1, L, 1, hd]
    cos = torch.cat((cos, cos), dim=-1)[None, :, None, :]
    sin = torch.cat((sin, sin), dim=-1)[None, :, None, :]

    def _apply(x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * cos + _rotate_half(xf) * sin).to(x.dtype)

    return _apply(xq), _apply(xk)


class GeluMLP(nn.Module):
    """Linear(d->4d, bias) -> GELU(tanh) -> Linear(4d->d, bias) (sflm profile)."""

    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, mlp_ratio * dim, bias=True)
        self.w2 = nn.Linear(mlp_ratio * dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))

    def init_weights(self, init_std: float) -> None:
        del init_std  # the sflm profile uses the nn.Linear default init, as the reference does
        for linear in (self.w1, self.w2):
            linear.reset_parameters()


class Attention(nn.Module):
    """Multi-head attention (MHA, SDPA). Bidirectional by default (is_causal=False).

    `causal=True` is used only by the AR models, so that AR and flow models share
    the same backbone code (QK-norm, RoPE, initialization) and differ only in the
    objective. The flow path (`SphereDiTBlock`) always uses the default; see
    `test_flow_attention_stays_bidirectional`.
    """

    def __init__(
        self, model_args: SwaFlowModelArgs, layer_id: int = 0, causal: bool = False
    ):
        super().__init__()
        self.causal = bool(causal)
        if model_args.attn_type != "sdpa":
            raise NotImplementedError(
                f"swaflow supports attn_type='sdpa' only, got {model_args.attn_type}"
            )
        self.layer_id = layer_id
        self.n_heads = model_args.n_heads
        self.head_dim = model_args.dim // model_args.n_heads
        self.profile = model_args.backbone_profile

        # The reference's fused qkv Linear (d->3d, no bias) maps to the row slices
        # [0:d]=q / [d:2d]=k / [2d:3d]=v (see sflm_weights.py).
        self.wq = nn.Linear(model_args.dim, model_args.dim, bias=False)
        self.wk = nn.Linear(model_args.dim, model_args.dim, bias=False)
        self.wv = nn.Linear(model_args.dim, model_args.dim, bias=False)
        self.wo = nn.Linear(model_args.dim, model_args.dim, bias=False)

        # QK-norm: per-head RMSNorm.
        if model_args.qk_norm:
            self.q_norm: nn.Module | None = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps
            )
            self.k_norm: nn.Module | None = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps
            )
        else:
            self.q_norm = self.k_norm = None

        self.inner_attention = ScaledDotProductAttentionWrapper()

    def init_weights(self, init_std: float):
        if self.profile == PROFILE_SFLM:
            # the reference uses the nn.Linear default init (kaiming_uniform, a=√5)
            for linear in (self.wq, self.wk, self.wv, self.wo):
                linear.reset_parameters()
        else:
            for linear in (self.wq, self.wk, self.wv):
                trunc_normal_(linear.weight, mean=0.0, std=0.02)
            trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # -1 infers the local head count under TP sharding (as in llama3)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        if self.profile == PROFILE_SFLM:
            xq, xk = apply_rotary_emb_half(xq, xk, freqs_cis)
        else:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # q/k statistics for monitoring QK-norm. Recorded only every `every` steps;
        # a single attribute lookup when disabled (every=0).
        stats = get_attention_stats()
        if stats is not None and stats.active:
            stats.record(self.layer_id, xq, xk)

        # Flow models are bidirectional; only the AR models are causal.
        output = (
            self.inner_attention(xq, xk, xv, is_causal=self.causal)
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU FFN (same as llama3)."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            trunc_normal_(linear.weight, mean=0.0, std=init_std)


class SphereDiTBlock(nn.Module):
    """Transformer block with adaLN(-Zero) conditioning.

    Norms have no affine (adaLN supplies scale and shift). The adaLN Linear is
    zero-initialized, so each block starts with gate=0, i.e. as the identity
    (DiT's adaLN-Zero).
    """

    def __init__(self, layer_id: int, model_args: SwaFlowModelArgs):
        super().__init__()
        self.profile = model_args.backbone_profile
        self.attention = Attention(model_args, layer_id)
        if self.profile == PROFILE_SFLM:
            self.feed_forward: nn.Module = GeluMLP(model_args.dim)  # D2
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim,
                hidden_dim=4 * model_args.dim,
                multiple_of=model_args.multiple_of,
                ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            )
        self.attention_norm = make_norm(model_args)  # D1
        self.ffn_norm = make_norm(model_args)
        self.adaLN_modulation = nn.Linear(
            model_args.cond_dim, 6 * model_args.dim, bias=True
        )
        # dropout before the gate, as in the reference bias_dropout_add_scale
        self.dropout = nn.Dropout(model_args.dropout)

        # depth-scaled init (same scheme as llama3)
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """x: [B, L, d]; cond: [B, cond_dim], the raw condition (before SiLU)."""
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaLN_modulation(
            F.silu(cond)
        ).chunk(6, dim=-1)
        h = x + gate_a.unsqueeze(1) * self.dropout(
            self.attention(
                modulate(self.attention_norm(x), shift_a, scale_a), freqs_cis
            )
        )
        out = h + gate_f.unsqueeze(1) * self.dropout(
            self.feed_forward(modulate(self.ffn_norm(h), shift_f, scale_f))
        )
        return out

    def init_weights(self):
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)
        for norm in (self.attention_norm, self.ffn_norm):
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        # adaLN-Zero (both profiles; the reference also zero-initializes)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)


class SphereDiT(ModelProtocol):
    """Sphere-DiT. forward takes continuous vectors on the sphere and α.

    Attribute names (tok_embeddings / layers / norm / output) match llama3 so the
    upstream parallelize (FSDP etc.) is reused unchanged. self.output is the
    lm_head (untied from the embedding).
    """

    def __init__(self, model_args: SwaFlowModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.profile = model_args.backbone_profile

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.sigma_map = SigmaEmbedder(model_args.cond_dim, model_args.sigma_freq_dim)

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = SphereDiTBlock(layer_id, model_args)

        self.norm = make_norm(model_args)
        # final adaLN (shift/scale only, no gate)
        self.final_modulation = nn.Linear(
            model_args.cond_dim, 2 * model_args.dim, bias=True
        )
        # lm_head, untied from the embedding. The sflm profile has a bias, as in
        # the reference DDiTFinalLayer.linear.
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            bias=(self.profile == PROFILE_SFLM),
        )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        """RoPE table. The buffer name matches llama3 (to reuse parallelize).

        production: complex [L, hd/2] (same as llama3)
        sflm: real [L, hd/2, 2] (cos, sin), rotate-half with base 1e4
        """
        head_dim = self.model_args.dim // self.model_args.n_heads
        if self.profile == PROFILE_SFLM:
            return precompute_rope_cos_sin(
                head_dim, self.model_args.max_seq_len, 10000.0
            )
        return precompute_freqs_cis(
            head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        if self.profile == PROFILE_SFLM:
            # reference init ("ngpt"): N(0, 1/√d)
            nn.init.normal_(
                self.tok_embeddings.weight, std=self.model_args.dim**-0.5
            )
        else:
            nn.init.normal_(self.tok_embeddings.weight)
        self.sigma_map.init_weights(self.profile)
        for layer in self.layers.values():
            layer.init_weights()
        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()
        # final adaLN is zero-initialized too (identity modulation); both profiles
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        if self.profile == PROFILE_SFLM:
            # the reference DDiTFinalLayer.linear zero-initializes weight and bias
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            final_out_std = self.model_args.dim**-0.5
            cutoff_factor = 3
            trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def get_sphere_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token ids [B, L] -> embeddings on the unit sphere [B, L, d]."""
        return sphere_normalize(self.tok_embeddings(token_ids))

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        raise NotImplementedError(
            "swaflow uses bidirectional SDPA; no attention masks are needed"
        )

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: continuous vectors on the sphere [B, L, d]
            alpha: cleanness α ∈ [0, 1] (α=1 clean)

        Returns:
            vocabulary logits [B, L, V]

        The blocks are conditioned on σ = −log(1 − α); the conversion is kept here
        so the API stays in α.
        """
        cond = self.sigma_map(alpha_to_sigma(alpha))
        h = x
        for layer in self.layers.values():
            h = layer(h, cond, self.freqs_cis)
        h = self.norm(h)
        shift, scale = self.final_modulation(F.silu(cond)).chunk(2, dim=-1)
        h = modulate(h, shift, scale)
        return self.output(h)


class SphereDiTForPretrain(SphereDiT):
    """Wrapper for the torchtitan train loop.

    forward(tokens): embed -> noise by SLERP -> predict the clean tokens. The
    dataloader returns next-token-shifted labels; the loss
    (swaflow_denoising_loss) realigns them.

    t ~ U[0, 1] per sample and α = schedule.alpha(t). The schedule is LogLinear,
    truncated to [0, α_max] when schedule_alpha_max is set.

    Returns (logits, alpha). The trainer passes pred straight to loss_fn, which is
    how α reaches the per-sample (α, loss) logging in build_swaflow_denoising_loss.
    """

    def __init__(self, model_args: SwaFlowModelArgs):
        super().__init__(model_args)
        base: NoiseSchedule = LogLinear()
        if model_args.schedule_alpha_max is not None:
            self.schedule: NoiseSchedule = TruncatedScheduleWrapper(
                base, alpha_max=model_args.schedule_alpha_max
            )
        else:
            self.schedule = base
        self.antithetic_t_sampling = model_args.antithetic_t_sampling
        self.t_sampling_eps = model_args.t_sampling_eps

    def sample_t(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n training times t (t=0 is the clean side).

        antithetic: a grid stratified within the rank,
        u = (rand(n)/n + arange(n)/n) mod 1. The reference stratifies over the
        global batch; only the granularity differs (same distribution).

        t_sampling_eps: t = (1 − eps)·u. The reference uses t_ref = (1 − eps)·u + eps,
        which avoids the noise end. With t = 1 − t_ref this is t = (1 − eps)(1 − u),
        equal in distribution to (1 − eps)·u. The effect is to keep α away from 0
        (pure noise), not from the clean end.
        """
        u = torch.rand(n, device=device, dtype=torch.float32)
        if self.antithetic_t_sampling:
            offset = torch.arange(n, device=device, dtype=torch.float32) / n
            u = (u / n + offset) % 1.0
        return (1.0 - self.t_sampling_eps) * u

    def forward(
        self,
        tokens: torch.Tensor,
        alpha: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        clean_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: token ids [B, L]
            alpha: cleanness [B]. None samples t as in training; evaluation passes
                a fixed α grid here.
            noise: [B, L, d] source of the sphere noise; None uses randn.
                Evaluation passes deterministic noise here.
            clean_mask: [B, L]. Positions with 1 keep the clean embedding (no
                noise), i.e. they are fixed as a condition. Used for SFT and
                clean-prefix mid-training; it mirrors prefix conditioning at
                inference (`sampling._project_prefix`). The adaLN condition still
                uses a single α for the whole sequence, as in generation. None
                (default) is identical to pretraining.

        Returns:
            (logits for the clean tokens [B, L, V], alpha [B])
        """
        x1 = self.get_sphere_embeddings(tokens)  # clean embeddings
        if alpha is None:
            t = self.sample_t(tokens.shape[0], tokens.device)
            alpha = self.schedule.alpha(t)
        else:
            alpha = alpha.to(device=tokens.device)
        # uniform noise on the sphere
        x0 = sphere_normalize(noise.to(x1.dtype) if noise is not None else torch.randn_like(x1))
        # slerp(noise, clean; α): α=1 is clean. Computed in fp64.
        x_t = slerp(x0, x1, alpha.to(x1.dtype).view(-1, 1, 1))
        if clean_mask is not None:
            x_t = torch.where(
                clean_mask.to(device=x_t.device).bool().unsqueeze(-1), x1, x_t
            )
        return SphereDiT.forward(self, x_t, alpha), alpha
