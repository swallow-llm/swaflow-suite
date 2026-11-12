# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Load S-FLM reference checkpoints into SwaFlow models.

Maps the state_dict of the S-FLM reference SphereDiT key-for-key onto our model
with backbone_profile="sflm", whose block components match the reference. Used
to evaluate the released S-FLM weights with our code.

The one non-trivial mapping is the fused qkv: the reference uses one
Linear(d -> 3d) split by einops into (three, h, d), so weight rows [0:d] / [d:2d] /
[2d:3d] are q / k / v; they are sliced into our separate wq / wk / wv.
"""

from __future__ import annotations

import re

import torch

__all__ = ["sflm_to_swaflow_state_dict", "SWAFLOW_ONLY_KEYS"]

# non-persistent buffers that exist only in our model (not loaded)
SWAFLOW_ONLY_KEYS = ("freqs_cis",)

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")

# reference block keys -> our block keys (1:1 except qkv)
_BLOCK_MAP = {
    "norm1.weight": "attention_norm.weight",
    "norm2.weight": "ffn_norm.weight",
    "attn_out.weight": "attention.wo.weight",
    "mlp.0.weight": "feed_forward.w1.weight",
    "mlp.0.bias": "feed_forward.w1.bias",
    "mlp.2.weight": "feed_forward.w2.weight",
    "mlp.2.bias": "feed_forward.w2.bias",
    "adaLN_modulation.weight": "adaLN_modulation.weight",
    "adaLN_modulation.bias": "adaLN_modulation.bias",
}

_TOP_MAP = {
    "sphere_embed.weight": "tok_embeddings.weight",
    "sigma_map.mlp.0.weight": "sigma_map.w1.weight",
    "sigma_map.mlp.0.bias": "sigma_map.w1.bias",
    "sigma_map.mlp.2.weight": "sigma_map.w2.weight",
    "sigma_map.mlp.2.bias": "sigma_map.w2.bias",
    "output_layer.norm_final.weight": "norm.weight",
    "output_layer.linear.weight": "output.weight",
    "output_layer.linear.bias": "output.bias",
    "output_layer.adaLN_modulation.weight": "final_modulation.weight",
    "output_layer.adaLN_modulation.bias": "final_modulation.bias",
}

# reference-only keys that do not affect outputs
_IGNORED_SFLM_KEYS = ("rotary_emb.inv_freq",)


def sflm_to_swaflow_state_dict(
    sflm_sd: dict[str, torch.Tensor], dim: int
) -> dict[str, torch.Tensor]:
    """Convert an S-FLM reference SphereDiT state_dict for our model (sflm profile).

    Args:
        sflm_sd: reference state_dict (keys such as `sphere_embed.weight`)
        dim: hidden size (row-slice width of the fused qkv)

    Returns:
        a dict that can be passed directly to SphereDiT.load_state_dict

    Raises:
        KeyError: if any reference key is left unmapped (never dropped silently)
    """
    out: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for key, value in sflm_sd.items():
        if key in _IGNORED_SFLM_KEYS:
            continue
        if key in _TOP_MAP:
            out[_TOP_MAP[key]] = value
            continue

        m = _BLOCK_RE.match(key)
        if m is None:
            unmapped.append(key)
            continue
        idx, sub = m.group(1), m.group(2)

        if sub == "attn_qkv.weight":
            if value.shape[0] != 3 * dim:
                raise ValueError(
                    f"{key}: expected first dim {3 * dim}, got {value.shape[0]}"
                )
            q, k, v = value[0:dim], value[dim : 2 * dim], value[2 * dim : 3 * dim]
            out[f"layers.{idx}.attention.wq.weight"] = q
            out[f"layers.{idx}.attention.wk.weight"] = k
            out[f"layers.{idx}.attention.wv.weight"] = v
            continue

        if sub in _BLOCK_MAP:
            out[f"layers.{idx}.{_BLOCK_MAP[sub]}"] = value
            continue

        unmapped.append(key)

    if unmapped:
        raise KeyError(
            "reference state_dict has unmapped keys: "
            + ", ".join(sorted(unmapped))
        )
    return out
