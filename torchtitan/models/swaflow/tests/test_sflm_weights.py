# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Checks that the same weights can be loaded into both this implementation and
the reference implementation.

The reference model needs a GPU and flash_attn, so it is not instantiated here.
Instead a synthetic dict with the reference state_dict's keys and shapes checks
that the key mapping is a bijection.
"""

import pytest
import torch

from torchtitan.models.swaflow.model import PROFILE_SFLM, SwaFlowModelArgs, SphereDiT
from torchtitan.models.swaflow.sflm_weights import sflm_to_swaflow_state_dict

D, V, NL, NH, COND = 64, 128, 2, 4, 32


def titan_parity_model() -> SphereDiT:
    return SphereDiT(
        SwaFlowModelArgs(
            dim=D,
            n_layers=NL,
            n_heads=NH,
            vocab_size=V,
            cond_dim=COND,
            sigma_freq_dim=COND,
            max_seq_len=16,
            backbone_profile=PROFILE_SFLM,
        )
    )


def synthetic_sflm_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic dict with the keys and shapes of the reference SphereDiT (small-sphere-dit).

    Key names and shapes follow the structure of models/dit.py and
    models/sphere_dit.py in the reference implementation.
    """
    sd = {
        "sphere_embed.weight": torch.randn(V, D),
        "sigma_map.mlp.0.weight": torch.randn(COND, COND),
        "sigma_map.mlp.0.bias": torch.randn(COND),
        "sigma_map.mlp.2.weight": torch.randn(COND, COND),
        "sigma_map.mlp.2.bias": torch.randn(COND),
        "rotary_emb.inv_freq": torch.randn(D // NH // 2),  # must be ignored
        "output_layer.norm_final.weight": torch.randn(D),
        "output_layer.linear.weight": torch.randn(V, D),
        "output_layer.linear.bias": torch.randn(V),
        "output_layer.adaLN_modulation.weight": torch.randn(2 * D, COND),
        "output_layer.adaLN_modulation.bias": torch.randn(2 * D),
    }
    for i in range(NL):
        sd.update(
            {
                f"blocks.{i}.norm1.weight": torch.randn(D),
                f"blocks.{i}.norm2.weight": torch.randn(D),
                f"blocks.{i}.attn_qkv.weight": torch.randn(3 * D, D),
                f"blocks.{i}.attn_out.weight": torch.randn(D, D),
                f"blocks.{i}.mlp.0.weight": torch.randn(4 * D, D),
                f"blocks.{i}.mlp.0.bias": torch.randn(4 * D),
                f"blocks.{i}.mlp.2.weight": torch.randn(D, 4 * D),
                f"blocks.{i}.mlp.2.bias": torch.randn(D),
                f"blocks.{i}.adaLN_modulation.weight": torch.randn(6 * D, COND),
                f"blocks.{i}.adaLN_modulation.bias": torch.randn(6 * D),
            }
        )
    return sd


def test_mapping_covers_titan_parameters_exactly():
    """The converted keys match the parameter set of the sflm_small model exactly.

    Otherwise the "same weights" premise of the comparison does not hold.
    """
    model = titan_parity_model()
    converted = sflm_to_swaflow_state_dict(synthetic_sflm_state_dict(), dim=D)
    # freqs_cis is persistent=False and does not appear in the state_dict
    assert set(converted.keys()) == set(model.state_dict().keys())


def test_load_state_dict_strict_succeeds():
    """strict=True succeeds, i.e. nothing missing and nothing extra."""
    model = titan_parity_model()
    converted = sflm_to_swaflow_state_dict(synthetic_sflm_state_dict(), dim=D)
    model.load_state_dict(converted, strict=True)
    for key, want in converted.items():
        torch.testing.assert_close(model.state_dict()[key], want)


def test_qkv_row_slices_go_to_wq_wk_wv():
    sflm_sd = synthetic_sflm_state_dict()
    qkv = sflm_sd["blocks.0.attn_qkv.weight"]
    converted = sflm_to_swaflow_state_dict(sflm_sd, dim=D)
    torch.testing.assert_close(converted["layers.0.attention.wq.weight"], qkv[0:D])
    torch.testing.assert_close(converted["layers.0.attention.wk.weight"], qkv[D : 2 * D])
    torch.testing.assert_close(
        converted["layers.0.attention.wv.weight"], qkv[2 * D : 3 * D]
    )


def test_unmapped_key_raises():
    """Unknown keys are not silently dropped (that would corrupt the comparison)."""
    sflm_sd = synthetic_sflm_state_dict()
    sflm_sd["blocks.0.some_new_thing.weight"] = torch.randn(D)
    with pytest.raises(KeyError, match="some_new_thing"):
        sflm_to_swaflow_state_dict(sflm_sd, dim=D)


def test_wrong_qkv_shape_raises():
    sflm_sd = synthetic_sflm_state_dict()
    sflm_sd["blocks.0.attn_qkv.weight"] = torch.randn(2 * D, D)
    with pytest.raises(ValueError, match="attn_qkv"):
        sflm_to_swaflow_state_dict(sflm_sd, dim=D)


def test_production_profile_is_not_loadable():
    """Reference weights do not load into the production profile (structural differences)."""
    model = SphereDiT(
        SwaFlowModelArgs(
            dim=D,
            n_layers=NL,
            n_heads=NH,
            vocab_size=V,
            cond_dim=COND,
            sigma_freq_dim=COND,
            max_seq_len=16,
        )
    )
    converted = sflm_to_swaflow_state_dict(synthetic_sflm_state_dict(), dim=D)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert unexpected, "reference weights must not load as-is into the production profile"
