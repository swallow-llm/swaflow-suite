# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Round trip of the Hugging Face Hub layout (`hf.save_pretrained` / `hf.from_pretrained`)."""

import json

import pytest
import torch

from scripts.eval.loader import build_eval_model
from torchtitan.models.swaflow.hf import CONFIG_NAME, WEIGHTS_NAME, from_pretrained, save_pretrained


@pytest.mark.parametrize("kind", ["flow", "ar"])
def test_round_trip_restores_weights_and_outputs(tmp_path, kind):
    torch.manual_seed(0)
    model = build_eval_model(kind, "debugmodel", alpha_max=0.15)
    with torch.no_grad():
        model.init_weights()  # build_eval_model leaves the weights uninitialized
    save_pretrained(model, tmp_path, kind=kind, flavor="debugmodel")
    assert (tmp_path / CONFIG_NAME).is_file() and (tmp_path / WEIGHTS_NAME).is_file()

    loaded = from_pretrained(tmp_path)
    for (k, a), (k2, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k == k2 and torch.equal(a, b), k

    tokens = torch.randint(0, model.model_args.vocab_size, (2, 16))
    model.eval()
    with torch.no_grad():
        if kind == "flow":
            alpha = torch.full((2,), 0.05)
            outs = []
            for m in (model, loaded):  # forward draws the sphere noise: same seed for both
                torch.manual_seed(1)
                outs.append(m(tokens, alpha)[0])
            assert torch.equal(*outs)
        else:
            assert torch.equal(model(tokens), loaded(tokens))


def test_config_records_the_window_and_no_rope_table(tmp_path):
    model = build_eval_model("flow", "debugmodel", alpha_max=0.15)
    save_pretrained(model, tmp_path, kind="flow", flavor="debugmodel")
    config = json.loads((tmp_path / CONFIG_NAME).read_text())
    assert config["model_args"]["schedule_alpha_max"] == 0.15
    from safetensors.torch import load_file

    assert not any(k.endswith("freqs_cis") for k in load_file(str(tmp_path / WEIGHTS_NAME)))


@pytest.mark.parametrize("kind", ["flow", "ar"])
def test_weights_load_from_a_training_state_folder(tmp_path, kind):
    """A DCP training state (model + optimizer tensors) in a subfolder loads its model weights."""
    import torch.distributed.checkpoint as dcp

    from torchtitan.models.swaflow.hf import write_config

    torch.manual_seed(0)
    model = build_eval_model(kind, "debugmodel", alpha_max=0.15)
    with torch.no_grad():
        model.init_weights()
    state = dict(model.state_dict())
    state.update({f"optimizer.state.{k}.exp_avg": torch.randn(v.shape) for k, v in list(state.items())[:3]})
    dcp.save(state, checkpoint_id=str(tmp_path / "uniform" / "step-20"), no_dist=True)
    write_config(model.model_args, tmp_path, kind=kind, flavor="debugmodel")

    loaded = from_pretrained(tmp_path, subfolder="uniform/step-20")
    for (k, a), (k2, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k == k2 and torch.equal(a, b), k


def test_a_training_state_without_model_tensors_is_rejected(tmp_path):
    import torch.distributed.checkpoint as dcp

    from torchtitan.models.swaflow.hf import write_config

    model = build_eval_model("flow", "debugmodel", alpha_max=0.15)
    dcp.save({"optimizer.lr": torch.zeros(1)}, checkpoint_id=str(tmp_path / "step-1"), no_dist=True)
    write_config(model.model_args, tmp_path, kind="flow", flavor="debugmodel")
    with pytest.raises(RuntimeError, match="lacks model tensors"):
        from_pretrained(tmp_path, subfolder="step-1")
