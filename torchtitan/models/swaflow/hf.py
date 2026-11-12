# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Save and load SwaFlow weights in a Hugging Face Hub layout.

A model directory (or Hub repository) holds `config.json` (the model arguments)
and `model.safetensors` (fp32 weights). The RoPE table is not stored; it is
rebuilt from the arguments.

    from torchtitan.models.swaflow.hf import from_pretrained
    model = from_pretrained("tokyotech-llm/SwaFlow-150M", device="cuda")
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
#: Buffers rebuilt from the model arguments rather than stored.
RECOMPUTED_BUFFERS = ("freqs_cis",)

__all__ = ["CONFIG_NAME", "WEIGHTS_NAME", "from_pretrained", "save_pretrained"]


def _is_recomputed(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in RECOMPUTED_BUFFERS


def save_pretrained(
    model: torch.nn.Module,
    out_dir: str | Path,
    *,
    kind: str,
    flavor: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `config.json` and `model.safetensors` for a flow ("flow") or AR ("ar") model."""
    from safetensors.torch import save_file

    if kind not in ("flow", "ar"):
        raise ValueError(f"kind must be 'flow' or 'ar' (got {kind!r})")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().to("cpu", torch.float32).contiguous()
        for k, v in model.state_dict().items()
        if not _is_recomputed(k)
    }
    save_file(state, str(out / WEIGHTS_NAME), metadata={"format": "pt"})
    config = {
        "model_type": "swaflow" if kind == "flow" else "swaflow_ar",
        "kind": kind,
        "flavor": flavor,
        "model_args": {
            k: v for k, v in dataclasses.asdict(model.model_args).items() if not k.startswith("_")
        },
        "torch_dtype": "float32",
        **(extra or {}),
    }
    (out / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    return out


def _resolve(path_or_repo: str | Path, revision: str | None) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(path_or_repo), revision=revision, allow_patterns=[CONFIG_NAME, WEIGHTS_NAME]
        )
    )


def from_pretrained(
    path_or_repo: str | Path,
    *,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    """Build the model described by `config.json` and load its weights.

    Returns `SphereDiTForPretrain` for a flow model (it carries the noise
    schedule used by the sampler) or `ARTransformer` for an AR twin, in eval
    mode. Only parameters are cast to `dtype`; the RoPE buffers keep their dtype.
    """
    from safetensors.torch import load_file

    root = _resolve(path_or_repo, revision)
    config = json.loads((root / CONFIG_NAME).read_text())
    if config["kind"] == "flow":
        from torchtitan.models.swaflow.model import SphereDiTForPretrain, SwaFlowModelArgs

        args = SwaFlowModelArgs(**config["model_args"])
        build = SphereDiTForPretrain
    elif config["kind"] == "ar":
        from torchtitan.models.swaflow.ar.model import ARModelArgs, ARTransformer

        args = ARModelArgs(**config["model_args"])
        build = ARTransformer
    else:
        raise ValueError(f"unknown model kind {config['kind']!r} in {root / CONFIG_NAME}")

    with torch.device(device):
        model = build(args)
    state = load_file(str(root / WEIGHTS_NAME), device=str(device))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not _is_recomputed(k)]
    if missing or unexpected:
        raise RuntimeError(f"weights do not match the model: missing {missing}, unexpected {unexpected}")
    for p in model.parameters():
        p.data = p.data.to(dtype)
    return model.eval()
