# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Save and load SwaFlow weights in a Hugging Face Hub layout.

A model directory (or Hub repository) holds `config.json` (the model arguments)
and either `model.safetensors` (fp32 weights) or training states in folders,
one torchtitan DCP checkpoint per folder (e.g. `uniform/step-238400/`). The RoPE
table is not stored; it is rebuilt from the arguments.

    from torchtitan.models.swaflow.hf import from_pretrained
    model = from_pretrained("tokyotech-llm/SwaFlow-3B-EMA", device="cuda")
    model = from_pretrained("tokyotech-llm/SwaFlow-150M", subfolder="uniform/step-238400")
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
DCP_METADATA = ".metadata"
#: Buffers rebuilt from the model arguments rather than stored.
RECOMPUTED_BUFFERS = ("freqs_cis",)

__all__ = ["CONFIG_NAME", "WEIGHTS_NAME", "from_pretrained", "save_pretrained", "write_config"]


def _is_recomputed(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in RECOMPUTED_BUFFERS


def write_config(
    model_args: Any,
    out_dir: str | Path,
    *,
    kind: str,
    flavor: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `config.json` describing a flow ("flow") or AR ("ar") model."""
    if kind not in ("flow", "ar"):
        raise ValueError(f"kind must be 'flow' or 'ar' (got {kind!r})")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "swaflow" if kind == "flow" else "swaflow_ar",
        "kind": kind,
        "flavor": flavor,
        "model_args": {
            k: v for k, v in dataclasses.asdict(model_args).items() if not k.startswith("_")
        },
        "torch_dtype": "float32",
        **(extra or {}),
    }
    (out / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    return out


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

    out = write_config(model.model_args, out_dir, kind=kind, flavor=flavor, extra=extra)
    state = {
        k: v.detach().to("cpu", torch.float32).contiguous()
        for k, v in model.state_dict().items()
        if not _is_recomputed(k)
    }
    save_file(state, str(out / WEIGHTS_NAME), metadata={"format": "pt"})
    return out


def _resolve(path_or_repo: str | Path, revision: str | None, subfolder: str | None) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    patterns = [CONFIG_NAME] + ([f"{subfolder.strip('/')}/*"] if subfolder else [WEIGHTS_NAME])
    return Path(snapshot_download(str(path_or_repo), revision=revision, allow_patterns=patterns))


def _load_dcp_weights(model: torch.nn.Module, ckpt_dir: Path) -> None:
    """Copy the model tensors of a DCP training state into `model` (optimizer state is skipped)."""
    import torch.distributed.checkpoint as dcp

    have = set(dcp.FileSystemReader(str(ckpt_dir)).read_metadata().state_dict_metadata)
    sd = model.state_dict()
    missing = [k for k in sd if k not in have and not _is_recomputed(k)]
    if missing:
        raise RuntimeError(f"{ckpt_dir} lacks model tensors: {missing[:5]}")
    keys = [k for k in sd if k in have]
    staging = {k: torch.empty(sd[k].shape, dtype=sd[k].dtype, device="cpu") for k in keys}
    dcp.load(staging, checkpoint_id=str(ckpt_dir), no_dist=True)
    for k in keys:
        sd[k].copy_(staging[k])


def from_pretrained(
    path_or_repo: str | Path,
    *,
    subfolder: str | None = None,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    """Build the model described by `config.json` and load its weights.

    With `subfolder`, the weights are read from the training state in that folder
    (a DCP checkpoint; only that folder is downloaded). Otherwise they are read
    from `model.safetensors`. Returns `SphereDiTForPretrain` for a flow model (it
    carries the noise schedule used by the sampler) or `ARTransformer` for an AR
    twin, in eval mode. Only parameters are cast to `dtype`; the RoPE buffers
    keep their dtype.
    """
    root = _resolve(path_or_repo, revision, subfolder)
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

    weights_dir = root / subfolder.strip("/") if subfolder else root
    if (weights_dir / DCP_METADATA).is_file():
        with torch.device("cpu"):
            model = build(args)
        _load_dcp_weights(model, weights_dir)
        model = model.to(device)
    else:
        from safetensors.torch import load_file

        with torch.device(device):
            model = build(args)
        state = load_file(str(weights_dir / WEIGHTS_NAME), device=str(device))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not _is_recomputed(k)]
        if missing or unexpected:
            raise RuntimeError(
                f"weights do not match the model: missing {missing}, unexpected {unexpected}"
            )
    for p in model.parameters():
        p.data = p.data.to(dtype)
    return model.eval()
