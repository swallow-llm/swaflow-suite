# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""TrainSpec for the autoregressive (AR) models.

The AR models (150M / 500M / 1B) are trained under the same conditions as the
flow models (same data, order, global batch and shapes) as an external reference
for scaling. The spec is registered as `swaflow_ar` with `register_train_spec`
rather than added to upstream `_supported_models`, so training must be launched
through `ar/train.py`; running `torchtitan.train` directly fails with
`TrainSpec swaflow_ar is not registered`.
"""

from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from torchtitan.models.swaflow import _build_swaflow_dataloader
from torchtitan.models.swaflow.ar.loss import build_ar_next_token_loss
from torchtitan.models.swaflow.ar.model import ARModelArgs, ARTransformer
from torchtitan.models.swaflow.ar.validate import build_ar_validator
from torchtitan.protocols.train_spec import (
    register_train_spec,
    TrainSpec,
    _extra_train_specs,
)

__all__ = [
    "AR_TRAIN_SPEC_NAME",
    "ARModelArgs",
    "ARTransformer",
    "ar_args",
    "get_ar_train_spec",
    "register",
]

#: Value of `[model] name`.
AR_TRAIN_SPEC_NAME = "swaflow_ar"


#: Shapes match the flow flavors (`swaflow_args`) one-to-one; parameter counts
#: differ only by the flow models' adaLN / sigma_map.
ar_args = {
    # 2-layer model for installation checks and unit tests, same shape as the flow `debugmodel`.
    "debugmodel": ARModelArgs(
        dim=256, n_layers=2, n_heads=8, vocab_size=2048, rope_theta=500000
    ),
    # 150M: d 768 / 12 layers / 12 heads / head_dim 64
    "150M": ARModelArgs(
        dim=768, n_layers=12, n_heads=12, vocab_size=49152, rope_theta=500000
    ),
    # 500M: d 1280 / 24 layers / 10 heads / head_dim 128
    "500M": ARModelArgs(
        dim=1280, n_layers=24, n_heads=10, vocab_size=49152, rope_theta=500000
    ),
    # 1B: d 2048 / 24 layers / 16 heads / head_dim 128
    "1B": ARModelArgs(
        dim=2048, n_layers=24, n_heads=16, vocab_size=49152, rope_theta=500000
    ),
}


def get_ar_train_spec() -> TrainSpec:
    """Differs from the flow `get_train_spec()` in three places only:

        model_cls        SphereDiTForPretrain -> ARTransformer
        build_loss_fn    denoising CE         -> next-token CE
        build_validator  α grid               -> plain held-out CE

    The dataloader (`indexed`), optimizer, LR schedule and parallelize are the
    same functions, keeping the training conditions identical.
    """
    return TrainSpec(
        model_cls=ARTransformer,
        model_args=ar_args,
        parallelize_fn=parallelize_llama,
        pipelining_fn=None,  # pipeline parallelism is not supported
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=_build_swaflow_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_ar_next_token_loss,
        build_validator_fn=build_ar_validator,
    )


def register() -> None:
    """Register `swaflow_ar` in the TrainSpec registry (idempotent).

    Called on import. Re-registration is skipped instead of raising, since tests
    may import this module repeatedly in one process.
    """
    if AR_TRAIN_SPEC_NAME not in _extra_train_specs:
        register_train_spec(AR_TRAIN_SPEC_NAME, get_ar_train_spec())


register()
