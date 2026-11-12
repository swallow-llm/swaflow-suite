# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.hf_datasets.text_datasets import build_text_dataloader

# Reuse the upstream FSDP/TP plan (module attribute names match llama3).
from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from torchtitan.models.swaflow.indexed_data import build_indexed_dataloader
from torchtitan.models.swaflow.loss import build_swaflow_denoising_loss
from torchtitan.models.swaflow.midtrain_data import build_midtrain_dataloader
from torchtitan.models.swaflow.model import (
    PROFILE_SFLM,
    PROFILE_PRODUCTION,
    SwaFlowModelArgs,
    SphereDiT,
    SphereDiTForPretrain,
    alpha_to_sigma,
)
from torchtitan.models.swaflow.pretokenized_data import build_pretokenized_dataloader
from torchtitan.models.swaflow.schedules import (
    CosineSquared,
    LogLinear,
    NoiseSchedule,
    TruncatedScheduleWrapper,
)
from torchtitan.models.swaflow.sft_data import build_sft_dataloader
from torchtitan.models.swaflow.sphere_ops import (
    exp_map,
    log_map,
    slerp,
    sphere_normalize,
)
from torchtitan.models.swaflow.validate import build_swaflow_validator
from torchtitan.protocols.train_spec import TrainSpec

__all__ = [
    "INDEXED_DATASET",
    "MIDTRAIN_DATASET",
    "PRETOKENIZED_DATASET",
    "SFT_AR_DATASET",
    "SFT_DATASET",
    "PROFILE_SFLM",
    "PROFILE_PRODUCTION",
    "CosineSquared",
    "LogLinear",
    "NoiseSchedule",
    "SwaFlowModelArgs",
    "SphereDiT",
    "SphereDiTForPretrain",
    "TruncatedScheduleWrapper",
    "alpha_to_sigma",
    "build_indexed_dataloader",
    "build_midtrain_dataloader",
    "build_pretokenized_dataloader",
    "build_sft_dataloader",
    "build_swaflow_validator",
    "exp_map",
    "get_train_spec",
    "log_map",
    "swaflow_args",
    "slerp",
    "sphere_normalize",
]


swaflow_args = {
    # 2-layer model for installation checks and unit tests.
    "debugmodel": SwaFlowModelArgs(
        dim=256,
        n_layers=2,
        n_heads=8,
        vocab_size=2048,
        cond_dim=128,
        sigma_freq_dim=128,
        rope_theta=500000,
    ),
    # Scaling ladder (production profile).
    "150M": SwaFlowModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        vocab_size=49152,
        cond_dim=256,
        rope_theta=500000,
    ),
    # head_dim is 128 for 500M/1B/3B (1280/10 = 2048/16 = 3072/24), so head shape
    # does not confound trends across model width. cond_dim (capacity of the time
    # conditioning) grows with width within 256-512.
    "500M": SwaFlowModelArgs(
        dim=1280,
        n_layers=24,
        n_heads=10,
        vocab_size=49152,
        cond_dim=384,
        sigma_freq_dim=384,
        rope_theta=500000,
    ),
    "1B": SwaFlowModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        vocab_size=49152,
        cond_dim=512,
        sigma_freq_dim=512,
        rope_theta=500000,
    ),
    # Width/depth/heads follow Llama-3.2-3B. The default FFN sizing with
    # multiple_of=256 gives 8192, matching Llama-3.2-3B.
    "3B": SwaFlowModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=49152,
        cond_dim=512,
        sigma_freq_dim=512,
        rope_theta=500000,
    ),
    # Matches the S-FLM reference config small-sphere-dit (hidden 768, cond_dim 128,
    # 12 blocks, 12 heads, dropout 0.1, length 1024) with the GPT-2 vocabulary (50257).
    "sflm_small": SwaFlowModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        vocab_size=50257,
        cond_dim=128,
        max_seq_len=1024,
        backbone_profile=PROFILE_SFLM,
        qk_norm=False,  # the reference implementation has no QK-norm
        dropout=0.1,
        antithetic_t_sampling=True,
        t_sampling_eps=1e-3,
        # The reference LogLinear eps=1e-3 acts at the clean end in our convention
        # (α=1 clean), so the equivalent is an upper cap on α.
        schedule_alpha_max=1.0 - 1e-3,
    ),
}


#: training.dataset name for pre-tokenized data; training.dataset_path is a .dat file.
PRETOKENIZED_DATASET = "pretokenized"

#: uint16 indexed dataset (pretraining data pipeline).
#: training.dataset_path is the build directory (containing plan.json).
INDEXED_DATASET = "indexed"

#: Packed SFT blocks (output of `scripts/data/build_sft.py`).
#: training.dataset_path is the build directory (containing meta.json).
SFT_DATASET = "sft"

#: The same SFT data for the AR models: identical to `sft` except that no
#: `clean_mask` is passed (an AR model always reads the prompt as clean). Label
#: masking is the same, so this is standard AR SFT with loss on the response only.
SFT_AR_DATASET = "sft_ar"

#: Clean-prefix mid-training (`midtrain_data.py`). Same build, order and slot
#: convention as `indexed`; each sample additionally gets a clean prefix
#: (`clean_mask` is passed and labels at those positions are dropped). A sample
#: with k = 0 is bit-identical to `indexed`. `dataset_path` is the pretraining build.
MIDTRAIN_DATASET = "indexed_midtrain"


def _build_swaflow_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer,
    job_config,
    infinite: bool = True,
):
    """Dispatch on training.dataset to our dataloaders or the upstream text dataloader."""
    if job_config.training.dataset == INDEXED_DATASET:
        return build_indexed_dataloader(
            dp_world_size, dp_rank, tokenizer, job_config, infinite
        )
    if job_config.training.dataset == MIDTRAIN_DATASET:
        return build_midtrain_dataloader(
            dp_world_size, dp_rank, tokenizer, job_config, infinite
        )
    if job_config.training.dataset == SFT_DATASET:
        return build_sft_dataloader(
            dp_world_size, dp_rank, tokenizer, job_config, infinite
        )
    if job_config.training.dataset == SFT_AR_DATASET:
        return build_sft_dataloader(
            dp_world_size, dp_rank, tokenizer, job_config, infinite,
            emit_clean_mask=False,
        )
    if job_config.training.dataset == PRETOKENIZED_DATASET:
        return build_pretokenized_dataloader(
            dp_world_size, dp_rank, tokenizer, job_config, infinite
        )
    return build_text_dataloader(
        dp_world_size, dp_rank, tokenizer, job_config, infinite
    )


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=SphereDiTForPretrain,
        model_args=swaflow_args,
        parallelize_fn=parallelize_llama,
        pipelining_fn=None,  # pipeline parallelism is not supported
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=_build_swaflow_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_swaflow_denoising_loss,
        # Held-out val loss and per-α loss profile; built only when
        # validation.enable = true.
        build_validator_fn=build_swaflow_validator,
    )
