# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Dataloader for clean-prefix mid-training.

Streams the same indexed data in the same order as pretraining and draws a
split point k for each sample:

    positions 0 .. k-1 : clean (`clean_mask = 1`, no noise)
    positions k .. L-1 : noised with α as usual; the loss covers only these

With probability `clean_prefix_prob`, k ~ U{0, ..., k_max-1}; otherwise k = 0.
A sample with k = 0 is identical to pretraining (an all-zero `clean_mask`
leaves `x_t` unchanged and masks no labels). The mid-training branches resume
from a pretraining checkpoint with the same data order.

k is a pure function of the slot index rather than drawn from an RNG:

1. The default torch generator is not consumed. It drives only the α samples
   and the spherical noise (see `rng.py`), so both stay identical to a
   uniform-α continuation and only the clean prefix differs.
2. Resuming needs no extra state.
3. The distribution and its reproducibility can be tested on one device.

`clean_mask` is consumed by the model (`SphereDiTForPretrain.forward`), the
same input used for SFT.
"""

from __future__ import annotations

from typing import Any

import torch

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.models.swaflow.indexed_data import IndexedMixtureDataset
from torchtitan.tools.logging import logger

from scripts.data.order_log import write_run_order  # noqa: E402

__all__ = [
    "NonUniformAlphaDataset",
    "build_midtrain_dataloader",
    "draw_split_point",
]

_U64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """SplitMix64: an integer hash that mixes well even for sequential inputs."""
    z = (x + 0x9E3779B97F4A7C15) & _U64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    return (z ^ (z >> 31)) & _U64


def _u01(seed: int, counter: int) -> float:
    """Uniform value in [0, 1) as a pure function of (seed, counter)."""
    return _splitmix64((seed & _U64) ^ _splitmix64(counter & _U64)) / 2.0**64


def draw_split_point(
    slot: int, seed: int, clean_prefix_prob: float, k_max: int
) -> int:
    """Split point k for a slot (pure function).

    With probability `clean_prefix_prob`, k ~ U{0, ..., k_max-1}; otherwise
    k = 0 (identical to uniform α).

    k is random so that no positional rule is learned: with a fixed length the
    model would learn where the noise starts, which does not transfer to real
    prompts (MCQ, SFT) of varying length.
    """
    if _u01(seed, 2 * slot) >= clean_prefix_prob:
        return 0
    return int(_u01(seed, 2 * slot + 1) * k_max)


class NonUniformAlphaDataset(IndexedMixtureDataset):
    """Indexed data with a clean prefix per sample.

    Order, slot convention and `state_dict` are inherited, so resuming from a
    pretraining checkpoint keeps `sample_index` and sees the same slots at the
    same steps (identical data order).
    """

    def __init__(
        self,
        *args,
        clean_prefix_prob: float = 0.5,
        k_max: int = 1024,
        k_seed: int = 20260903,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= clean_prefix_prob <= 1.0:
            raise ValueError(
                f"clean_prefix_prob must be in [0, 1] (got {clean_prefix_prob})"
            )
        if not 1 <= k_max <= self.seq_len:
            raise ValueError(
                f"k_max must be between 1 and seq_len ({self.seq_len}) (got {k_max})"
            )
        self.clean_prefix_prob = float(clean_prefix_prob)
        self.k_max = int(k_max)
        self.k_seed = int(k_seed)
        logger.info(
            "[midtrain] clean prefix: with probability "
            f"{self.clean_prefix_prob:.3f} draw k ~ U(0, {self.k_max}) "
            f"(seed {self.k_seed}); expected clean fraction "
            f"{self.clean_prefix_prob * (self.k_max - 1) / 2 / self.seq_len:.4f}"
        )

    def __iter__(self):
        for input_dict, label in super().__iter__():
            # The parent generator advances `_sample_index` before yielding, so
            # the sample just yielded belongs to the previous index.
            slot = (self._sample_index - 1) * self.dp_world_size + self.dp_rank
            k = draw_split_point(
                slot, self.k_seed, self.clean_prefix_prob, self.k_max
            )
            block = input_dict["input"]
            clean = torch.zeros_like(block, dtype=torch.uint8)
            if k > 0:
                clean[:k] = 1
                # label[j] is the token at position j+1 (`indexed_data`
                # convention) and the loss maps back with
                # `pred[:, i] <-> labels[:, i-1]`, so dropping position i means
                # dropping label[i-1]. Positions i < k are dropped: label[:k-1]
                # (as in `sft_data`).
                label[: k - 1] = IGNORE_INDEX
            yield {"input": block, "clean_mask": clean}, label

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Parent state plus the k distribution, so that it cannot change silently.

        Only `sample_index` is needed to resume (k is a pure function of the
        slot); the extra fields are only checked.
        """
        state = super().state_dict()
        state["midtrain"] = {
            "clean_prefix_prob": self.clean_prefix_prob,
            "k_max": self.k_max,
            "k_seed": self.k_seed,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Branching from a pretraining checkpoint is the normal use, so a
        # missing `midtrain` entry is fine. Fail only if it is present and
        # differs, so the k distribution never changes within a run.
        saved = state_dict.get("midtrain")
        if saved is not None:
            now = {
                "clean_prefix_prob": self.clean_prefix_prob,
                "k_max": self.k_max,
                "k_seed": self.k_seed,
            }
            if saved != now:
                raise ValueError(
                    f"clean-prefix settings differ: checkpoint ({saved}) vs current ({now})"
                )
        super().load_state_dict(state_dict)


def build_midtrain_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer | None,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Dataloader for `training.dataset = "indexed_midtrain"`.

    `dataset_path` is the same build directory as in pretraining. The k
    distribution is set in `[swaflow.midtrain]` (`job_config.py`).
    """
    del tokenizer
    build_dir = job_config.training.dataset_path
    if not build_dir:
        raise ValueError(
            "indexed_midtrain dataset requires training.dataset_path"
        )
    n_workers = getattr(job_config.training.dataloader, "num_workers", 0)
    if n_workers:
        raise ValueError(
            f"indexed_midtrain dataloader supports only num_workers=0 (got {n_workers})"
        )
    cfg = getattr(getattr(job_config, "swaflow", None), "midtrain", None)
    if cfg is None:
        raise ValueError("[swaflow.midtrain] is missing from the config")

    ds = NonUniformAlphaDataset(
        build_dir=build_dir,
        seq_len=job_config.training.seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        clean_prefix_prob=cfg.clean_prefix_prob,
        k_max=cfg.k_max,
        k_seed=cfg.seed,
    )

    local_bs = job_config.training.local_batch_size
    if dp_rank == 0:
        # Same order-log format as pretraining: matching `plan_checksum` and
        # `global_batch_size` in the `run_order.json` of both runs show that the
        # data order is identical.
        path = write_run_order(
            dump_folder=job_config.job.dump_folder,
            build_dir=build_dir,
            plan=ds.plan,
            global_batch_size=local_bs * dp_world_size,
            dp_world_size=dp_world_size,
            extra={
                "local_batch_size": local_bs,
                "midtrain": {
                    "clean_prefix_prob": ds.clean_prefix_prob,
                    "k_max": ds.k_max,
                    "k_seed": ds.k_seed,
                },
            },
        )
        logger.info(f"Wrote data order log: {path}")

    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=local_bs,
    )
