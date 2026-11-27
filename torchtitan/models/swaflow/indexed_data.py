# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Dataloader that feeds the uint16 indexed dataset to the torchtitan training loop.

Data content, mixture and order are defined by `scripts/data/` (`plan.json`);
this module only consumes them, so the order has a single definition.

Resuming is exact. The slot convention is

    slot = sample_index * dp_world_size + dp_rank

where `sample_index` is the rank-local sample counter. Over all ranks and
samples this covers the slot space exactly once, so the only state is one
integer (`sample_index`) and a resumed run has no duplicates or gaps. Unlike
`pretokenized_data.py`, which relies on HF datasets' `skip()`, nothing depends
on reproducing a shuffle or a shard split. The same property lets a checkpoint
be resumed with a different number of ranks (`data_reshard.py`).

Labels follow the convention of `pretokenized_data.py`
(`label[j] = block[j+1]`, last position IGNORE_INDEX); see its docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

# scripts/ is not an installed package (pyproject only includes torchtitan*),
# so add the repository root to sys.path.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.data.indexed_dataset import Shard  # noqa: E402
from scripts.data.order_log import write_run_order  # noqa: E402
from scripts.data.plan import BuildPlan  # noqa: E402

__all__ = ["IndexedMixtureDataset", "build_indexed_dataloader", "global_batch_size"]

#: Slots resolved per call. Feistel and searchsorted are vectorised, so batching
#: is faster. Does not affect the resume granularity.
_RESOLVE_CHUNK = 1024


class IndexedMixtureDataset(IterableDataset, Stateful):
    """Yield blocks from the indexed dataset in plan.json order."""

    def __init__(
        self,
        build_dir: str,
        seq_len: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = True,
    ) -> None:
        self.plan = BuildPlan.load(build_dir)
        if self.plan.seq_len != seq_len:
            raise ValueError(
                f"plan seq_len {self.plan.seq_len} does not match training.seq_len {seq_len}; "
                "block boundaries would change, so the data is not repacked"
            )
        self.build_dir = Path(build_dir)
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.infinite = infinite
        self._sample_index = 0
        self._shards: dict[tuple[int, int], Shard] = {}
        # Drop the remainder so that every rank has the same number of samples.
        # At most dp_world_size - 1 slots are lost; ranks with different step
        # counts would deadlock.
        self._n_samples = self.plan.total_slots // dp_world_size

        logger.info(
            f"Indexed dataset '{self.plan.build_name}': "
            f"{self.plan.total_slots:,} slots × {seq_len} = "
            f"{self.plan.total_tokens / 1e9:.3f}B tokens\n" + self.plan.summary()
        )

    # ------------------------------------------------------------------
    def _shard(self, source_index: int, shard_index: int) -> Shard:
        key = (source_index, shard_index)
        if key not in self._shards:
            ref = self.plan.sources[source_index].shards[shard_index]
            self._shards[key] = Shard(self.build_dir / ref.prefix)
        return self._shards[key]

    def _block_tokens(self, resolved: dict, k: int) -> np.ndarray:
        shard = self._shard(int(resolved["source_index"][k]), int(resolved["shard_index"][k]))
        return shard.read_block(int(resolved["block_in_shard"][k]), self.seq_len)

    def _log_position(self, what: str) -> None:
        """Log where iteration starts.

        The slot follows from `sample_index` (see the module docstring), so this
        line shows directly that a resumed run continues without duplicates or
        gaps.
        """
        logger.info(
            f"[indexed-data] {what}: rank {self.dp_rank}/{self.dp_world_size} "
            f"sample_index={self._sample_index} "
            f"next_slot={self._sample_index * self.dp_world_size + self.dp_rank} "
            f"({self._sample_index / max(self._n_samples, 1):.4%} of plan)"
        )

    def __iter__(self):
        self._log_position("iteration start")
        while True:
            if self._sample_index >= self._n_samples:
                if not self.infinite:
                    logger.warning(
                        f"Indexed dataset '{self.plan.build_name}' exhausted "
                        f"({self.plan.total_slots:,} slots)"
                    )
                    return
                # Running past the end of the plan: the order repeats from the
                # start (target_tokens in the build config should cover the run).
                logger.warning(
                    f"Indexed dataset '{self.plan.build_name}' wrapped around: "
                    f"the run exceeded the plan's {self.plan.total_slots:,} slots; "
                    "replaying the same order (check target_tokens)"
                )
                self._sample_index = 0

            start = self._sample_index
            end = min(start + _RESOLVE_CHUNK, self._n_samples)
            slots = (
                np.arange(start, end, dtype=np.int64) * self.dp_world_size
                + self.dp_rank
            )
            resolved = self.plan.resolve(slots)
            for k in range(slots.size):
                block = torch.from_numpy(
                    self._block_tokens(resolved, k).astype(np.int64)
                )
                label = torch.empty_like(block)
                label[:-1] = block[1:]
                label[-1] = IGNORE_INDEX  # unused by _denoising_ce
                self._sample_index += 1
                yield {"input": block}, label

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self._sample_index,
            "build_name": self.plan.build_name,
            "plan_checksum": self.plan.checksum(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        name = state_dict.get("build_name")
        if name and name != self.plan.build_name:
            raise ValueError(
                f"checkpoint data build '{name}' differs from the current "
                f"'{self.plan.build_name}'; the data order would not continue"
            )
        checksum = state_dict.get("plan_checksum")
        if checksum and checksum != self.plan.checksum():
            raise ValueError(
                "the plan order at checkpoint time does not match the current plan "
                "(plan.json may have been rebuilt)"
            )
        self._sample_index = int(state_dict["sample_index"])
        self._log_position("resumed from checkpoint")


def global_batch_size(job_config: JobConfig, dp_world_size: int) -> int:
    """Sequences per optimizer step, gradient accumulation included.

    Same rule as `Trainer.__init__`: `training.global_batch_size`, or one
    microbatch per rank if it is not set. The order log needs this value, not
    `local_batch_size * dp_world_size`, which is smaller when gradients are
    accumulated.
    """
    gb = job_config.training.global_batch_size
    return gb if gb > 0 else job_config.training.local_batch_size * dp_world_size


def build_indexed_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer | None,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Dataloader for `training.dataset = "indexed"`.

    `training.dataset_path` is the build directory (containing plan.json). The
    tokenizer is unused (the data is already tokenized).
    """
    del tokenizer
    build_dir = job_config.training.dataset_path
    if not build_dir:
        raise ValueError(
            "indexed dataset requires training.dataset_path (build directory)"
        )

    # Worker sharding of the IterableDataset is not implemented; fail instead of
    # silently duplicating data (as in pretokenized_data.py).
    n_workers = getattr(job_config.training.dataloader, "num_workers", 0)
    if n_workers:
        raise ValueError(
            f"indexed dataloader supports only num_workers=0 (got {n_workers})"
        )

    ds = IndexedMixtureDataset(
        build_dir=build_dir,
        seq_len=job_config.training.seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    local_bs = job_config.training.local_batch_size
    if dp_rank == 0:
        # Run-side order log. The data seen by checkpoint N can be recovered from
        # this file and plan.json with scripts/data/order_log.py.
        path = write_run_order(
            dump_folder=job_config.job.dump_folder,
            build_dir=build_dir,
            plan=ds.plan,
            global_batch_size=global_batch_size(job_config, dp_world_size),
            dp_world_size=dp_world_size,
            extra={"local_batch_size": local_bs},
        )
        logger.info(f"Wrote data order log: {path}")

    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=local_bs,
    )
