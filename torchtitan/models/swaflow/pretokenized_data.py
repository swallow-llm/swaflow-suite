# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Dataloader for pre-tokenized fixed-length blocks.

Reads the `.dat` files produced by the S-FLM data pipeline
(`datasets.save_to_disk` format; each row is a fixed-length `input_ids` block
laid out as `[BOS] + (block_size - 2) tokens + [EOS]`). Training on the same
files as the S-FLM reference implementation gives token-identical data, free
of differences in tokenizer settings, wrapping or shuffling.

## Labels (alignment)

In torchtitan the dataloader returns next-token-shifted labels, and
`_denoising_ce` in `loss.py` undoes the shift with `pred[:, i] <-> labels[:, i-1]`.
Returning `label[j] = block[j+1]` therefore aligns `pred[:, i] <-> block[i]`
(predict the original token at position i). `_denoising_ce` only uses
`labels[:, :-1]`; the last position is set to IGNORE_INDEX to make that explicit.

Known difference: position 0 drops out of the loss (S-FLM scores all
positions), a 1/block_size effect.
"""

from typing import Any

import torch
from datasets import Dataset, load_from_disk
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

__all__ = ["PretokenizedBlockDataset", "build_pretokenized_dataloader"]


class PretokenizedBlockDataset(IterableDataset, Stateful):
    """Yield fixed-length blocks saved with `save_to_disk` as (input, label)."""

    def __init__(
        self,
        dataset_path: str,
        seq_len: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        ds = load_from_disk(dataset_path)
        if not isinstance(ds, Dataset):
            raise TypeError(
                f"{dataset_path} must be a single-split Dataset "
                f"(got {type(ds).__name__})"
            )
        block_len = len(ds[0]["input_ids"])
        if block_len != seq_len:
            raise ValueError(
                f"pre-tokenized block length {block_len} does not match "
                f"training.seq_len {seq_len}; "
                "the blocks are not repacked"
            )
        logger.info(
            f"Pretokenized dataset: {dataset_path} "
            f"({len(ds)} blocks × {block_len} tokens = "
            f"{len(ds) * block_len / 1e6:.1f}M tokens, shuffle={shuffle})"
        )

        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.infinite = infinite
        self.shuffle = shuffle
        self.seed = seed
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        # Shuffle the whole dataset before sharding (as DistributedSampler does);
        # shuffling after sharding would give each rank the same blocks every epoch.
        self._full = ds
        self._sample_idx = 0
        self._epoch = 0
        self._data = self._make_shard()

    def _make_shard(self):
        ds = self._full
        if self.shuffle:
            ds = ds.shuffle(seed=self.seed + self._epoch)
        return split_dataset_by_node(ds, self.dp_rank, self.dp_world_size)

    def _get_data_iter(self):
        if self._sample_idx >= len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_idx += 1
                block = torch.as_tensor(sample["input_ids"], dtype=torch.long)
                label = torch.empty_like(block)
                label[:-1] = block[1:]
                label[-1] = IGNORE_INDEX  # unused by _denoising_ce
                yield {"input": block}, label

            if not self.infinite:
                logger.warning(f"Pretokenized dataset {self.dataset_path} exhausted")
                break
            self._sample_idx = 0
            self._epoch += 1
            self._data = self._make_shard()   # new order every epoch
            logger.warning(
                f"Pretokenized dataset {self.dataset_path} re-looping "
                f"(epoch {self._epoch})"
            )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._sample_idx = state_dict["sample_idx"]
        epoch = state_dict.get("epoch", 0)
        if epoch != self._epoch:
            self._epoch = epoch
            self._data = self._make_shard()

    def state_dict(self) -> dict[str, Any]:
        return {"sample_idx": self._sample_idx, "epoch": self._epoch}


def build_pretokenized_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer | None,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Dataloader for pre-tokenized data.

    The tokenizer is unused (the data is already tokenized) but accepted to
    match the TrainSpec `build_dataloader_fn` signature. Set
    `training.dataset_path` to the .dat path.
    """
    del tokenizer
    dataset_path = job_config.training.dataset_path
    if not dataset_path:
        raise ValueError(
            "pretokenized dataset requires training.dataset_path (path to the .dat)"
        )

    # PretokenizedBlockDataset does not shard across DataLoader workers: as an
    # IterableDataset with num_workers > 0, every worker would yield the same
    # blocks and silently duplicate data. Fail explicitly instead; a split via
    # torch.utils.data.get_worker_info() in __iter__ would be needed first.
    n_workers = getattr(job_config.training.dataloader, "num_workers", 0)
    if n_workers:
        raise ValueError(
            f"pretokenized dataloader supports only num_workers=0 "
            f"(got {n_workers}); worker sharding of the IterableDataset is not "
            "implemented, so more workers would duplicate data"
        )

    ds = PretokenizedBlockDataset(
        dataset_path=dataset_path,
        seq_len=job_config.training.seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        seed=getattr(getattr(job_config, "debug", None), "seed", None) or 0,
    )
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
    )
