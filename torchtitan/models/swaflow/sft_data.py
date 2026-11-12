# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""SFT dataloader.

Reads the two bin files written by `scripts/data/build_sft.py` and yields
samples that keep the prompt clean, noise only the response and score only
the response:

    {"input": tokens [L], "clean_mask": clean [L]},  labels [L]

- positions with `clean_mask[i] = 1` stay clean embeddings, without noise, in
  the model (`SphereDiTForPretrain.forward`)
- `labels[j] = tokens[j+1]` (torchtitan's next-token shift), set to
  IGNORE_INDEX where position j+1 is prompt. The loss (`_denoising_ce` in
  `loss.py`) maps back with `pred[:, i] <-> labels[:, i-1]`, so only response
  positions are scored.

Why the prompt is kept clean: for a model pretrained with uniform α, clean
positions next to noised ones are out of distribution. SFT trains on the
inference-time input format (clean prefix, noised response) directly.

The slot convention is that of `indexed_data.py`:

    slot = sample_index * dp_world_size + dp_rank

so the state is a single integer and resuming has no duplicates or gaps.
Blocks are shuffled at build time (`build_sft.py`, design choice 6) and read
sequentially here.
"""

from __future__ import annotations

import json
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

__all__ = ["SFTBlockDataset", "build_sft_dataloader", "load_sft_blocks"]


def load_sft_blocks(
    build_dir: str | Path, split: str = "train"
) -> tuple[np.memmap, np.memmap, dict]:
    """Open `<build>/<split>.{tokens,clean}.bin` as [n_blocks, seq_len] arrays."""
    build = Path(build_dir)
    meta = json.loads((build / "meta.json").read_text(encoding="utf-8"))
    seq_len = int(meta["seq_len"])
    tok_path = build / f"{split}.tokens.bin"
    n_blocks = tok_path.stat().st_size // (2 * seq_len)
    tokens = np.memmap(
        tok_path, dtype=np.uint16, mode="r", shape=(n_blocks, seq_len)
    )
    clean = np.memmap(
        build / f"{split}.clean.bin", dtype=np.uint8, mode="r", shape=(n_blocks, seq_len)
    )
    return tokens, clean, meta


class SFTBlockDataset(IterableDataset, Stateful):
    """Yield SFT blocks as (input + clean_mask, label)."""

    def __init__(
        self,
        build_dir: str,
        seq_len: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        emit_clean_mask: bool = True,
    ) -> None:
        self.tokens, self.clean, self.meta = load_sft_blocks(build_dir)
        if int(self.meta["seq_len"]) != seq_len:
            raise ValueError(
                f"SFT build has seq_len {self.meta['seq_len']} but "
                f"training.seq_len is {seq_len}; they must match"
            )
        self.build_dir = build_dir
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.infinite = infinite
        # AR models take no `clean_mask` (`ARTransformer.forward(tokens)` has no
        # kwargs). For AR SFT only the label mask applies, which is standard AR
        # SFT (no loss on the prompt); an AR model always reads the prompt
        # clean, so the input needs no marking.
        self.emit_clean_mask = emit_clean_mask
        self._sample_index = 0
        self._n_samples = self.tokens.shape[0] // dp_world_size
        logger.info(
            f"SFT dataset '{self.meta['build_name']}': "
            f"{self.tokens.shape[0]:,} blocks × {seq_len} = "
            f"{self.tokens.size / 1e9:.3f}B tokens "
            f"(clean fraction {self.meta.get('clean_fraction', float('nan')):.3f}, "
            f"template {self.meta.get('template')})"
        )

    def __iter__(self):
        while True:
            if self._sample_index >= self._n_samples:
                if not self.infinite:
                    logger.warning(
                        f"SFT dataset exhausted ({self._n_samples:,} blocks/rank)"
                    )
                    return
                logger.warning("SFT dataset wrapped around (more than 1 epoch)")
                self._sample_index = 0
            slot = self._sample_index * self.dp_world_size + self.dp_rank
            block = torch.from_numpy(np.array(self.tokens[slot], dtype=np.int64))
            clean = torch.from_numpy(np.array(self.clean[slot], dtype=np.uint8))
            label = torch.empty_like(block)
            label[:-1] = block[1:]
            label[-1] = IGNORE_INDEX  # unused by _denoising_ce
            # Prompt (clean) positions are not scored. label[j] is the token at
            # position j+1, so the mask is shifted by one.
            label[:-1] = torch.where(
                clean[1:].bool(), torch.full_like(label[:-1], IGNORE_INDEX), label[:-1]
            )
            self._sample_index += 1
            if not self.emit_clean_mask:
                yield {"input": block}, label
                continue
            yield {"input": block, "clean_mask": clean}, label

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self._sample_index,
            "build_name": self.meta.get("build_name"),
            "sample_checksum": self.meta.get("sample_checksum"),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        name = state_dict.get("build_name")
        if name and name != self.meta.get("build_name"):
            raise ValueError(
                f"SFT build '{name}' in the checkpoint differs from the current "
                f"'{self.meta.get('build_name')}'"
            )
        checksum = state_dict.get("sample_checksum")
        if checksum and checksum != self.meta.get("sample_checksum"):
            raise ValueError("the SFT build has been rebuilt (the data order would not continue)")
        self._sample_index = int(state_dict["sample_index"])


def build_sft_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer | None,
    job_config: JobConfig,
    infinite: bool = True,
    emit_clean_mask: bool = True,
) -> ParallelAwareDataloader:
    """Dataloader for `training.dataset = "sft"`; `dataset_path` is the build directory.

    The `infinite` argument is ignored. SFT runs for one epoch by default, and a
    step past the end should not silently start a second pass;
    `[swaflow.sft] allow_multiple_epochs = true` enables looping.
    """
    del tokenizer
    build_dir = job_config.training.dataset_path
    if not build_dir:
        raise ValueError("sft dataset requires training.dataset_path")
    n_workers = getattr(job_config.training.dataloader, "num_workers", 0)
    if n_workers:
        raise ValueError(f"sft dataloader supports only num_workers=0 (got {n_workers})")

    ds = SFTBlockDataset(
        build_dir=build_dir,
        seq_len=job_config.training.seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        # One pass by default; loops only with `[swaflow.sft]
        # allow_multiple_epochs = true` (for many epochs over a small build,
        # instead of packing every repeat into the build).
        infinite=bool(
            getattr(getattr(job_config, "swaflow", None), "sft", None)
            and job_config.swaflow.sft.allow_multiple_epochs
        ),
        emit_clean_mask=emit_clean_mask,
    )
    del infinite
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
    )
