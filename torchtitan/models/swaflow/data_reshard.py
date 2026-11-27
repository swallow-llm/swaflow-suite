# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Resume the indexed data stream with a different number of data-parallel ranks.

torchtitan stores the dataloader state once per data-parallel rank
(`dataloader.dp_rank_<r>`) and refuses to load it under another world size. For
the indexed dataset (`indexed_data.py`) the per-rank state is a single integer,
and every rank holds the same value, because of the slot convention

    slot = sample_index * dp_world_size + dp_rank.

A checkpoint saved with W ranks at sample index s has therefore consumed exactly
the slots [0, s * W). With W' ranks the same position is s' = s * W / W', and the
run continues with the next slot. As long as the global batch is unchanged
(`local_batch_size` x ranks x gradient accumulation), step N reads the slots
[N * B, (N + 1) * B) whatever the number of ranks, so each step sees the same
sequences as in the original run; only their assignment to ranks and
microbatches differs.

`attach_dataloader_resharding` wraps the checkpointer's `dcp_load`. When the
checkpoint was saved with the current number of ranks it does nothing. Otherwise
it reads the saved world size from the checkpoint metadata, loads the saved
per-rank states, checks that they all agree, converts the position and hands
the result to the dataloader's own `load_state_dict`.

The weights and the optimizer state need no conversion: DCP stores them as
global tensors and reshards them on load.
"""

from __future__ import annotations

import copy
import pickle
import re
from collections.abc import Callable, Mapping
from typing import Any

from torch.distributed.checkpoint import FileSystemReader

from torchtitan.components.checkpoint import DATALOADER
from torchtitan.models.swaflow.indexed_data import IndexedMixtureDataset
from torchtitan.tools.logging import logger

__all__ = [
    "attach_dataloader_resharding",
    "reshard_loader_state",
    "saved_dp_world_size",
    "source_ranks",
]

_RANK_KEY = re.compile(rf"^{DATALOADER}\.dp_rank_(\d+)$")
# StatefulDataLoader counters. With a stateful dataset torchdata does not use
# them to fast-forward; they are kept consistent with the new position anyway.
_SAMPLE_COUNTERS = ("_num_yielded", "_sampler_iter_yielded")


def saved_dp_world_size(checkpoint_id: str) -> int | None:
    """Number of data-parallel ranks the checkpoint was saved with.

    Counted from the `dataloader.dp_rank_<r>` entries in the DCP metadata.
    Returns None if the checkpoint has no dataloader state.
    """
    metadata = FileSystemReader(checkpoint_id).read_metadata()
    ranks = sorted(
        int(m.group(1))
        for key in metadata.state_dict_metadata
        if (m := _RANK_KEY.match(key))
    )
    if not ranks:
        return None
    if ranks != list(range(len(ranks))):
        raise ValueError(
            f"{checkpoint_id}: dataloader states are not stored for ranks "
            f"0..{len(ranks) - 1} (found {len(ranks)} entries, highest rank {ranks[-1]})"
        )
    return len(ranks)


def source_ranks(rank: int, world_size: int, saved_world_size: int) -> list[int]:
    """Saved ranks whose state the given new rank reads.

    With fewer ranks than before, the saved ranks are split among the new ranks,
    so that together they check every saved state. Every new rank also reads
    saved rank 0 and compares against it, so all new ranks agree on the position
    without communicating.
    """
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside [0, {world_size})")
    if world_size <= saved_world_size:
        own = range(rank, saved_world_size, world_size)
    else:
        own = [rank % saved_world_size]
    return sorted({0, *own})


def reshard_loader_state(
    states: Mapping[int, dict[str, Any]], saved_world_size: int, world_size: int
) -> dict[str, Any]:
    """Convert saved per-rank dataloader states to the state of one new rank.

    Args:
        states: unpickled `StatefulDataLoader` states, keyed by saved rank
            (must include rank 0)
        saved_world_size: number of data-parallel ranks at save time
        world_size: number of data-parallel ranks now

    Returns:
        A `StatefulDataLoader` state for any rank of the new run (the slot
        convention makes it the same for all of them).
    """
    if 0 not in states:
        raise ValueError("the state of saved rank 0 is required")
    ref = states[0]
    if ref.get("dataset_state") is None:
        raise ValueError("the saved dataloader state holds no dataset position")
    for rank, state in sorted(states.items()):
        if state.get("dataset_state") != ref["dataset_state"] or any(
            state.get(k) != ref.get(k) for k in _SAMPLE_COUNTERS
        ):
            raise ValueError(
                f"saved ranks 0 and {rank} are at different data positions "
                f"({ref['dataset_state']} vs {state.get('dataset_state')}); "
                "the position cannot be converted"
            )

    sample_index = int(ref["dataset_state"]["sample_index"])
    consumed = sample_index * saved_world_size
    if consumed % world_size:
        raise ValueError(
            f"{consumed:,} consumed slots ({saved_world_size} ranks x {sample_index:,}) "
            f"cannot be split evenly across {world_size} ranks"
        )

    out = copy.deepcopy(ref)
    out["dataset_state"]["sample_index"] = consumed // world_size
    for key in _SAMPLE_COUNTERS:
        if isinstance(out.get(key), int):
            out[key] = out[key] * saved_world_size // world_size
    sampler = out.get("_sampler_iter_state")
    if isinstance(sampler, dict) and isinstance(sampler.get("samples_yielded"), int):
        sampler["samples_yielded"] = sampler["samples_yielded"] * saved_world_size // world_size
    return out


class _ReshardedLoad:
    """Stands in for the dataloader during `dcp.load` when the world size changed.

    `state_dict()` gives the template DCP fills: the saved ranks this rank reads
    (`source_ranks`) and the saved world size. `load_state_dict()` checks and
    converts them and passes the result to the real dataloader.
    """

    def __init__(self, dataloader, saved_world_size: int) -> None:
        self.dataloader = dataloader
        self.saved_world_size = saved_world_size
        self.consumed_slots: int | None = None

    def state_dict(self) -> dict[str, Any]:
        rank, world = self.dataloader.dp_rank, self.dataloader.dp_world_size
        template: dict[str, Any] = {
            f"dp_rank_{r}": b"" for r in source_ranks(rank, world, self.saved_world_size)
        }
        template["world_size"] = 0
        return template

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        rank, world = self.dataloader.dp_rank, self.dataloader.dp_world_size
        stored = int(state_dict.pop("world_size"))
        if stored != self.saved_world_size:
            raise ValueError(
                f"the checkpoint records {stored} data-parallel ranks but has states "
                f"for {self.saved_world_size}"
            )
        states = {
            int(key.removeprefix("dp_rank_")): pickle.loads(blob)
            for key, blob in state_dict.items()
        }
        new_state = reshard_loader_state(states, self.saved_world_size, world)
        self.consumed_slots = self.saved_world_size * int(states[0]["dataset_state"]["sample_index"])
        if rank == 0:
            logger.info(
                f"[data-reshard] checkpoint saved with {self.saved_world_size} "
                f"data-parallel ranks, resuming with {world}: "
                f"{self.consumed_slots:,} slots consumed, sample_index per rank "
                f"{states[0]['dataset_state']['sample_index']:,} -> "
                f"{new_state['dataset_state']['sample_index']:,}"
            )
        self.dataloader.load_state_dict(
            {f"dp_rank_{rank}": pickle.dumps(new_state), "world_size": world}
        )


def attach_dataloader_resharding(
    checkpointer,
    *,
    step_getter: Callable[[], int],
    gradient_accumulation_steps: int,
) -> bool:
    """Let `checkpointer` resume the indexed dataloader under another world size.

    Args:
        checkpointer: torchtitan `CheckpointManager`
        step_getter: returns the training step (valid after the load)
        gradient_accumulation_steps: microbatches per optimizer step in this run

    Returns:
        True if attached (the dataloader reads the indexed dataset).
    """
    dataloader = checkpointer.states.get(DATALOADER)
    if not isinstance(getattr(dataloader, "dataset", None), IndexedMixtureDataset):
        return False
    global_batch_size = (
        dataloader.batch_size * gradient_accumulation_steps * dataloader.dp_world_size
    )
    inner = checkpointer.dcp_load

    def _dcp_load(state_dict, checkpoint_id, from_hf, from_quantized):
        adapter = None
        if not from_hf and state_dict.get(DATALOADER) is dataloader:
            saved = saved_dp_world_size(checkpoint_id)
            if saved is not None and saved != dataloader.dp_world_size:
                adapter = _ReshardedLoad(dataloader, saved)
                state_dict[DATALOADER] = adapter
        out = inner(
            state_dict,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
        )
        if adapter is not None:
            step = step_getter()
            expected = step * global_batch_size
            if adapter.consumed_slots != expected:
                logger.warning(
                    f"[data-reshard] the checkpoint consumed {adapter.consumed_slots:,} "
                    f"slots, but step {step} x global batch {global_batch_size} = "
                    f"{expected:,}: the global batch differs from the original run's, "
                    "so later steps will not read the same data as that run did"
                )
        return out

    checkpointer.dcp_load = _dcp_load
    return True
