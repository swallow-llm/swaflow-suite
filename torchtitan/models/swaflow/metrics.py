# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Training-time logging: (α, per-sample CE), attention logits, gradient norms.

(α, per-sample CE) is always logged, for measuring the truncation window (binned
loss vs α) and monitoring it during training. Each rank appends JSONL to
<dump_folder>/<kind>/rank<r>.jsonl; detached GPU tensors are buffered and written
every FLUSH_EVERY calls to avoid a GPU->CPU sync per step, with a final flush at exit.
"""

import atexit
import json
import os

import torch
from torch import Tensor


def _rank() -> int:
    """Rank of this process, preferring the `RANK` environment variable.

    A short run may first flush at `atexit`, after `destroy_process_group()`, when
    `torch.distributed.get_rank()` would make every rank write to rank0.jsonl.
    torchrun always sets `RANK`.
    """
    env = os.environ.get("RANK")
    if env is not None:
        return int(env)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


#: Observer that receives (α, CE) from `AlphaCELogger.flush`; the watchdog uses it
#: to detect a flattening α profile. It gets the CPU values at flush time, so no
#: syncs are added. A process global, because `AlphaCELogger` is created in
#: `build_loss_fn` (inside `Trainer.__init__`) before the watchdog exists.
_ALPHA_CE_OBSERVER = None


def set_alpha_ce_observer(fn) -> None:
    """fn(alphas: list[float], ces: list[float]) -> None; None removes it."""
    global _ALPHA_CE_OBSERVER
    _ALPHA_CE_OBSERVER = fn


class AlphaCELogger:
    """Log (α, per-sample CE) during training to JSONL.

    record() is called by the loss once per microbatch (assumes forward -> loss
    runs sequentially within a rank; no PP).
    """

    FLUSH_EVERY = 50

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._buf: list[tuple[int, Tensor, Tensor]] = []
        self._call = 0
        self._file = None
        atexit.register(self.flush)

    def record(self, alpha: Tensor, per_sample_ce: Tensor) -> None:
        """alpha: [B] (cleanness); per_sample_ce: [B] (CE averaged over valid tokens)."""
        self._buf.append(
            (self._call, alpha.detach().float(), per_sample_ce.detach().float())
        )
        self._call += 1
        if len(self._buf) >= self.FLUSH_EVERY:
            self.flush()

    def _ensure_file(self):
        if self._file is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self._file = open(
                os.path.join(self.log_dir, f"rank{_rank()}.jsonl"), "a"
            )
        return self._file

    def flush(self) -> None:
        if not self._buf:
            return
        f = self._ensure_file()
        observer = _ALPHA_CE_OBSERVER
        for call, alpha, ce in self._buf:
            alpha_list = alpha.tolist()
            ce_list = ce.tolist()
            f.write(
                json.dumps(
                    {
                        "call": call,
                        "alpha": [round(v, 6) for v in alpha_list],
                        "ce": [round(v, 5) for v in ce_list],
                    }
                )
                + "\n"
            )
            if observer is not None:
                observer(alpha_list, ce_list)
        self._buf.clear()
        f.flush()


class AttentionStatsLogger:
    """Log the maximum attention logit and q/k norms per layer.

    Records the attention-logit scale that QK-norm controls; recording only.
    Output: `<dump_folder>/attn_stats/rank<r>.jsonl`.

    The exact max needs the [L, L] logit matrix, so it is measured once every
    `every` microbatches on every `query_stride`-th query (by default 6.25% of
    QK^T once per 50 steps, <0.1% overhead). The reported max is therefore a lower
    bound on the true max.

    Do not combine with torch.compile: with `every > 1` a Python bool flips inside
    the attention block and triggers recompilation; use `every = 0` when compiling.

    Under activation checkpointing record() is also called during recomputation;
    the values equal the forward ones, so entries keyed by layer_id are overwritten.
    """

    FLUSH_EVERY = 20

    def __init__(self, log_dir: str, every: int = 50, query_stride: int = 16) -> None:
        self.log_dir = log_dir
        self.every = int(every)
        self.query_stride = max(1, int(query_stride))
        self._pending: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        self._buf: list[tuple[int, dict[int, tuple[Tensor, Tensor, Tensor]]]] = []
        self._call = -1
        self._active = False
        self._file = None
        atexit.register(self.flush)

    @property
    def enabled(self) -> bool:
        return self.every > 0

    @property
    def active(self) -> bool:
        """Whether to measure in this microbatch (read by Attention.forward)."""
        return self._active

    def begin_microbatch(self) -> None:
        """Called by the loss once per microbatch, before the attention forward."""
        self._call += 1
        self._active = self.enabled and self._call % self.every == 0

    def end_microbatch(self) -> None:
        """Finalize the records of this microbatch."""
        if self._pending:
            self._buf.append((self._call, self._pending))
            self._pending = {}
            if len(self._buf) >= self.FLUSH_EVERY:
                self.flush()
        self._active = False

    @torch.no_grad()
    def record(self, layer_id: int, xq: Tensor, xk: Tensor) -> None:
        """xq / xk: [B, H, L, D] after RoPE, as passed to SDPA."""
        if not self._active:
            return
        scale = xq.shape[-1] ** -0.5
        q = xq[:, :, :: self.query_stride, :]
        logit_max = (torch.matmul(q, xk.transpose(-2, -1)) * scale).abs().amax()
        self._pending[layer_id] = (
            logit_max.detach().float(),
            xq.norm(dim=-1).amax().detach().float(),
            xk.norm(dim=-1).amax().detach().float(),
        )

    def _ensure_file(self):
        if self._file is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self._file = open(os.path.join(self.log_dir, f"rank{_rank()}.jsonl"), "a")
        return self._file

    def flush(self) -> None:
        if not self._buf:
            return
        f = self._ensure_file()
        for call, per_layer in self._buf:
            layers = sorted(per_layer)
            f.write(
                json.dumps(
                    {
                        "call": call,
                        "query_stride": self.query_stride,
                        "layer": layers,
                        "logit_absmax": [
                            round(per_layer[i][0].item(), 4) for i in layers
                        ],
                        "q_norm_max": [round(per_layer[i][1].item(), 4) for i in layers],
                        "k_norm_max": [round(per_layer[i][2].item(), 4) for i in layers],
                    }
                )
                + "\n"
            )
        self._buf.clear()
        f.flush()


#: Process-wide AttentionStatsLogger read by Attention.forward. A global rather
#: than a model arg, so that runtime paths do not leak into the run config.
_ATTENTION_STATS: "AttentionStatsLogger | None" = None


def configure_attention_stats(
    log_dir: str, every: int = 50, query_stride: int = 16
) -> "AttentionStatsLogger | None":
    global _ATTENTION_STATS
    _ATTENTION_STATS = AttentionStatsLogger(log_dir, every, query_stride)
    return _ATTENTION_STATS if _ATTENTION_STATS.enabled else None


def get_attention_stats() -> "AttentionStatsLogger | None":
    return _ATTENTION_STATS


def reset_attention_stats() -> None:
    """For tests: clear the process global."""
    global _ATTENTION_STATS
    _ATTENTION_STATS = None


class GradNormLogger:
    """Log the pre-clipping gradient norm every step.

    Upstream MetricsProcessor logs `grad_norm` only every `metrics.log_freq` steps
    and misses isolated spikes. Tensors are buffered and converted with `.item()`
    every FLUSH_EVERY calls, adding no GPU->CPU syncs.
    """

    FLUSH_EVERY = 50

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._buf: list[tuple[int, Tensor]] = []
        self._call = 0
        self._file = None
        atexit.register(self.flush)

    def record(self, grad_norm: Tensor) -> None:
        self._buf.append((self._call, grad_norm.detach().float()))
        self._call += 1
        if len(self._buf) >= self.FLUSH_EVERY:
            self.flush()

    def _ensure_file(self):
        if self._file is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self._file = open(os.path.join(self.log_dir, f"rank{_rank()}.jsonl"), "a")
        return self._file

    def flush(self) -> None:
        if not self._buf:
            return
        f = self._ensure_file()
        for call, gn in self._buf:
            f.write(json.dumps({"call": call, "grad_norm": round(gn.item(), 5)}) + "\n")
        self._buf.clear()
        f.flush()
