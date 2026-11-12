# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Noise schedules. Time convention: α=1 is clean, α=0 is pure noise (as in MDLM).

A schedule maps t ∈ [0, 1] monotonically to α ∈ [0, 1]: t=0 -> α=1 (clean),
t=1 -> α=0 (noise). The S-FLM setting invert_time_convention=true uses the
opposite convention; convert with α_new = 1 − α_old.
Truncation is static only (`TruncatedScheduleWrapper`, window [0, α_max]).
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

__all__ = ["CosineSquared", "LogLinear", "NoiseSchedule", "TruncatedScheduleWrapper"]


def _as_tensor(x: Tensor | float) -> Tensor:
    """Return a Tensor unchanged; convert a Python float in fp64.

    device="cpu" is explicit because torchtitan builds the model under the meta
    device, where a scalar constant would otherwise become a meta tensor and
    float() would fail (e.g. in TruncatedScheduleWrapper.__init__). On the
    training hot path t is always a Tensor.
    """
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.float64, device="cpu")


class NoiseSchedule(ABC):
    """Monotonically decreasing map t ∈ [0, 1] -> α ∈ [0, 1] (α=1 clean, α=0 noise)."""

    @abstractmethod
    def alpha(self, t: Tensor | float) -> Tensor:
        """Cleanness α(t) at time t."""

    @abstractmethod
    def alpha_prime(self, t: Tensor | float) -> Tensor:
        """Derivative dα/dt (negative)."""

    @abstractmethod
    def alpha_inv(self, alpha: Tensor | float) -> Tensor:
        """Inverse of α(t): alpha_inv(alpha(t)) = t."""


class LogLinear(NoiseSchedule):
    """α(t) = 1 − (1−eps)·t (linear in α).

    MDLM's log-linear schedule σ(t) = −log(1 − (1−eps)·t) with α = exp(−σ).
    With the default eps=0, α(1) = 0 exactly.
    """

    def __init__(self, eps: float = 0.0) -> None:
        if not 0.0 <= eps < 1.0:
            raise ValueError(f"eps must be in [0, 1), got {eps}")
        self.eps = eps

    def alpha(self, t: Tensor | float) -> Tensor:
        t = _as_tensor(t)
        return 1.0 - (1.0 - self.eps) * t

    def alpha_prime(self, t: Tensor | float) -> Tensor:
        t = _as_tensor(t)
        return torch.full_like(t, -(1.0 - self.eps))

    def alpha_inv(self, alpha: Tensor | float) -> Tensor:
        alpha = _as_tensor(alpha)
        return (1.0 - alpha) / (1.0 - self.eps)


class CosineSquared(NoiseSchedule):
    """α(t) = cos²(π·t/2), with zero slope at t=0 and t=1."""

    def alpha(self, t: Tensor | float) -> Tensor:
        t = _as_tensor(t)
        return torch.cos(torch.pi / 2.0 * t) ** 2

    def alpha_prime(self, t: Tensor | float) -> Tensor:
        # d/dt cos²(πt/2) = −(π/2)·sin(πt)
        t = _as_tensor(t)
        return -(torch.pi / 2.0) * torch.sin(torch.pi * t)

    def alpha_inv(self, alpha: Tensor | float) -> Tensor:
        alpha = _as_tensor(alpha)
        return 2.0 / torch.pi * torch.acos(torch.sqrt(alpha.clamp(0.0, 1.0)))


class TruncatedScheduleWrapper(NoiseSchedule):
    """Statically restrict the base schedule's α to [alpha_min, alpha_max] (α-remap).

    The base α is mapped affinely (same semantics as the S-FLM reference):

        scale    = (alpha_max − alpha_min) / (base_max − base_min)
        offset   = alpha_min − scale · base_min
        α(t)     = scale · base.α(t) + offset
        α'(t)    = scale · base.α'(t)

    base_max = base.alpha(0) and base_min = base.alpha(1) are evaluated at init,
    so alpha(0) = alpha_max and alpha(1) = alpha_min.

    Remapping t before the base schedule (t-remap) agrees with this for the linear
    LogLinear but changes the density of α inside the window for nonlinear
    schedules (e.g. CosineSquared); tests/test_schedules.py pins the α-remap.

    Normally alpha_min=0 (sampling starts at the noise end) and only alpha_max is
    set, giving the window [0, α_max].
    """

    def __init__(
        self, base: NoiseSchedule, *, alpha_max: float, alpha_min: float = 0.0
    ) -> None:
        if not 0.0 <= alpha_min < alpha_max <= 1.0:
            raise ValueError(
                "expected 0 <= alpha_min < alpha_max <= 1, got "
                f"alpha_min={alpha_min}, alpha_max={alpha_max}"
            )
        self.base = base
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        base_max = float(base.alpha(0.0))
        base_min = float(base.alpha(1.0))
        self._scale = (alpha_max - alpha_min) / (base_max - base_min)
        self._offset = alpha_min - self._scale * base_min

    def alpha(self, t: Tensor | float) -> Tensor:
        return self._scale * self.base.alpha(t) + self._offset

    def alpha_prime(self, t: Tensor | float) -> Tensor:
        return self._scale * self.base.alpha_prime(t)

    def alpha_inv(self, alpha: Tensor | float) -> Tensor:
        alpha = _as_tensor(alpha)
        return self.base.alpha_inv((alpha - self._offset) / self._scale)
