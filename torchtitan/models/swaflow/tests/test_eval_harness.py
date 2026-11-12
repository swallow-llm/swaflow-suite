# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Tests for the offline evaluation harness (`scripts/eval/`).

Guards the ways this plumbing can break silently:

1. `model.to(dtype=...)` drops the imaginary part of the complex buffer
   `freqs_cis` and breaks RoPE (this degraded the 150M val CE from 1.669 to 3.156)
2. the common alpha grid drifts from the 3B window (breaks cross-size comparison)
3. the a@ce0.1 interpolation (log vs linear) gets swapped
4. thinning to 100B spacing yields fewer than 20 points through accumulated rounding
5. an MCQ batch mixes sequence lengths (the model has no padding mask, so this
   silently measures something else)
6. the AR scoring position is off by one (a loss is still produced)
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import pytest
import torch

from scripts.eval.ckpt_eval import (
    COMMON_GRID,
    FINE16_GRID,
    FINE32_GRID,
    WINDOW_GRIDS,
    _cast_params,
    window_grid,
)
from scripts.eval.loader import build_eval_model, select_steps
from scripts.eval.mcq import Doc, Request, _length_batches, build_requests, score_accuracy
from scripts.eval.profiles import SIZES, crossing

_CONFIGS = Path(__file__).resolve().parents[1] / "train_configs"


def _alpha_grid(config: str) -> list[float]:
    cfg = tomllib.loads((_CONFIGS / config).read_text(encoding="utf-8"))
    return list(cfg["swaflow"]["eval"]["alpha_grid"])


# --------------------------------------------------------------- grids
def test_common_grid_is_exactly_the_3b_training_grid() -> None:
    """The common grid is the 3B window [0, 0.10] split into 8 equal steps.

    It must be identical to the 3B training grid, so the 3B validation series
    can be reused as the common series without re-evaluating its checkpoints.
    """
    assert COMMON_GRID == pytest.approx(_alpha_grid("flow_3b.toml"))


def test_common_grid_is_inside_every_size_window() -> None:
    """The common grid lies inside every size's window."""
    for _run, (_label, _dim, amax, _ar) in SIZES.items():
        assert max(COMMON_GRID) <= amax + 1e-12


@pytest.mark.parametrize("grid", [FINE16_GRID, FINE32_GRID])
def test_fine_grids_contain_the_common_grid(grid: list[float]) -> None:
    for a in COMMON_GRID:
        assert any(abs(a - b) < 1e-9 for b in grid), a


@pytest.mark.parametrize("n", sorted(WINDOW_GRIDS.values()))
def test_window_grid_is_uniform_and_ends_at_alpha_max(n: int) -> None:
    amax = 0.15
    g = window_grid(amax, n)
    assert len(g) == n
    assert g[-1] == pytest.approx(amax)
    diffs = [b - a for a, b in zip(g, g[1:])]
    assert max(diffs) - min(diffs) < 1e-12


def test_window_grid_is_needed_because_150m_a_at_ce01_is_outside_the_common_grid() -> None:
    """The 150M a@ce0.1 lies outside the common grid (measured 0.1074 > 0.10).

    This is why a@ce0.1 is measured on a per-size window grid: the common grid
    cannot bracket the crossing for 150M / 500M.
    """
    assert SIZES["flow_150m"][2] > max(COMMON_GRID)
    assert max(window_grid(SIZES["flow_150m"][2], 32)) > 0.1074


# --------------------------------------------------------------- interpolation
def test_crossing_uses_log_ce_and_beats_linear_on_an_exponential_profile() -> None:
    """When CE is exponential in alpha, log interpolation is exact and linear overshoots.

    Measured profiles are close to this shape (CE drops 3-4x per 0.01875 of
    alpha), so linear interpolation turns grid coarseness into a systematic error.
    """
    # CE(alpha) = exp(-100 alpha) -> CE = 0.1 at alpha = ln(10)/100 = 0.023026
    pts = [(a / 1000, math.exp(-100 * a / 1000)) for a in range(0, 60, 10)]
    exact = math.log(10) / 100
    a_log = crossing(pts, 0.1, log=True)
    a_lin = crossing(pts, 0.1, log=False)
    assert a_log == pytest.approx(exact, rel=1e-9)
    assert a_lin > a_log  # the chord always lies above


def test_crossing_returns_none_when_the_level_is_outside_the_grid() -> None:
    pts = [(0.01, 5.0), (0.02, 3.0), (0.03, 1.0)]
    assert crossing(pts, 0.1) is None


def test_crossing_takes_the_largest_alpha_crossing() -> None:
    """Spurious crossings from low-alpha noise do not move the trivialization onset."""
    pts = [(0.01, 0.05), (0.02, 5.0), (0.03, 1.0), (0.04, 0.01)]
    a = crossing(pts, 0.1, log=True)
    assert a is not None and a > 0.03


# --------------------------------------------------------------- thinning
def test_select_steps_hits_every_100b_target_for_a_2t_run() -> None:
    """Thinning 20B-spaced checkpoints to 100B spacing gives exactly 20 points (2T / 100B).

    "Take one every stride steps" accumulates rounding and yields only 18.
    100B / 8.39M = 11,920.93 steps is not a multiple of the checkpoint interval
    2,384, so the checkpoint nearest each target is taken (off by at most half
    an interval = 1,192 steps).
    """
    steps = [2384 * k for k in range(1, 101)] + [238418]
    sel = select_steps(steps, every_tokens=100_000_000_000)
    assert len(sel) == 20
    assert sel[0] == 11920  # nearest to the 100B target 11,920.93
    assert sel[-1] == 238418  # the final step, exactly 2T
    assert all(s in steps for s in sel)
    stride = 100_000_000_000 / 8_388_608
    for k, s in enumerate(sel, start=1):
        assert abs(s - k * stride) <= 2384 / 2 + 1


def test_select_steps_without_thinning_keeps_everything() -> None:
    steps = [2384, 4768, 7152]
    assert select_steps(steps) == steps


# --------------------------------------------------------------- dtype
def test_cast_params_keeps_the_complex_rope_buffer_intact() -> None:
    """Why `model.to(dtype=...)` must not be used.

    `freqs_cis` is a complex buffer and `.to(bfloat16)` drops its imaginary part.
    The training-side FSDP MixedPrecisionPolicy casts only parameters to bf16,
    so evaluation must also cast only the parameters.
    """
    model = build_eval_model("flow", "debugmodel", alpha_max=0.1)
    assert model.freqs_cis.is_complex()
    before = model.freqs_cis.clone()

    _cast_params(model, torch.bfloat16)
    assert model.freqs_cis.is_complex()
    assert torch.equal(model.freqs_cis, before)
    assert all(p.dtype is torch.bfloat16 for p in model.parameters())

    # control: a plain .to(dtype) drops the imaginary part (breaks RoPE)
    model.to(dtype=torch.bfloat16)
    assert not model.freqs_cis.is_complex()


def test_cast_params_round_trips_back_to_fp32() -> None:
    """Loading is done in fp32, so casting back from bf16 must work."""
    model = build_eval_model("ar", "debugmodel")
    _cast_params(model, torch.bfloat16)
    _cast_params(model, torch.float32)
    assert all(p.dtype is torch.float32 for p in model.parameters())


# --------------------------------------------------------------- MCQ
class _Tok:
    """Whitespace pseudo-tokenizer (needs neither datasets nor a model)."""

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [len(w) for w in text.split()] or [1]}


def test_length_batches_never_mix_lengths() -> None:
    """The model has no padding mask, so mixed lengths are a bug."""
    reqs = [
        Request(i, 0, [1] * (3 + i % 4), [2] * (1 + i % 3), 5, i) for i in range(40)
    ]
    for batch in _length_batches(reqs, 8):
        assert len({r.total_len for r in batch}) == 1
        assert len(batch) <= 8
    assert sum(len(b) for b in _length_batches(reqs, 8)) == len(reqs)


def test_build_requests_keeps_the_continuation_and_trims_the_context_left() -> None:
    docs = [Doc(["aa bb cc dd ee"], ["ff gg"], 0)]
    reqs = build_requests(docs, _Tok(), max_len=4)
    assert len(reqs) == 1
    assert len(reqs[0].cont_ids) == 2  # the continuation is always kept
    assert reqs[0].total_len <= 4
    assert len(reqs[0].ctx_ids) >= 1  # position 0 is dropped by the loss convention, so keep it non-empty


def test_build_requests_numbers_requests_consecutively() -> None:
    """`index` seeds the noise, so changing the numbering breaks reproducibility."""
    docs = [Doc(["a", "a"], ["x", "y"], 0), Doc(["b", "b", "b"], ["p", "q", "r"], 2)]
    reqs = build_requests(docs, _Tok(), max_len=16)
    assert [r.index for r in reqs] == list(range(5))
    assert [(r.doc_id, r.choice_id) for r in reqs] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (1, 2)
    ]


def test_score_accuracy_picks_the_minimum_nll() -> None:
    """Both flow and AR pick the choice with minimum nll (same criterion)."""
    docs = [Doc(["c", "c"], ["short", "longer one"], 1)]
    reqs = build_requests(docs, _Tok(), max_len=16)
    # choice 0 wins on the raw sum, choice 1 wins after per-token normalization
    nll = [3.0, 4.0]
    assert len(reqs[0].cont_ids) == 1 and len(reqs[1].cont_ids) == 2
    out = score_accuracy(docs, reqs, nll)
    assert out["acc"] == 0.0          # 3.0 < 4.0 -> choice 0 (wrong)
    assert out["acc_norm_tok"] == 1.0  # 3.0 vs 2.0 -> choice 1 (correct)
    assert out["n_docs"] == 1.0


def test_score_accuracy_reports_binomial_stderr() -> None:
    docs = [Doc(["c", "c"], ["a", "b"], 0) for _ in range(100)]
    reqs = build_requests(docs, _Tok(), max_len=16)
    nll = [0.0 if r.choice_id == 0 else 1.0 for r in reqs]
    out = score_accuracy(docs, reqs, nll)
    assert out["acc"] == 1.0
    assert out["acc_stderr"] == 0.0


def test_ar_scoring_masks_start_one_position_before_the_continuation() -> None:
    """For AR, `logits[:, j]` predicts position j+1.

    The first choice token (position `len(ctx)`) is predicted at
    `j = len(ctx) - 1`, so the mask starts there. An off-by-one still yields
    a loss, so this is checked explicitly.
    """
    from scripts.eval import mcq as mcqlib

    class _Ident(torch.nn.Module):
        """Toy model mapping token ids to one-hot (perfect prediction)."""

        def forward(self, tokens):
            return torch.nn.functional.one_hot(tokens, num_classes=8).float() * 30.0

    reqs = [Request(0, 0, [1, 2, 3], [4, 5], 2, 0)]
    # under the next-token convention, a perfect predictor gives nll ~ 0
    nll = mcqlib.score_ar(_Ident(), reqs, device=torch.device("cpu"), batch_size=4)
    # one_hot has the logits at position j point to the token at position j, so
    # reading it under the next-token convention always misses (large nll).
    # This pins the direction of the shift.
    assert nll[0] > 10.0

    class _Shift(torch.nn.Module):
        """Logits at position j point to the token at position j+1 (the AR convention)."""

        def forward(self, tokens):
            nxt = torch.roll(tokens, shifts=-1, dims=1)
            return torch.nn.functional.one_hot(nxt, num_classes=8).float() * 30.0

    nll2 = mcqlib.score_ar(_Shift(), reqs, device=torch.device("cpu"), batch_size=4)
    assert nll2[0] < 1e-3


def test_flow_mcq_keeps_the_prefix_clean_and_noises_only_the_choice() -> None:
    """Flow MCQ keeps the prefix clean (alpha=1) and noises only the choice.

    This is the same mechanism as prefix conditioning at inference
    (`sampling._project_prefix`) and matches SFT (prompt as clean prefix,
    response noised). Otherwise the question itself would be corrupted
    before scoring the choices.
    """
    from torchtitan.models.swaflow import model as model_mod
    from scripts.eval import mcq as mcqlib

    net = build_eval_model("flow", "debugmodel", alpha_max=0.1)
    net.eval()
    seen: list[torch.Tensor] = []
    real = model_mod.SphereDiT.forward

    def spy(self, x, alpha):
        seen.append(x.detach().clone())
        return real(self, x, alpha)

    reqs = [Request(0, 0, [5, 6, 7], [8, 9], 2, 0)]
    model_mod.SphereDiT.forward = spy
    try:
        mcqlib.score_flow(
            net, reqs, alpha_grid=[0.05], seed=1234, task_index=0,
            device=torch.device("cpu"), batch_size=1,
        )
    finally:
        model_mod.SphereDiT.forward = real

    assert len(seen) == 1
    x_t = seen[0][0]
    clean = net.get_sphere_embeddings(
        torch.tensor([[5, 6, 7, 8, 9]])
    )[0]
    # prefix (positions 0..2) equals clean; the choice (3..4) differs
    assert torch.allclose(x_t[:3], clean[:3], atol=1e-5)
    assert not torch.allclose(x_t[3:], clean[3:], atol=1e-2)
    # stays on the sphere (embeddings are unit vectors)
    assert torch.allclose(x_t.norm(dim=-1), torch.ones(5), atol=1e-4)


def test_flow_mcq_noise_is_deterministic_across_calls() -> None:
    """The same (task, request, alpha) gives the same draw on every call (needed for comparison)."""
    from scripts.eval import mcq as mcqlib

    net = build_eval_model("flow", "debugmodel", alpha_max=0.1)
    net.eval()
    reqs = [Request(0, 0, [5, 6, 7], [8, 9], 2, 0)]
    kw = dict(alpha_grid=[0.05, 0.1], seed=1234, task_index=0,
              device=torch.device("cpu"), batch_size=1)
    a = mcqlib.score_flow(net, reqs, **kw)
    b = mcqlib.score_flow(net, reqs, **kw)
    assert a[0] == b[0]
    # a different alpha grid changes the value (it is an average over alpha)
    c = mcqlib.score_flow(net, reqs, **{**kw, "alpha_grid": [0.05]})
    assert c[0] != a[0]
