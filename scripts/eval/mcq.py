# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Multiple-choice (MCQ) scoring for flow and AR models on the same requests.

Prompt formats follow lm-evaluation-harness v0.4 for six tasks; `run_mcq.py --kind hf`
scores a released AR model with the same code to check the formats. The harness
itself is not used: a flow score is not a log-likelihood (it depends on α), and the
model has no padding mask, so a batch holds sequences of one length only.

Flow score of a choice:

    nll(choice) = (1/|A|) · Σ_{α∈A} Σ_{i∈choice} CE(x_i | prefix, α, ε(doc, α))

- `prefix_mode="clean"` (default) holds the prefix clean (α=1) and noises only the
  choice positions, as in prefix-conditioned sampling; `"noised"` noises the whole
  sequence (the training distribution).
- A defaults to the common grid. The reported accuracies use A = {0} with the PMI
  correction (`run_mcq.py --alpha-list 0 --pmi`) and `acc_norm_tok`.
- Noise is derived from `(task, doc, α)` via `validate.noise_seed` and shared by all
  choices of a question (common random numbers); `torch.randn` fills row-major, so
  with equal prefix lengths each choice position gets the same noise.
- `noise_draws > 1` averages several draws per α; the second draw is antithetic (−ε).
- This is a plain CE, not an ELBO: no exact likelihood exists under the truncated
  schedule.

AR / HF: the standard `Σ_i −log p(x_i | x_<i)`. The choice with the smallest nll wins.
Output keys: `acc` (raw sum, lm-eval `acc`), `acc_norm_tok` (divided by the choice's
token count), `acc_norm_byte` (divided by its byte count, lm-eval `acc_norm`).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import torch

from torchtitan.models.swaflow.sphere_ops import slerp, sphere_normalize
from torchtitan.models.swaflow.validate import noise_seed


#: Context wrappers. `sft` is the SFT format (`plain-user-assistant-v1` in
#: `scripts/data/build_sft.py`), to separate learned conditioning from a learned format.
PROMPT_TEMPLATES = {
    "none": "{ctx}",
    "sft": "User: {ctx}\n\nAssistant:",
}


__all__ = [
    "PROMPT_TEMPLATES",
    "TASKS",
    "Request",
    "build_requests",
    "load_task",
    "score_accuracy",
    "score_flow",
    "score_ar",
    "score_hf",
]


# ==========================================================================
# Task definitions (formats follow lm-eval v0.4)
# ==========================================================================
@dataclass
class Doc:
    """One question. `contexts` is per choice (only winogrande varies it)."""

    contexts: list[str]
    continuations: list[str]
    gold: int


def _hellaswag_preprocess(text: str) -> str:
    """Same as lm-eval `hellaswag.utils.preprocess`."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def _load_boolq(split: str) -> list[Doc]:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split=split)
    out = []
    for d in ds:
        ctx = f"{d['passage']}\nQuestion: {d['question']}?\nAnswer:"
        out.append(Doc([ctx, ctx], [" no", " yes"], int(bool(d["answer"]))))
    return out


def _load_openbookqa(split: str) -> list[Doc]:
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "main", split=split)
    out = []
    for d in ds:
        texts = list(d["choices"]["text"])
        labels = list(d["choices"]["label"])
        gold = labels.index(d["answerKey"])
        ctx = d["question_stem"]
        out.append(Doc([ctx] * len(texts), [" " + t for t in texts], gold))
    return out


def _load_piqa(split: str) -> list[Doc]:
    from datasets import load_dataset

    # `ybisk/piqa` is script-based and does not load with datasets 5.x;
    # `baber/piqa` is a parquet copy with the same content (validation: 1,838).
    ds = load_dataset("baber/piqa", split=split)
    out = []
    for d in ds:
        ctx = f"Question: {d['goal']}\nAnswer:"
        out.append(
            Doc([ctx, ctx], [" " + d["sol1"], " " + d["sol2"]], int(d["label"]))
        )
    return out


def _load_hellaswag(split: str) -> list[Doc]:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split=split)
    out = []
    for d in ds:
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        query = _hellaswag_preprocess(d["activity_label"] + ": " + ctx)
        ends = [" " + _hellaswag_preprocess(e) for e in d["endings"]]
        out.append(Doc([query] * len(ends), ends, int(d["label"])))
    return out


def _load_arc_easy(split: str) -> list[Doc]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    out = []
    for d in ds:
        texts = list(d["choices"]["text"])
        labels = list(d["choices"]["label"])
        if d["answerKey"] not in labels:
            continue  # a few rows are malformed
        ctx = f"Question: {d['question']}\nAnswer:"
        out.append(
            Doc([ctx] * len(texts), [" " + t for t in texts],
                labels.index(d["answerKey"]))
        )
    return out


def _load_winogrande(split: str) -> list[Doc]:
    """Context varies per choice (lm-eval partial evaluation).

    The sentence up to the blank `_`, with the option filled in, is the context and
    the rest is the continuation, so one continuation is scored under two contexts.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
    out = []
    for d in ds:
        idx = d["sentence"].index("_")
        cont = " " + d["sentence"][idx + 1 :].strip()
        ctxs = [d["sentence"][:idx] + opt for opt in (d["option1"], d["option2"])]
        out.append(Doc(ctxs, [cont, cont], int(d["answer"]) - 1))
    return out


#: task name -> (loader, default split).
TASKS: dict[str, tuple[Callable[[str], list[Doc]], str]] = {
    "boolq": (_load_boolq, "validation"),
    "openbookqa": (_load_openbookqa, "test"),
    "piqa": (_load_piqa, "validation"),
    "hellaswag": (_load_hellaswag, "validation"),
    "arc_easy": (_load_arc_easy, "test"),
    "winogrande": (_load_winogrande, "validation"),
}


def load_task(name: str, *, limit: int = 0, split: str = "") -> list[Doc]:
    loader, default_split = TASKS[name]
    docs = loader(split or default_split)
    if limit:
        docs = docs[:limit]
    return docs


# ==========================================================================
# Requests (one per (question, choice) pair)
# ==========================================================================
@dataclass
class Request:
    doc_id: int
    choice_id: int
    ctx_ids: list[int]
    cont_ids: list[int]
    n_bytes: int
    #: Global index over all requests: where the score is written back, so results
    #: can be reassembled after sharding across ranks. (Noise is seeded by `doc_id`.)
    index: int = 0

    @property
    def total_len(self) -> int:
        return len(self.ctx_ids) + len(self.cont_ids)


def build_requests(
    docs: list[Doc], tokenizer, *, max_len: int = 2048, template: str = "none"
) -> list[Request]:
    """Tokenize (question, choice) pairs.

    - `add_special_tokens=False` (no BOS): training blocks are 2048-token windows of
      packed documents, so most positions are mid-document
    - over-long sequences are truncated from the left of the context to `max_len`
      (the continuation is always kept)
    - `template` wraps the context (`PROMPT_TEMPLATES`); the default `"none"` is the
      plain lm-eval format
    """
    wrap = PROMPT_TEMPLATES[template]
    reqs: list[Request] = []
    n = 0
    for di, doc in enumerate(docs):
        for ci, (ctx, cont) in enumerate(zip(doc.contexts, doc.continuations)):
            ctx_ids = tokenizer(wrap.format(ctx=ctx), add_special_tokens=False)["input_ids"]
            cont_ids = tokenizer(cont, add_special_tokens=False)["input_ids"]
            if not cont_ids:  # an empty choice cannot be scored
                cont_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
            keep = max_len - len(cont_ids)
            if keep < 1:
                cont_ids = cont_ids[: max_len - 1]
                keep = 1
            if len(ctx_ids) > keep:
                ctx_ids = ctx_ids[-keep:]
            if not ctx_ids:
                # keep at least one context token: the flow loss convention drops position 0
                raise ValueError("context became empty (max_len is too small)")
            reqs.append(
                Request(di, ci, ctx_ids, cont_ids, len(cont.encode("utf-8")), n)
            )
            n += 1
    return reqs


def _out_size(reqs: list[Request], n_total: int) -> int:
    """Length of the output array: the total number of requests, not `len(reqs)`.

    After sharding, `reqs` is a subset, but scores are written at `Request.index`
    (the global index), so the array must cover all requests.
    """
    if n_total:
        return n_total
    return (max((r.index for r in reqs), default=-1)) + 1


def _length_batches(
    reqs: list[Request], batch_size: int
) -> Iterable[list[Request]]:
    """Batch only requests of equal length (no padding)."""
    by_len: dict[int, list[Request]] = {}
    for r in reqs:
        by_len.setdefault(r.total_len, []).append(r)
    for length in sorted(by_len):
        group = by_len[length]
        for s in range(0, len(group), batch_size):
            yield group[s : s + batch_size]


# ==========================================================================
# Scoring
# ==========================================================================
@torch.no_grad()
def score_flow(
    model,
    reqs: list[Request],
    *,
    alpha_grid: list[float],
    seed: int,
    task_index: int,
    device: torch.device,
    batch_size: int = 32,
    n_total: int = 0,
    prefix_mode: str = "clean",
    noise_draws: int = 1,
    neutral_token: int | None = None,
    progress_every: int = 0,
    time_cond: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> np.ndarray:
    """Flow nll: choice-position CE sum, averaged over α. Returns [n_total].

    Args:
        prefix_mode: ``"clean"`` (default) keeps the prefix at α=1. ``"noised"``
            noises the whole sequence (the training distribution, but the
            question is corrupted too).
        noise_draws: noise draws per α; the second is antithetic (−ε).
        neutral_token: if set, every prefix token is replaced by this id. Length,
            choice positions and noise are unchanged, so this gives the PMI term
            in `nll(choice|prefix) − nll(choice|neutral)`, which cancels the
            per-token term measuring how readily a token is recovered from noise.
        time_cond: map applied to α before the model's time input (default:
            identity); x_t is still built by slerp at α. Only used to score
            external checkpoints with another time parametrization (S-FLM on
            OpenWebText uses σ = −log α; see `run_mcq_external.py`).
    """
    from torchtitan.models.swaflow.model import SphereDiT

    dim = model.model_args.dim
    out = np.zeros(_out_size(reqs, n_total), dtype=np.float64)
    done = 0
    for batch in _length_batches(reqs, batch_size):
        L = batch[0].total_len
        b = len(batch)
        tokens = torch.tensor(
            [r.ctx_ids + r.cont_ids for r in batch], dtype=torch.long, device=device
        )
        # mask of choice positions [B, L]
        cont_mask = torch.zeros((b, L), dtype=torch.bool, device=device)
        for i, r in enumerate(batch):
            cont_mask[i, len(r.ctx_ids) :] = True
        if neutral_token is not None:
            # replace only the prefix by the neutral token (length, positions, noise unchanged)
            tokens = torch.where(
                cont_mask, tokens,
                torch.full_like(tokens, int(neutral_token)),
            )

        x1 = model.get_sphere_embeddings(tokens)  # [B, L, d]
        acc = torch.zeros(b, dtype=torch.float64, device=device)
        prefix_alpha = 1.0 if prefix_mode == "clean" else None
        for ai, a in enumerate(alpha_grid):
            base = torch.empty((b, L, dim), device=device, dtype=torch.float32)
            gen = torch.Generator(device=device)
            for i, r in enumerate(batch):
                # one draw per question, shared by its choices (common random numbers)
                gen.manual_seed(noise_seed(seed, task_index, r.doc_id, ai))
                base[i] = torch.randn(
                    (L, dim), generator=gen, device=device, dtype=torch.float32
                )
            for k in range(noise_draws):
                if k == 0:
                    noise = base
                elif k == 1:
                    noise = -base  # antithetic
                else:
                    gen2 = torch.Generator(device=device)
                    noise = torch.empty_like(base)
                    for i, r in enumerate(batch):
                        gen2.manual_seed(
                            noise_seed(seed + k, task_index, r.doc_id, ai)
                        )
                        noise[i] = torch.randn(
                            (L, dim), generator=gen2, device=device,
                            dtype=torch.float32,
                        )
                x0 = sphere_normalize(noise.to(x1.dtype))
                if prefix_alpha is None:
                    x_t = slerp(x0, x1, float(a))
                else:
                    # per-position α: 1.0 (clean) on the prefix, a on the choice
                    a_pos = torch.where(
                        cont_mask.unsqueeze(-1),
                        torch.tensor(float(a), device=device, dtype=x1.dtype),
                        torch.tensor(prefix_alpha, device=device, dtype=x1.dtype),
                    )
                    x_t = slerp(x0, x1, a_pos)
                alpha = torch.full((b,), float(a), device=device, dtype=torch.float32)
                if time_cond is not None:
                    alpha = time_cond(alpha)
                logits = SphereDiT.forward(model, x_t, alpha)
                # flow convention: logits at position i predict the clean token at position i
                ce = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1).float(),
                    tokens.flatten(0, 1),
                    reduction="none",
                ).view(b, L)
                acc += (ce * cont_mask).sum(dim=1).double()
        acc /= len(alpha_grid) * noise_draws
        for i, r in enumerate(batch):
            out[r.index] = float(acc[i])
        done += b
        if progress_every and done % progress_every < b:
            print(f"    [mcq/flow] {done}/{len(reqs)}", flush=True)
    return out


@torch.no_grad()
def score_ar(
    model,
    reqs: list[Request],
    *,
    device: torch.device,
    batch_size: int = 32,
    n_total: int = 0,
    progress_every: int = 0,
) -> np.ndarray:
    """nll for the AR twins. `logits[:, j]` predicts position j+1."""
    out = np.zeros(_out_size(reqs, n_total), dtype=np.float64)
    done = 0
    for batch in _length_batches(reqs, batch_size):
        L = batch[0].total_len
        b = len(batch)
        tokens = torch.tensor(
            [r.ctx_ids + r.cont_ids for r in batch], dtype=torch.long, device=device
        )
        logits = model(tokens)
        ce = torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1).float(),
            tokens[:, 1:].flatten(0, 1),
            reduction="none",
        ).view(b, L - 1)
        mask = torch.zeros((b, L - 1), dtype=torch.bool, device=device)
        for i, r in enumerate(batch):
            # position j predicts tokens[j+1], so the first choice token
            # (index len(ctx)) is predicted at j = len(ctx) - 1
            mask[i, len(r.ctx_ids) - 1 :] = True
        vals = (ce * mask).sum(dim=1).double()
        for i, r in enumerate(batch):
            out[r.index] = float(vals[i])
        done += b
        if progress_every and done % progress_every < b:
            print(f"    [mcq/ar] {done}/{len(reqs)}", flush=True)
    return out


@torch.no_grad()
def score_hf(
    model,
    reqs: list[Request],
    *,
    device: torch.device,
    batch_size: int = 16,
    n_total: int = 0,
    progress_every: int = 0,
) -> np.ndarray:
    """nll for a released HF causal LM (e.g. SmolLM2); used to check the prompt formats."""
    out = np.zeros(_out_size(reqs, n_total), dtype=np.float64)
    done = 0
    for batch in _length_batches(reqs, batch_size):
        L = batch[0].total_len
        b = len(batch)
        tokens = torch.tensor(
            [r.ctx_ids + r.cont_ids for r in batch], dtype=torch.long, device=device
        )
        logits = model(tokens).logits
        ce = torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1).float(),
            tokens[:, 1:].flatten(0, 1),
            reduction="none",
        ).view(b, L - 1)
        mask = torch.zeros((b, L - 1), dtype=torch.bool, device=device)
        for i, r in enumerate(batch):
            mask[i, len(r.ctx_ids) - 1 :] = True
        vals = (ce * mask).sum(dim=1).double()
        for i, r in enumerate(batch):
            out[r.index] = float(vals[i])
        done += b
        if progress_every and done % progress_every < b:
            print(f"    [mcq/hf] {done}/{len(reqs)}", flush=True)
    return out


# ==========================================================================
# Accuracy
# ==========================================================================
def score_accuracy(
    docs: list[Doc], reqs: list[Request], nll: np.ndarray
) -> dict[str, float]:
    """Accuracy under three normalizations (the choice with the smallest nll wins)."""
    per_doc: dict[int, list[tuple[int, float, int, int]]] = {}
    for r in reqs:
        per_doc.setdefault(r.doc_id, []).append(
            (r.choice_id, float(nll[r.index]), len(r.cont_ids), r.n_bytes)
        )
    hits = {"acc": 0, "acc_norm_tok": 0, "acc_norm_byte": 0}
    n = 0
    for di, items in per_doc.items():
        gold = docs[di].gold
        items.sort()
        raw = [v for _, v, _, _ in items]
        tok = [v / max(1, t) for _, v, t, _ in items]
        byt = [v / max(1, b) for _, v, _, b in items]
        hits["acc"] += int(int(np.argmin(raw)) == gold)
        hits["acc_norm_tok"] += int(int(np.argmin(tok)) == gold)
        hits["acc_norm_byte"] += int(int(np.argmin(byt)) == gold)
        n += 1
    out = {k: v / max(1, n) for k, v in hits.items()}
    out["n_docs"] = float(n)
    # binomial standard error
    for k in ("acc", "acc_norm_tok", "acc_norm_byte"):
        p = out[k]
        out[f"{k}_stderr"] = math.sqrt(max(p * (1 - p), 0.0) / max(1, n))
    return out
