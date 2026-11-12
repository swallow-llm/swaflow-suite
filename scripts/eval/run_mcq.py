# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Run the MCQ evaluation (`mcq.py`) over a run's checkpoints or on a released model.

    # flow (setting of the reported accuracies: α = 0, PMI correction)
    torchrun --nproc_per_node=8 -m scripts.eval.run_mcq \\
        --dump <dump> --kind flow --flavor 3B --alpha-max 0.10 --alpha-list 0 --pmi \\
        --every-tokens 100000000000

    # AR twin (standard log-probability)
    torchrun --nproc_per_node=8 -m scripts.eval.run_mcq \\
        --dump <dump> --kind ar --flavor 1B --every-tokens 100000000000

    # check the prompt formats on a released AR model
    python -m scripts.eval.run_mcq --kind hf --hf-model HuggingFaceTB/SmolLM2-135M

Requests are sharded across ranks and the model is replicated, since sequence
length varies per question.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.eval.ckpt_eval import COMMON_GRID, GRIDS, _cast_params
from scripts.eval.loader import (
    build_eval_model,
    checkpoint_steps,
    load_checkpoint_distributed,
    select_steps,
)
from scripts.eval import mcq as mcqlib


DEFAULT_TASKS = "boolq,openbookqa,piqa,hellaswag,arc_easy,winogrande"


def _tokenizer():
    from transformers import AutoTokenizer

    path = os.environ.get("SWAFLOW_TOKENIZER_DIR")
    if not path:
        raise SystemExit("SWAFLOW_TOKENIZER_DIR is not set")
    return AutoTokenizer.from_pretrained(path)


def _prepare(tasks: list[str], tokenizer, limit: int, max_len: int, rank: int,
             template: str = "none"):
    """Load tasks and build requests; rank 0 goes first to warm the dataset cache."""
    if rank != 0 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    out = {}
    for ti, name in enumerate(tasks):
        docs = mcqlib.load_task(name, limit=limit)
        reqs = mcqlib.build_requests(docs, tokenizer, max_len=max_len,
                                     template=template)
        out[name] = (ti, docs, reqs)
    if rank == 0 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    return out


def _shard(reqs, world: int, rank: int):
    return reqs[rank::world] if world > 1 else reqs


def _gather(local: np.ndarray, n: int, device) -> np.ndarray:
    if not (dist.is_available() and dist.is_initialized()):
        return local
    t = torch.from_numpy(local).to(device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.cpu().numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default="", help="dump_folder of the run (not needed for hf)")
    ap.add_argument("--kind", choices=["flow", "ar", "hf"], required=True)
    ap.add_argument("--flavor", default="")
    ap.add_argument("--alpha-max", type=float, default=None)
    ap.add_argument("--hf-model", default="", help="model name for --kind hf")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--grid", default="common", choices=sorted(GRIDS))
    ap.add_argument("--alpha-list", default="",
                    help="explicit α values, colon/comma separated (overrides --grid)")
    ap.add_argument("--limit", type=int, default=0, help="max number of questions per task (for debugging)")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prefix-mode", choices=["clean", "noised"], default="clean",
                    help="clean (default): prefix held at α=1 / "
                         "noised: whole sequence noised at α (the training distribution)")
    ap.add_argument("--noise-draws", type=int, default=1,
                    help="noise draws per α (the second is antithetic)")
    ap.add_argument("--pmi", action="store_true",
                    help="PMI correction: subtract the nll of the same sequence with the "
                         "prefix replaced by a neutral token, which cancels the per-token "
                         "term measuring how readily a token is recovered from noise")
    ap.add_argument("--neutral-token", type=int, default=198,
                    help="token id that fills the prefix for --pmi (default: newline)")
    ap.add_argument("--prompt-template", default="none",
                    choices=sorted(mcqlib.PROMPT_TEMPLATES),
                    help="context wrapper: none (default) is the plain lm-eval "
                         "format; sft is the SFT format "
                         "(User:/Assistant:)")
    ap.add_argument("--out-name", default="mcq")
    ap.add_argument("--out", default="", help="output JSON for --kind hf")
    ap.add_argument("--steps", default="")
    ap.add_argument("--every-tokens", type=int, default=100_000_000_000)
    ap.add_argument("--tokens-per-step", type=int, default=8_388_608)
    ap.add_argument("--folder", default="checkpoint")
    ap.add_argument("--no-qk-norm", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--time-budget-sec", type=int, default=0)
    args = ap.parse_args(argv)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    tasks = [t for t in args.tasks.split(",") if t.strip()]
    if args.alpha_list:
        alpha_grid = [
            float(x) for x in args.alpha_list.replace(":", ",").split(",") if x.strip()
        ]
    else:
        alpha_grid = GRIDS[args.grid]

    tok = _tokenizer()
    prepared = _prepare(tasks, tok, args.limit, args.max_len, rank,
                        template=args.prompt_template)
    if is_main:
        for name, (_, docs, reqs) in prepared.items():
            print(f"[mcq] {name}: {len(docs)} docs / {len(reqs)} requests", flush=True)

    # ---------------- released HF model (prompt-format check) ----------------
    if args.kind == "hf":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, dtype=torch.float32
        ).to(device).eval()
        results = {}
        for name, (ti, docs, reqs) in prepared.items():
            t0 = time.time()
            nll = mcqlib.score_hf(model, reqs, device=device,
                                  batch_size=args.batch, n_total=len(reqs))
            results[name] = mcqlib.score_accuracy(docs, reqs, nll)
            print(f"[mcq/hf] {name}: {results[name]} ({time.time() - t0:.0f}s)",
                  flush=True)
        payload = {"kind": "hf", "model": args.hf_model, "tasks": results,
                   "mean_acc_norm_tok": float(
                       np.mean([r["acc_norm_tok"] for r in results.values()]))}
        out = args.out or f"mcq_hf_{Path(args.hf_model).name}.json"
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")
        return 0

    # ---------------- trained checkpoints ----------------
    dump = Path(args.dump)
    out_dir = dump / args.out_name
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if args.steps:
        steps = [int(s) for s in args.steps.replace(":", ",").split(",") if s.strip()]
    else:
        steps = select_steps(
            checkpoint_steps(dump, folder=args.folder),
            every_tokens=args.every_tokens,
            tokens_per_step=args.tokens_per_step,
        )
    if not args.overwrite:
        steps = [s for s in steps if not (out_dir / f"step-{s}.json").exists()]
    if is_main:
        print(f"[mcq] {dump.name} / {args.kind} {args.flavor}: {len(steps)} ckpts "
              f"-> {out_dir}  grid={alpha_grid if args.kind == 'flow' else 'n/a'}",
              flush=True)
    if not steps:
        if world > 1:
            dist.destroy_process_group()
        return 0

    model = build_eval_model(
        args.kind, args.flavor,
        alpha_max=args.alpha_max, qk_norm=not args.no_qk_norm,
        max_seq_len=args.max_len,
    )
    model.to(device=device)
    _cast_params(model, torch.float32)
    model.eval()

    t_start = time.time()
    for i, step in enumerate(steps):
        if args.time_budget_sec and time.time() - t_start > args.time_budget_sec:
            if is_main:
                print(f"[mcq] time budget exceeded; stopping with {len(steps) - i} steps left", flush=True)
            break
        t0 = time.time()
        _cast_params(model, torch.float32)
        load_checkpoint_distributed(
            model, dump / args.folder / f"step-{step}", device=device
        )
        _cast_params(model, torch.bfloat16)

        results = {}
        for name, (ti, docs, reqs) in prepared.items():
            mine = _shard(reqs, world, rank)
            if args.kind == "flow":
                local = mcqlib.score_flow(
                    model, mine, alpha_grid=alpha_grid, seed=args.seed,
                    task_index=ti, device=device, batch_size=args.batch,
                    n_total=len(reqs), prefix_mode=args.prefix_mode,
                    noise_draws=args.noise_draws,
                )
                if args.pmi:
                    local = local - mcqlib.score_flow(
                        model, mine, alpha_grid=alpha_grid, seed=args.seed,
                        task_index=ti, device=device, batch_size=args.batch,
                        n_total=len(reqs), prefix_mode=args.prefix_mode,
                        noise_draws=args.noise_draws,
                        neutral_token=args.neutral_token,
                    )
            else:
                local = mcqlib.score_ar(
                    model, mine, device=device, batch_size=args.batch,
                    n_total=len(reqs),
                )
            # Scorers write into a full-length array at `Request.index`, so each
            # rank is nonzero only on its share and all_reduce(SUM) merges them.
            nll = _gather(local, len(reqs), device)
            results[name] = mcqlib.score_accuracy(docs, reqs, nll)

        if is_main:
            mean_tok = float(np.mean([r["acc_norm_tok"] for r in results.values()]))
            payload = {
                "step": step,
                "kind": args.kind,
                "flavor": args.flavor,
                "alpha_grid": alpha_grid if args.kind == "flow" else [],
                "alpha_max": args.alpha_max,
                "seed": args.seed,
                "prefix_mode": args.prefix_mode,
                "prompt_template": args.prompt_template,
                "noise_draws": args.noise_draws,
                "pmi": args.pmi,
                "tasks": results,
                "mean_acc": float(np.mean([r["acc"] for r in results.values()])),
                "mean_acc_norm_tok": mean_tok,
                "mean_acc_norm_byte": float(
                    np.mean([r["acc_norm_byte"] for r in results.values()])),
                "elapsed_sec": time.time() - t0,
            }
            (out_dir / f"step-{step}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False)
            )
            per = "  ".join(
                f"{n}={r['acc_norm_tok']:.3f}" for n, r in results.items()
            )
            print(f"[mcq] step {step:>7}  mean {mean_tok:.4f}  {per}  "
                  f"({time.time() - t0:.0f}s) [{i + 1}/{len(steps)}]", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
