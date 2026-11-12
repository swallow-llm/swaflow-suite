# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""GSM8K: generate one completion per problem and score the extracted answer.

`--scoring code` (TinyGSM-finetuned models): 0-shot prompt ``question + "\\n"``,
the TinyGSM training format also used by S-FLM; the generated Python is executed
and its return value compared with the gold answer (`code_exec.py`). `ran_rate`
is the executable rate.

`--scoring text`: k-shot prompts in the lm-evaluation-harness v0.4 `gsm8k` format
(``Question: {question}\\nAnswer: {answer}`` joined by ``\\n\\n``), scored with its
`strict-match` / `flexible-extract` filters and `exact_match` normalization (no
partial credit). The few-shot examples are one fixed set drawn from the train
split with a fixed seed, shared by all problems and all models.

The α grid comes from the model's schedule, so sampling stays inside the training
window. Defaults: NFE 64, temperature 1.0, exact velocity, greedy final decode,
fp32 weights. Noise is seeded per problem (`seed*1000003 + doc_id`), so results do
not depend on how problems are sharded across ranks. Stop strings are applied to
the decoded text, since a flow fills the whole sequence at once.

    torchrun --nproc_per_node=8 -m scripts.eval.gsm8k \\
        --dump $SWAFLOW_OUTPUTS/<run> --kind flow --flavor 3B --alpha-max 0.10 \\
        --scoring code --num-fewshot 0
    # check prompt format and extraction on a released AR model
    python -m scripts.eval.gsm8k --kind hf --hf-model HuggingFaceTB/SmolLM2-1.7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.eval.ckpt_eval import _cast_params
from scripts.eval.loader import (
    build_eval_model,
    checkpoint_steps,
    load_checkpoint_distributed,
    select_steps,
)


#: lm-eval `until`; applied after decoding, since a flow generates the whole sequence at once.
STOP_STRINGS = ("Question:", "<|endoftext|>", "</s>", "<|im_end|>")

#: The two lm-eval `gsm8k` filters.
RE_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
RE_FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

#: lm-eval `exact_match` `regexes_to_ignore` (as in gsm8k.yaml).
IGNORE_PATTERNS = [re.compile(p) for p in (r",", r"\$", r"(?s).*#### ", r"\.$")]

#: Prompt templates; `sft` is the SFT format (`plain-user-assistant-v1` in `build_sft.py`).
TEMPLATES = {
    "plain": "{prompt}",
    "sft": "User: {prompt}\n\nAssistant:",
}

#: Stop strings for `--scoring code`. A TinyGSM example is
#: `question \n code <|endoftext|>`, so EOS alone ends the program.
STOP_STRINGS_CODE = ("<|endoftext|>", "</s>", "<|im_end|>")

#: Signature appended to the prompt with `--code-prefill`, so that generation starts
#: from the function body (every TinyGSM response begins with this line). This
#: departs from S-FLM's scoring setup.
CODE_SIGNATURE = "def simple_math_problem() -> int:\n"


# ==========================================================================
# Prompt format and answer extraction
# ==========================================================================
def load_gsm8k(split: str = "test", limit: int = 0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    docs = [{"question": d["question"], "answer": d["answer"]} for d in ds]
    return docs[:limit] if limit else docs


def fewshot_prefix(k: int, seed: int) -> str:
    """Fixed k-shot prefix: one set of examples drawn from the train split."""
    if k <= 0:
        return ""
    train = load_gsm8k("train")
    idx = random.Random(seed).sample(range(len(train)), k)
    shots = [
        f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}" for i in idx
    ]
    return "\n\n".join(shots) + "\n\n"


def build_prompt(prefix: str, doc: dict, template: str = "plain",
                 scoring: str = "text", prefill: bool = False) -> str:
    """Build the prompt for one problem.

    With `scoring="code"` the prompt is the TinyGSM training format
    ``question + "\n"``, the same 0-shot prompt as S-FLM
    (`dataloader.get_tiny_gsm_dataset`), with no few-shot prefix and no
    `Question:` / `Answer:`.
    """
    if scoring == "code":
        if prefix:
            raise ValueError("--scoring code is 0-shot only (--num-fewshot 0)")
        body = doc["question"].strip() + "\n"
        if prefill:
            body += CODE_SIGNATURE
        return TEMPLATES[template].format(prompt=body)
    body = f"{prefix}Question: {doc['question']}\nAnswer:"
    return TEMPLATES[template].format(prompt=body)


def cut_at_stop(text: str, scoring: str = "text") -> str:
    """Apply lm-eval `until` to the decoded text (cut at the first match)."""
    cut = len(text)
    for s in (STOP_STRINGS_CODE if scoring == "code" else STOP_STRINGS):
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut]


def _normalize(s: str) -> str:
    """lm-eval `exact_match` normalization (`regexes_to_ignore` + lowercase)."""
    for pat in IGNORE_PATTERNS:
        s = pat.sub("", s)
    return s.strip().lower()


def extract_strict(text: str) -> str:
    m = RE_STRICT.search(text)
    return m.group(1) if m else ""


def extract_flexible(text: str) -> str:
    """`group_select: -1`, i.e. the last match (lm-eval flexible-extract)."""
    matches = list(RE_FLEXIBLE.finditer(text))
    if not matches:
        return ""
    m = matches[-1]
    return m.group(1) if m.group(1) is not None else m.group(2)


def score_one(completion: str, gold_answer: str, scoring: str = "text",
              exec_timeout: float = 2.0, prefill: bool = False) -> dict:
    """Score one problem; no partial credit (as lm-eval exact_match).

    `scoring="code"` executes the generated Python and compares its return value
    (`scripts/eval/code_exec.py`, the same scoring as S-FLM).
    """
    if scoring == "code":
        from scripts.eval.code_exec import score_code

        text = cut_at_stop(completion, scoring="code")
        # The prefilled signature is not part of the completion; restore it before scoring.
        r = score_code((CODE_SIGNATURE + text) if prefill else text,
                       gold_answer, timeout_s=exec_timeout)
        return {
            "text": text,
            "gold": "" if r["gold_value"] is None else str(r["gold_value"]),
            "pred_strict": "" if r["pred_value"] is None else str(r["pred_value"]),
            "pred_flexible": "" if r["pred_value"] is None else str(r["pred_value"]),
            "strict": r["strict"],
            "flexible": r["flexible"],
            # In code mode, `has_marker` means the entry point was defined.
            "has_marker": r["has_entry"],
            "compiled": r["compiled"],
            "ran": r["ran"],
            "error": r["error"],
            "n_chars": len(text),
        }
    text = cut_at_stop(completion)
    gold = _normalize(gold_answer)
    strict = _normalize(extract_strict(text))
    flexible = _normalize(extract_flexible(text))
    return {
        "text": text,
        "gold": gold,
        "pred_strict": strict,
        "pred_flexible": flexible,
        "strict": bool(strict) and strict == gold,
        "flexible": bool(flexible) and flexible == gold,
        "has_marker": "####" in text,
        "n_chars": len(text),
    }


# ==========================================================================
# Flow generation
# ==========================================================================
def _noise_for(doc_ids, seq_len: int, dim: int, seed: int) -> torch.Tensor:
    """Initial noise seeded per problem (independent of rank sharding)."""
    out = torch.empty((len(doc_ids), seq_len, dim), dtype=torch.float32)
    for i, d in enumerate(doc_ids):
        g = torch.Generator(device="cpu").manual_seed(seed * 1000003 + int(d))
        out[i] = torch.randn((seq_len, dim), generator=g, dtype=torch.float32)
    return out


@torch.no_grad()
def generate_flow(
    model,
    prompts_ids: list[list[int]],
    doc_ids: list[int],
    *,
    seq_len: int,
    nfe: int,
    temperature: float,
    seed: int,
    device: torch.device,
    micro_batch: int = 8,
    chunk_size: int = 16,
    velocity: str = "exact",
    top_k_velocity: int = -1,
) -> list[list[int]]:
    """Generate full sequences with the prefix pinned; return the tokens after the prefix."""
    from torchtitan.models.swaflow.sampling import SamplerConfig, generate

    cfg = SamplerConfig(
        steps=nfe,
        temperature=temperature,
        velocity=velocity,
        top_k_velocity=top_k_velocity,
        noise_removal="greedy",
        chunk_size=chunk_size,
    )
    dim = model.model_args.dim
    out: list[list[int]] = []
    for s in range(0, len(prompts_ids), micro_batch):
        chunk = prompts_ids[s : s + micro_batch]
        ids = doc_ids[s : s + micro_batch]
        p_len = max(len(c) for c in chunk)
        pref = torch.zeros((len(chunk), p_len), dtype=torch.long)
        for i, c in enumerate(chunk):
            pref[i, : len(c)] = torch.tensor(c, dtype=torch.long)
        lens = torch.tensor([len(c) for c in chunk], dtype=torch.long)
        noise = _noise_for(ids, seq_len, dim, seed).to(device)
        toks = generate(
            model,
            config=cfg,
            noise=noise,
            prefix_tokens=pref.to(device),
            prefix_lengths=lens.to(device),
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        seqs = toks.cpu().tolist()
        for i, c in enumerate(chunk):
            out.append(seqs[i][len(c) :])
    return out


@torch.no_grad()
def generate_hf(model, tok, prompts: list[str], *, max_new_tokens: int,
                device: torch.device, batch_size: int = 8) -> list[str]:
    """Released AR model (harness sanity check). Greedy, left padding."""
    outs: list[str] = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(prompts), batch_size):
        batch = prompts[s : s + batch_size]
        enc = [tok(p, add_special_tokens=False)["input_ids"] for p in batch]
        width = max(len(e) for e in enc)
        ids = torch.full((len(enc), width), pad, dtype=torch.long)
        mask = torch.zeros((len(enc), width), dtype=torch.long)
        for i, e in enumerate(enc):  # left padding: generation continues from the right end
            ids[i, width - len(e) :] = torch.tensor(e, dtype=torch.long)
            mask[i, width - len(e) :] = 1
        gen = model.generate(
            input_ids=ids.to(device), attention_mask=mask.to(device),
            max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad,
        )
        for i in range(len(enc)):
            outs.append(tok.decode(gen[i, width:], skip_special_tokens=True))
    return outs


# ==========================================================================
# Aggregation
# ==========================================================================
def degeneracy(text: str) -> tuple[float, int]:
    """Degeneracy of the generated text, e.g. a budget filled with one repeated token.

    Returns `(share of the most frequent character, longest run of one character)`.
    """
    if not text:
        return 0.0, 0
    counts: dict[str, int] = {}
    best_run, run = 1, 0
    prev = text[0]
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
        if ch == prev:
            run += 1
            if run > best_run:
                best_run = run
        else:
            run = 1
            prev = ch
    return max(counts.values()) / len(text), best_run


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def _mean_se(key):
        p = float(np.mean([r[key] for r in rows]))
        return p, float(np.sqrt(max(p * (1.0 - p), 0.0) / n))

    acc_s, se_s = _mean_se("strict")
    acc_f, se_f = _mean_se("flexible")
    _degs = [degeneracy(r.get("text", "")) for r in rows]
    return {
        "n": n,
        "acc_strict": acc_s,
        "se_strict": se_s,
        "acc_flexible": acc_f,
        "se_flexible": se_f,
        "has_marker_rate": float(np.mean([r["has_marker"] for r in rows])),
        "empty_pred_rate": float(np.mean([not r["pred_flexible"] for r in rows])),
        "mean_chars_until_stop": float(np.mean([r["n_chars"] for r in rows])),
        # Degeneracy: `degenerate_rate` is the fraction of problems whose most
        # frequent character covers at least 90% of the generated text.
        "mean_top_char_share": float(np.mean([_deg[0] for _deg in _degs])),
        "degenerate_rate": float(np.mean([_deg[0] >= 0.9 for _deg in _degs])),
        "mean_max_char_run": float(np.mean([_deg[1] for _deg in _degs])),
        # `--scoring code` only: separates "not valid code" from "ran but wrong
        # answer". `ran_rate` is the executable rate.
        **({
            "compiled_rate": float(np.mean([r["compiled"] for r in rows])),
            "ran_rate": float(np.mean([r["ran"] for r in rows])),
        } if "compiled" in rows[0] else {}),
    }


def _gather_rows(rows: list[dict]) -> list[dict]:
    if not (dist.is_available() and dist.is_initialized()):
        return rows
    buf = [None] * dist.get_world_size()
    dist.all_gather_object(buf, rows)
    merged = [r for part in buf for r in part]
    merged.sort(key=lambda r: r["doc_id"])
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default="", help="dump_folder of the run (not needed for hf)")
    ap.add_argument("--kind", choices=["flow", "hf"], required=True)
    ap.add_argument("--flavor", default="")
    ap.add_argument("--alpha-max", type=float, default=None)
    ap.add_argument("--hf-model", default="")
    ap.add_argument("--num-fewshot", type=int, default=5,
                    help="keep fixed across all models compared")
    ap.add_argument("--fewshot-seed", type=int, default=1234)
    ap.add_argument("--template", default="plain", choices=sorted(TEMPLATES))
    ap.add_argument("--scoring", default="text", choices=["text", "code"],
                    help="text = lm-eval regex extraction (default) / "
                         "code = execute the generated Python and compare"
                         " (0-shot only; same scoring as S-FLM)")
    ap.add_argument("--code-prefill", action="store_true",
                    help="append `def simple_math_problem() -> int:` to the prompt "
                         "so generation starts from the function body; "
                         "departs from S-FLM's scoring setup")
    ap.add_argument("--exec-timeout", type=float, default=2.0,
                    help="per-problem execution limit for --scoring code (seconds)")
    ap.add_argument("--limit", type=int, default=0, help="max number of problems (0 = all 1319)")
    ap.add_argument("--seq-len", type=int, default=1536,
                    help="generated sequence length, including the prefix")
    ap.add_argument("--max-seq-len", type=int, default=2048, help="RoPE table length")
    ap.add_argument("--nfe", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--velocity", default="exact", choices=["exact", "sample"],
                    help="velocity computation; exact = marginalize over the full vocabulary (default)")
    ap.add_argument("--top-k-velocity", type=int, default=-1,
                    help="number of vocabulary entries used for the velocity "
                         "(-1 = all; exposed as an option in S-FLM too). "
                         "k=1 aims the velocity at the arg-max embedding")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--max-new-tokens", type=int, default=384, help="for --kind hf")
    ap.add_argument("--hf-batch", type=int, default=8)
    ap.add_argument("--out-name", default="gsm8k")
    ap.add_argument("--out", default="", help="output JSON for --kind hf")
    ap.add_argument("--steps", default="")
    ap.add_argument("--every-tokens", type=int, default=0)
    ap.add_argument("--tokens-per-step", type=int, default=8_388_608)
    ap.add_argument("--folder", default="checkpoint")
    ap.add_argument("--no-qk-norm", action="store_true")
    ap.add_argument("--save-samples", type=int, default=32,
                    help="number of samples saved in the JSON (0 = none)")
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

    tok_dir = os.environ.get("SWAFLOW_TOKENIZER_DIR")
    from transformers import AutoTokenizer

    if args.kind == "hf":
        tok = AutoTokenizer.from_pretrained(args.hf_model)
    else:
        if not tok_dir:
            raise SystemExit("SWAFLOW_TOKENIZER_DIR is not set")
        tok = AutoTokenizer.from_pretrained(tok_dir)

    docs = load_gsm8k("test", limit=args.limit)
    if args.scoring == "code" and args.num_fewshot:
        raise SystemExit("--scoring code is 0-shot only; pass --num-fewshot 0")
    prefix = fewshot_prefix(args.num_fewshot, args.fewshot_seed)
    prompts = [build_prompt(prefix, d, args.template, args.scoring,
                            args.code_prefill) for d in docs]

    # ---------------- released HF model (harness sanity check) ----------------
    if args.kind == "hf":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, dtype=torch.float32
        ).to(device).eval()
        t0 = time.time()
        comps = generate_hf(model, tok, prompts, max_new_tokens=args.max_new_tokens,
                            device=device, batch_size=args.hf_batch)
        rows = []
        for i, (c, d) in enumerate(zip(comps, docs)):
            r = score_one(c, d["answer"], args.scoring, args.exec_timeout,
                          args.code_prefill)
            r["doc_id"] = i
            rows.append(r)
        payload = {
            "kind": "hf", "model": args.hf_model,
            "num_fewshot": args.num_fewshot, "fewshot_seed": args.fewshot_seed,
            "template": args.template, "scoring": args.scoring,
            "max_new_tokens": args.max_new_tokens,
            **summarize(rows),
            "samples": [
                {k: r[k] for k in ("doc_id", "gold", "pred_strict",
                                   "pred_flexible", "text")}
                for r in rows[: args.save_samples]
            ],
            "elapsed_sec": time.time() - t0,
        }
        out = args.out or f"gsm8k_hf_{Path(args.hf_model).name}.json"
        Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[gsm8k/hf] {args.hf_model}: strict {payload['acc_strict']:.4f} "
              f"flexible {payload['acc_flexible']:.4f} (n={payload['n']}) "
              f"-> {out}", flush=True)
        return 0

    # ---------------- flow ----------------
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
    if not steps:
        if world > 1:
            dist.destroy_process_group()
        return 0

    prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    too_long = [i for i, p in enumerate(prompt_ids) if len(p) >= args.seq_len - 32]
    if too_long:
        raise SystemExit(
            f"prefix too long for {len(too_long)} problems (--seq-len {args.seq_len},"
            f" longest {max(len(prompt_ids[i]) for i in too_long)}). "
            "Increase --seq-len or reduce --num-fewshot"
        )
    mine = list(range(rank, len(docs), world)) if world > 1 else list(range(len(docs)))
    if is_main:
        lens = [len(p) for p in prompt_ids]
        print(f"[gsm8k] {len(docs)} problems / {args.num_fewshot}-shot / "
              f"prompt {min(lens)}-{max(lens)} tok / seq_len {args.seq_len} / "
              f"world {world} -> {out_dir}", flush=True)

    model = build_eval_model(
        "flow", args.flavor, alpha_max=args.alpha_max,
        qk_norm=not args.no_qk_norm, max_seq_len=args.max_seq_len,
    )
    model.to(device=device)
    _cast_params(model, torch.float32)
    model.eval()

    t_start = time.time()
    for i, step in enumerate(steps):
        if args.time_budget_sec and time.time() - t_start > args.time_budget_sec:
            if is_main:
                print(f"[gsm8k] time budget exceeded; stopping with {len(steps) - i} steps left",
                      flush=True)
            break
        t0 = time.time()
        _cast_params(model, torch.float32)
        load_checkpoint_distributed(
            model, dump / args.folder / f"step-{step}", device=device
        )
        # Generate in fp32, matching the other generation evaluations.
        comps = generate_flow(
            model, [prompt_ids[j] for j in mine], mine,
            seq_len=args.seq_len, nfe=args.nfe, temperature=args.temperature,
            seed=args.seed, device=device, micro_batch=args.micro_batch,
            chunk_size=args.chunk_size, velocity=args.velocity,
            top_k_velocity=args.top_k_velocity,
        )
        rows = []
        for j, c in zip(mine, comps):
            r = score_one(tok.decode(c, skip_special_tokens=False),
                          docs[j]["answer"], args.scoring, args.exec_timeout,
                          args.code_prefill)
            r["doc_id"] = j
            rows.append(r)
        rows = _gather_rows(rows)

        if is_main:
            payload = {
                "step": step, "kind": "flow", "flavor": args.flavor,
                "alpha_max": args.alpha_max,
                "num_fewshot": args.num_fewshot,
                "fewshot_seed": args.fewshot_seed,
                "template": args.template, "scoring": args.scoring,
                "code_prefill": args.code_prefill,
                "velocity": args.velocity,
                "top_k_velocity": args.top_k_velocity,
                "seq_len": args.seq_len, "nfe": args.nfe,
                "temperature": args.temperature, "seed": args.seed,
                **summarize(rows),
                "samples": [
                    {k: r[k] for k in ("doc_id", "gold", "pred_strict",
                                       "pred_flexible", "text")}
                    for r in rows[: args.save_samples]
                ],
                "elapsed_sec": time.time() - t0,
            }
            (out_dir / f"step-{step}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False)
            )
            print(f"[gsm8k] step {step:>7}  strict {payload['acc_strict']:.4f}"
                  f"±{payload['se_strict']:.4f}  flexible "
                  f"{payload['acc_flexible']:.4f}±{payload['se_flexible']:.4f}  "
                  f"marker {payload['has_marker_rate']:.3f}  "
                  f"({time.time() - t0:.0f}s) [{i + 1}/{len(steps)}]", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
