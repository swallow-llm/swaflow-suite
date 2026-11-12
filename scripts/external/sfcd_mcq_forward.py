# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# Run MCQ sequences through the released SFCD checkpoint (HF `Jugc/SFCD`) and
# dump the summed CE at the choice positions. Runs inside the SFCD code's own
# environment and uses none of our code; tensors are exchanged with
# `mcq_via_dump.py` through files.
#
#   <sfcd venv>/bin/python sfcd_mcq_forward.py --sfcd-src <SFCD>/src \
#       --config <config.json> --ckpt <final.pt> \
#       --requests <output of mcq_via_dump export> --out <ce.pt>
#
# Model construction and weight loading (EMA by default, as in the SFCD
# evaluation) call SFCD's own `load_experiment` (`src/scripts/eval/eval_owt.py`);
# the vMF path, self-conditioning and time warp are not reimplemented.
#
# In the main setting (clean prefix, α=0) the forward receives exactly what the
# first step of the SFCD sampler does:
#   - choice positions: uniform noise on the sphere (vMF with κ=0, i.e. a
#     normalized Gaussian, as in `SphericalSampler.sample`)
#   - time input: σ = κ/κ_max = 0 (SFCD `_sigma`)
#   - self-conditioning: no input (zeros, as in the sampler's first step)
#   - prefix positions: clean embeddings (`SphericalPath.get_target_embeddings`)
#   - no clue_mask: OWT training used no clues and `clue_embed` is still at its
#     initial value (norm ≈ √768), so passing it would add a large vector never
#     seen in training. The prefix only places embeddings, as in our harness.
# Only the summed choice CE is returned; the PMI difference, length
# normalization and standard errors are computed on our side
# (`scripts/external/mcq_via_dump.py score`).

import argparse
import importlib.util
import os
import sys
import time

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--sfcd-src", required=True)
p.add_argument("--config", required=True)
p.add_argument("--ckpt", required=True)
p.add_argument("--requests", required=True)
p.add_argument("--out", required=True)
p.add_argument("--no-ema", action="store_true")
p.add_argument("--tasks", default="")
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()

sys.path.insert(0, args.sfcd_src)
spec = importlib.util.spec_from_file_location(
    "sfcd_eval_owt", os.path.join(args.sfcd_src, "scripts", "eval", "eval_owt.py"))
eval_owt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_owt)
from methods.continuous.spherical.path import SphericalPath  # noqa: E402  (SFCD)
from model import backbone as sfcd_backbone  # noqa: E402  (SFCD)

device = torch.device("cuda")
cfg, model, _psi, _cdf, _warp, step = eval_owt.load_experiment(
    args.config, device, checkpoint_path=args.ckpt, use_ema=not args.no_ema)
model.eval()
assert cfg.flow.noise_process == "vmf", cfg.flow.noise_process
assert model.time_conditioning and model.self_conditioning
D = model.embed_dim

req = torch.load(args.requests, weights_only=True)
neutral = int(req["meta"]["neutral_token"])
tasks = [t for t in (args.tasks or ",".join(req["meta"]["tasks"])).split(",") if t]

out = {
    "meta": {
        "source": "HF Jugc/SFCD owt/vmf_tc_selfcond_scp025/final.pt",
        "step": int(step),
        "weights": "raw" if args.no_ema else "ema",
        "noise_process": cfg.flow.noise_process,
        "kappa_max": float(cfg.flow.kappa_max),
        "self_conditioning": bool(model.self_conditioning),
        "learned_tau": bool(model.learned_tau),
        "vocab_size": int(model.vocab_size),
        "sequence_length": int(model.sequence_length),
        "has_flash_attn": bool(sfcd_backbone.HAS_FLASH_ATTN),
        "time_input": "sigma = kappa/kappa_max = 0 (uniform vMF, sampler step 1)",
        "self_cond_input": "none (zeros, sampler step 1)",
        "clue_mask": "none (OWT is trained without clues)",
        "torch": str(torch.__version__),
    },
    "tasks": {},
}

with torch.no_grad():
    W_E = model.get_W_E()                                     # (D, V), unit columns

    # ---- Sanity check: all positions clean at σ=1 (κ=κ_max) should predict themselves ----
    t0 = req["tasks"][tasks[0]]
    toks = torch.tensor([t0["ctx"][0] + t0["cont"][0]], device=device)
    h = SphericalPath.get_target_embeddings(W_E, toks)
    lg = model.compute_logits(model(h, sigma=torch.ones(1, device=device)), W_E=W_E)
    ce_clean = F.cross_entropy(lg.float()[0], toks[0]).item()
    out["meta"]["sanity_ce_all_clean_sigma1"] = ce_clean
    print(f"[sfcd] sanity: all-clean input at sigma=1 -> CE {ce_clean:.4f}", flush=True)

    for name in tasks:
        t = req["tasks"][name]
        ctx, cont, seeds = t["ctx"], t["cont"], t["noise_seed"]
        n = len(ctx)
        ce_p = [0.0] * n
        ce_n = [0.0] * n
        by_len = {}
        for i in range(n):
            by_len.setdefault(len(ctx[i]) + len(cont[i]), []).append(i)
        t_start = time.time()
        for L in sorted(by_len):
            assert L <= model.sequence_length, L
            idx_all = by_len[L]
            for s in range(0, len(idx_all), args.batch):
                idx = idx_all[s:s + args.batch]
                b = len(idx)
                tokens = torch.tensor([ctx[i] + cont[i] for i in idx], device=device)
                cmask = torch.zeros((b, L), dtype=torch.bool, device=device)
                noise = torch.empty((b, L, D), dtype=torch.float32, device=device)
                gen = torch.Generator(device=device)
                for j, i in enumerate(idx):
                    cmask[j, len(ctx[i]):] = True
                    gen.manual_seed(int(seeds[i]))    # one draw per question, shared by its choices
                    noise[j] = torch.randn((L, D), generator=gen, device=device,
                                           dtype=torch.float32)
                h0 = F.normalize(noise, dim=-1)       # vMF with κ=0 (same as SFCD's initialization)
                sigma = torch.zeros(b, device=device)  # κ/κ_max = 0
                for dst, tok_in in ((ce_p, tokens),
                                    (ce_n, torch.where(cmask, tokens,
                                                       torch.full_like(tokens, neutral)))):
                    clean = SphericalPath.get_target_embeddings(W_E, tok_in)
                    h_t = torch.where(cmask.unsqueeze(-1), h0, clean)
                    h_prime = model(h_t, clue_mask=None, sigma=sigma, x_self_cond=None)
                    logits = model.compute_logits(h_prime, W_E=W_E)
                    ce = F.cross_entropy(logits.float().flatten(0, 1),
                                         tokens.flatten(0, 1),
                                         reduction="none").view(b, L)
                    vals = (ce * cmask).sum(dim=1).double().tolist()
                    for j, i in enumerate(idx):
                        dst[i] = vals[j]
        out["tasks"][name] = {"ce_prefix": ce_p, "ce_neutral": ce_n,
                              "digest": t["digest"]}
        mp = sum(ce_p) / n
        mn = sum(ce_n) / n
        print(f"[sfcd] {name}: {n} requests, mean CE-sum prefix {mp:.3f} / "
              f"neutral {mn:.3f} ({time.time() - t_start:.0f}s)", flush=True)

torch.save(out, args.out)
print("wrote", args.out)
print("SFCD_MCQ_FORWARD_DONE")
