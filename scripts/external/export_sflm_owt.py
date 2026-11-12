# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# Export the released S-FLM checkpoint (HF `jdeschena/s-flm`, `owt/sfm.ckpt`) in
# a form our MCQ harness can load, and dump a reference forward pass to check
# it against. Runs inside the S-FLM code's own environment (GPU required), not
# ours: the S-FLM code computes the reference values, and `check_sflm_owt.py`
# reads the dump on our side (the two code bases never import each other).
#
#   <S-FLM env>/bin/python export_sflm_owt.py --sflm-repo <S-FLM repo> --ckpt <sfm.ckpt> --out <dir>
#
# Output (<out>/):
#   weights_raw.pt  : backbone state_dict (S-FLM keys, `backbone.` stripped)
#   weights_ema.pt  : same keys with the EMA (0.9999) shadow weights. The
#                     authors evaluate with `eval.disable_ema=false`, so this
#                     is the model they evaluated.
#   reference.pt    : fp32 forward with the EMA weights (below) and the main
#                     training-config fields
#
# This checkpoint has `invert_time_convention=false`: α points the same way as
# ours (α=1 clean), conditioning is σ = −log α (`trainer_base._sigma_from_alphat`)
# and x_t = slerp(clean, noisy, 1 − α). It was trained on LogLinear(eps=1e-3)
# truncated to α in [1e-3, 0.121]. The reference forward builds x_t and σ with
# the S-FLM functions under this convention.

import argparse
import contextlib
import math
import os
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument("--sflm-repo", default=os.environ.get("SFLM_REPO", ""))
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=20260919)
args = p.parse_args()

sys.path.insert(0, args.sflm_repo)
os.makedirs(args.out, exist_ok=True)

# ---- Compare in fp32: disable autocast ----
_real_autocast = torch.amp.autocast
torch.amp.autocast = lambda *a, **k: contextlib.nullcontext()

import omegaconf  # noqa: E402
import utils as sflm_utils  # noqa: E402  (S-FLM)
from models.sphere_dit import SphereDiT  # noqa: E402  (S-FLM)

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = ck["hyper_parameters"]["config"]
# `hyper_parameters["vocab_size"]` is saved as None; take the vocabulary size
# from the embedding table (GPT-2's 50,257 + 1 pad = 50,258).
vocab_size = int(ck["state_dict"]["backbone.sphere_embed.weight"].shape[0])
assert cfg.algo.name == "sfm" and cfg.model.type == "sphere-dit", cfg.algo.name
assert cfg.algo.invert_time_convention is False  # assumed by the convention above
assert cfg.algo.time_conditioning is True

device = torch.device("cuda")
model = SphereDiT(cfg, vocab_size=vocab_size)

raw = {k[len("backbone."):]: v for k, v in ck["state_dict"].items()
       if k.startswith("backbone.")}
model.load_state_dict(raw, strict=True)

# ---- EMA: shadow params follow `itertools.chain(backbone.parameters(),
#      noise.parameters())` (trainer_base._get_parameters); noise has none ----
shadow = ck["ema"]["shadow_params"]
names = [n for n, _ in model.named_parameters()]
assert len(shadow) == len(names), (len(shadow), len(names))
ema = dict(raw)
for n, s in zip(names, shadow):
    assert tuple(s.shape) == tuple(raw[n].shape), (n, s.shape, raw[n].shape)
    ema[n] = s.detach().float().clone()

meta = {
    "source": "HF jdeschena/s-flm owt/sfm.ckpt",
    "global_step": int(ck["global_step"]),
    "vocab_size": vocab_size,
    "hidden_size": int(cfg.model.hidden_size),
    "n_blocks": int(cfg.model.n_blocks),
    "n_heads": int(cfg.model.n_heads),
    "cond_dim": int(cfg.model.cond_dim),
    "length": int(cfg.model.length),
    "tokenizer": str(cfg.data.tokenizer_name_or_path),
    "invert_time_convention": bool(cfg.algo.invert_time_convention),
    "noise_eps": float(cfg.noise.eps),
    "alpha_max": float(cfg.noise.alpha_max),
    "adaptive": bool(cfg.noise.adaptive),
    "ema_decay": float(ck["ema"]["decay"]),
    "ema_num_updates": int(ck["ema"]["num_updates"]),
    "disable_ema_at_eval": bool(cfg.eval.disable_ema),
    "wrap": bool(cfg.data.wrap),
    "insert_train_eos": bool(cfg.data.insert_train_eos),
}
# Trained α range (the noise schedule's table, in our convention).
av = ck["state_dict"]["noise.alpha_vals"]
meta["alpha_vals_min"] = float(av.min())
meta["alpha_vals_max"] = float(av.max())
print("meta", meta)

torch.save({"meta": meta, "state_dict": {k: v.float() for k, v in raw.items()}},
           f"{args.out}/weights_raw.pt")
torch.save({"meta": meta, "state_dict": {k: v.float() for k, v in ema.items()}},
           f"{args.out}/weights_ema.pt")

# ---- Reference forward (EMA weights, fp32, eval mode) ----
model.load_state_dict(ema, strict=True)
model = model.to(device).eval()

TEXT = (
    "The city council met on Tuesday to discuss the new budget for public "
    "transport. Several members argued that the bus network should be "
    "expanded to the northern suburbs, where many residents have no access "
    "to a car. Others were worried about the cost and asked for a detailed "
    "study before any decision is made. The mayor said that a final vote "
    "would take place next month, after the public consultation has ended. "
    "Meanwhile, the local newspaper reported that ticket prices will rise "
    "by five percent at the start of the year."
)
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
ids = tok(TEXT, add_special_tokens=False)["input_ids"]
# Does the tokenizer pickled in the checkpoint give the same ids? It may not
# unpickle under other library versions; record that if so.
try:
    ids_ckpt = ck["hyper_parameters"]["tokenizer"](
        TEXT, add_special_tokens=False)["input_ids"]
except Exception as e:  # noqa: BLE001
    ids_ckpt = f"unavailable: {type(e).__name__}: {e}"
L = 96
assert len(ids) >= L, len(ids)
tokens = torch.tensor([ids[:L]] * 4, dtype=torch.long)

g = torch.Generator(device="cpu").manual_seed(args.seed)
noise = torch.randn((4, L, meta["hidden_size"]), generator=g, dtype=torch.float32)
# α in our convention (= this checkpoint's α). Sample 3 is a clean prefix
# (first half clean, second half pure noise).
alpha = torch.tensor([1e-3, 0.03, 0.121, 1e-3], dtype=torch.float64)
prefix_len = L // 2

with torch.no_grad():
    clean = model.get_sphere_embeddings(tokens.to(device))          # [B, L, d]
    noisy = sflm_utils.sphere_normalize(noise.to(device))
    # algo.SFM.q_xt (invert=false): slerp_t = 1 − α, slerp in float64 (slerp_precision)
    x_t = sflm_utils.slerp(clean=clean.double(), noisy=noisy.double(),
                           alpha_t=(1 - alpha).to(device), eps=float(cfg.algo.eps))
    x_t = x_t.float()
    # Clean prefix: first half x_t = clean, second half pure noise (α=0).
    x_t[3, :prefix_len] = clean[3, :prefix_len]
    x_t[3, prefix_len:] = noisy[3, prefix_len:]
    # trainer_base._sigma_from_alphat: σ = −log α
    sigma = (-torch.log(alpha)).float()
    logits = model(None, x_t, sigma.to(device)).float()
    ce = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), tokens.to(device).flatten(0, 1), reduction="none"
    ).view(4, L)

ce_mean = ce.mean(dim=1).tolist()
ce_cont = ce[3, prefix_len:].mean().item()
print("CE per sample (α = 1e-3, 0.03, 0.121, clean-prefix):", ce_mean)
print("clean-prefix sample: CE on the noised half =", ce_cont)

torch.save(
    {
        "meta": meta,
        "text": TEXT,
        "ids_hf_gpt2": ids,
        "ids_ckpt_tokenizer": ids_ckpt,
        "tokens": tokens,
        "noise": noise,
        "alpha": alpha.float(),
        "prefix_len": prefix_len,
        "x_t": x_t.cpu(),
        "sigma": sigma.cpu(),
        "logits": logits.cpu(),
        "ce_per_sample": ce_mean,
        "ce_clean_prefix_cont": ce_cont,
        "note": "EMA weights, fp32, autocast disabled, eval mode",
    },
    f"{args.out}/reference.pt",
)
torch.amp.autocast = _real_autocast
print("ids equal (hf gpt2 vs ckpt tokenizer):",
      ids == ids_ckpt if isinstance(ids_ckpt, list) else ids_ckpt)
print("finite:", torch.isfinite(logits).all().item(), "sigma", sigma.tolist(),
      "log(1e3) =", math.log(1e3))
print("wrote", args.out)
print("EXPORT_SFLM_OWT_DONE")
