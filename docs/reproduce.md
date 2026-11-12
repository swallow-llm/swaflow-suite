# Reproducing the paper

Every run in the paper, in the order it was trained, with the config and command that reproduce it. Commands start one node's share of a run; launch them on every node of a multi-node job with [`scripts/launch/pbs_example.sh`](../scripts/launch/pbs_example.sh) or [`slurm_example.sh`](../scripts/launch/slurm_example.sh), for example

```bash
qsub -v CMD="scripts/launch/train.sh torchtitan/models/swaflow/train_configs/flow_150m.toml flow_150m" scripts/launch/pbs_example.sh
```

Paths come from [`scripts/launch/env.sh`](../scripts/launch/env.sh). Run names below are the dump folders under `$SWAFLOW_OUTPUTS`; the evaluation scripts expect them.

All pretraining runs and branches used 32 nodes × 8 H200 (256 data-parallel ranks). The global batch (4096 × 2048 tokens) is fixed by the configs, so other node counts give the same optimisation, but the per-rank noise streams depend on the number of ranks, and a resumed run must keep the size it started with.

## 1. Data

Build the tokenized corpus, the order log and the SFT data as described in [`scripts/data/README.md`](../scripts/data/README.md). The runs below expect, under `$SWAFLOW_DATA_ROOT`:

| Directory | Contents |
|---|---|
| `production_6t/` | the full Dolma 3 Mix build: plan, shards, held-out split |
| `dolma3_1b_val/` | a 1B-token build from the same pipeline (window pilot, order-log checks) |
| `dolci_sft_2048/` | Dolci Instruct SFT build, 2048-token blocks |
| `tinygsm_pad512/` | TinyGSM SFT build, each example padded to 512 tokens |

The directory names are the build names recorded in each build's plan; a resumed run checks them, so keep them when resuming a training state.

## 2. Window pilot (α_max)

An untruncated pilot per size (α drawn from [0, 1]) sets the window edge α_max: the α at which the pilot's held-out loss falls below 0.1 nats. The pilot learning rates are 6.0e-4, 4.6e-4, 3.7e-4 and 3.0e-4 for 150M, 500M, 1B and 3B.

```bash
scripts/launch/train.sh torchtitan/models/swaflow/train_configs/pilot_window.toml pilot_150m \
    --model.flavor 150M --optimizer.lr 6.0e-4 --activation_checkpoint.mode none
python -m scripts.analysis.alpha_window --dump $SWAFLOW_OUTPUTS/pilot_150m --label 150M --dim 768 \
    --csv $SWAFLOW_OUTPUTS/pilot_150m/alpha_window.csv
```

The resulting windows (0.150, 0.125, 0.105, 0.100) are written into the configs below.

## 3. Pretraining

| Run | Config | Tokens |
|---|---|---|
| `flow_150m`, `flow_500m`, `flow_1b` | `flow_150m.toml`, `flow_500m.toml`, `flow_1b.toml` | 2T |
| `flow_3b` | `flow_3b.toml` | 3T |
| `ar_150m`, `ar_500m`, `ar_1b` | `ar_150m.toml`, `ar_500m.toml`, `ar_1b.toml` | 2T |

```bash
scripts/launch/train.sh torchtitan/models/swaflow/train_configs/flow_150m.toml flow_150m
scripts/launch/train.sh torchtitan/models/swaflow/train_configs/ar_150m.toml ar_150m
```

Each run writes a full training state (weights, optimizer, schedule, dataloader position) every 20B tokens, weights-only snapshots at steps 1, 2, 4, …, 512, and the held-out loss on its training α grid at every full state. The learning rate is constant after a 2,000-step warmup in every run.

## 4. Clean-prefix mid-training

A branch resumes a full state of a pretraining run and continues on the same data stream; only the way α is applied changes. [`scripts/launch/branch.sh`](../scripts/launch/branch.sh) links the source state into the new run and copies its random-number state when it was kept.

| Run | Source | Branch step | End step | Config |
|---|---|---|---|---|
| `midtrain_150m` | `flow_150m` | 226,480 (1.9T) | 238,400 (2T) | `midtrain_150m.toml` |
| `midtrain_500m` | `flow_500m` | 226,480 | 238,400 | `midtrain_500m.toml` |
| `midtrain_1b` | `flow_1b` | 226,480 | 238,400 | `midtrain_1b.toml` |
| `midtrain_1b_500b` | `flow_1b` | 47,680 (0.4T) | 59,600 (0.5T) | `midtrain_1b.toml` with `--training.steps 59600` |
| `midtrain_1b_1t` | `flow_1b` | 107,280 (0.9T) | 119,200 (1T) | `midtrain_1b.toml` with `--training.steps 119200` |
| `midtrain_3b` | `flow_3b` | 226,480 | 238,400 | `midtrain_3b.toml` |
| `midtrain_3b_final` | `flow_3b` | 357,600 (3T) | 476,800 (4T) | `midtrain_3b_final.toml` |

```bash
scripts/launch/branch.sh flow_150m midtrain_150m 226480
scripts/launch/train.sh torchtitan/models/swaflow/train_configs/midtrain_150m.toml midtrain_150m

scripts/launch/branch.sh flow_1b midtrain_1b_500b 47680
scripts/launch/train.sh torchtitan/models/swaflow/train_configs/midtrain_1b.toml midtrain_1b_500b --training.steps 59600
```

The random-number state is kept only for the last 20 full states of a run, so the 3B branch at 1.9T and the two early 1B branches draw a different (identically distributed) α and noise stream from the source run; optimizer, schedule and data position resume exactly.

## 5. Post-training

### Dolci Instruct (one epoch)

Bases are the 2T states (step 238,400) of the uniform-α runs, the mid-training branches and the autoregressive twins. Per-size settings are in [`scripts/launch/sizes.sh`](../scripts/launch/sizes.sh); the learning rate is one tenth of pretraining.

```bash
. scripts/launch/sizes.sh && size_params 500M
scripts/launch/train.sh torchtitan/models/swaflow/train_configs/sft_dolci_flow.toml sft_dolci_midtrain_500m \
    --model.flavor $FLAVOR --optimizer.lr $SFT_LR --activation_checkpoint.mode $AC_MODE \
    --swaflow.model.schedule_alpha_max $ALPHA_MAX --swaflow.eval.alpha_grid $ALPHA_GRID \
    --checkpoint.initial_load_path $SWAFLOW_OUTPUTS/midtrain_500m/checkpoint/step-238400
```

Use `sft_dolci_ar.toml` for the autoregressive twins (with `--model.flavor` and `--optimizer.lr`), and `sft_dolci_3b.toml` for the 3B.

### TinyGSM (65B tokens)

The same recipe for every size and base: LR 3e-4, 61,988 steps of 2048 × 512 tokens, scored at 20B, 40B and 65B tokens. The schedule spans all 61,988 steps; stop and resume at the two intermediate points so that their states are saved.

```bash
. scripts/launch/sizes.sh && size_params 1B
for STEPS in 19073 38146 61988; do
  scripts/launch/train.sh torchtitan/models/swaflow/train_configs/sft_tinygsm.toml tinygsm_midtrain_1b_500b \
      --model.flavor $FLAVOR --activation_checkpoint.mode $AC_MODE \
      --swaflow.model.schedule_alpha_max $ALPHA_MAX --swaflow.eval.alpha_grid $ALPHA_GRID \
      --checkpoint.initial_load_path $SWAFLOW_OUTPUTS/midtrain_1b_500b/checkpoint/step-59600 \
      --training.steps $STEPS
done
```

Bases are the mid-training states (150M and 500M at 2T; 1B at 0.5T, 1T and 2T; 3B at 2T and 4T); leave `--checkpoint.initial_load_path` out to train from scratch.

### Post-hoc EMA

The averaged 3B model is EMA-100, reconstructed from the full states saved every 20B tokens of `midtrain_3b_final` (followed back into `flow_3b` before the branch point):

```bash
python scripts/posthoc_ema.py --checkpoint-root $SWAFLOW_OUTPUTS/midtrain_3b_final \
    --target-tokens 4000000000000 --checkpoint-interval-tokens 20000000000 \
    --ema-k 100 --output-dir $SWAFLOW_OUTPUTS
```

## 6. Evaluation

All harnesses read full training states after the fact and write one JSON per step into the run's dump folder. Run them with [`scripts/launch/run_module.sh`](../scripts/launch/run_module.sh) on as many nodes as convenient. `--alpha-max` is the size's window edge.

**α profiles.** The training grid (8 points over the window) is logged by the run itself. The common grid (8 points on [0, 0.10], for comparing cross entropy across sizes) and the 32-point window grid (for locating α_sol) are computed by `ckpt_eval`; α_geo is computed from the embedding table in the same pass.

```bash
scripts/launch/run_module.sh scripts.eval.ckpt_eval --dump $SWAFLOW_OUTPUTS/flow_150m \
    --kind flow --flavor 150M --alpha-max 0.15 --grid common --max-blocks 2048 --local-batch 8
scripts/launch/run_module.sh scripts.eval.ckpt_eval --dump $SWAFLOW_OUTPUTS/flow_150m \
    --kind flow --flavor 150M --alpha-max 0.15 --grid window32 --max-blocks 2048 --local-batch 8
python -m scripts.eval.equivalence --series val_window32 --out $SWAFLOW_OUTPUTS/analysis/equivalence
```

**Generation.** Conditional and unconditional collapse rate and Gen-PPL, 256 samples of 512 tokens at 64 function evaluations, every 20B tokens:

```bash
scripts/launch/run_module.sh scripts.eval.collapse --outputs $SWAFLOW_OUTPUTS --runs flow_150m,flow_500m,flow_1b \
    --every-tokens 20000000000 --num-samples 256 --nfe 64 --temperature 1.0 --gen-ppl
```

**Multiple choice and the prefix probe.**

```bash
scripts/launch/run_module.sh scripts.eval.run_mcq --dump $SWAFLOW_OUTPUTS/midtrain_150m --kind flow \
    --flavor 150M --alpha-max 0.15 --prefix-mode clean --pmi --alpha-list 0.0 --batch 32
scripts/launch/run_module.sh scripts.eval.prefix_probe --dump $SWAFLOW_OUTPUTS/midtrain_150m --kind flow \
    --flavor 150M --alpha-max 0.15 --n-docs 1024 --batch 32
```

**Post-trained models.** Held-out SFT loss, GSM8K (5-shot, text answers) for Dolci models, and TinyGSM-style program execution (0-shot) for TinyGSM models:

```bash
scripts/launch/run_module.sh scripts.eval.sft_val --dump $SWAFLOW_OUTPUTS/sft_dolci_midtrain_500m \
    --build $SWAFLOW_DATA_ROOT/dolci_sft_2048 --kind flow --flavor 500M --alpha-max 0.125 --max-blocks 1024
scripts/launch/run_module.sh scripts.eval.gsm8k --dump $SWAFLOW_OUTPUTS/sft_dolci_midtrain_500m --kind flow \
    --flavor 500M --alpha-max 0.125 --num-fewshot 5 --template plain --seq-len 1536 --nfe 64 \
    --temperature 1.0 --scoring text --velocity exact --top-k-velocity -1
scripts/launch/run_module.sh scripts.eval.gsm8k --dump $SWAFLOW_OUTPUTS/tinygsm_midtrain_1b_500b --kind flow \
    --flavor 1B --alpha-max 0.105 --steps 19073,38146,61988 --out-name gsm8k_code --num-fewshot 0 \
    --template plain --seq-len 512 --nfe 64 --temperature 1.0 --scoring code --exec-timeout 2 \
    --velocity exact --top-k-velocity -1
```

**Released checkpoints of other models.** [`scripts/external/`](../scripts/external/) scores the S-FLM and SFCD OpenWebText checkpoints with the same multiple-choice harness.
