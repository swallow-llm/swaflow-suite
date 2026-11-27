<h1 align="center">SwaFlow Suite</h1>

<p align="center">
  <a href="https://huggingface.co/collections/tokyotech-llm/swaflow-suite"><img src="https://img.shields.io/badge/🤗-Huggingface-blue" alt="HuggingFace"></a>
</p>

SwaFlow is a suite of hyperspherical (Riemannian) flow language models trained from scratch under one recipe, so that how they scale can be studied. This repository contains the code used to train and evaluate them for the paper. It is built on [torchtitan](https://github.com/pytorch/torchtitan) v0.2.2.

> **Paper:** *SwaFlow: A Fully Open Suite for Studying Riemannian Flow Language Models Across Training and Scale* (link coming soon)

## Models

The paper trains flow models of four sizes on [Dolma 3 Mix](https://huggingface.co/datasets/allenai/dolma3_mix-6T): 150M, 500M and 1B to 2T tokens, and 3B to 4T (3T of pretraining, then 1T of clean-prefix mid-training). Autoregressive twins at 150M, 500M and 1B are trained on the same data, in the same order, with the same schedule.

Released models are collected in [tokyotech-llm/swaflow-suite](https://huggingface.co/collections/tokyotech-llm/swaflow-suite).

| Size | Parameters | Width | Layers | Heads | FFN | Window α_max |
|---|---|---|---|---|---|---|
| 150M | 175M | 768 | 12 | 12 | 2048 | 0.150 |
| 500M | 686M | 1280 | 24 | 10 | 3584 | 0.125 |
| 1B | 1.59B | 2048 | 24 | 16 | 5632 | 0.105 |
| 3B | 3.74B | 3072 | 28 | 24 | 8192 | 0.100 |

Time convention: α = 1 is clean data and α = 0 is pure noise. During training, one α per sequence is drawn uniformly from the static window [0, α_max].

### Using the released checkpoints

Training states are stored one per folder (for example `uniform/step-238400/`), each a full torchtitan checkpoint with the optimizer state. The weights of any state can be loaded directly; only that folder is downloaded:

```python
from torchtitan.models.swaflow.hf import from_pretrained

model = from_pretrained("tokyotech-llm/SwaFlow-150M", subfolder="uniform/step-238400", device="cuda")
```

To resume training, place the folder under `<dump folder>/checkpoint/` and launch the matching config. The number of GPUs may differ from the original run's: weights, optimizer state, learning-rate schedule and data position carry over, and gradient accumulation keeps the global batch at 4096 sequences, so every step reads the same sequences as in the original run. The random-number state is not included and floating-point sums are taken in a different order, so a resumed run does not retrace the original run exactly. See [Resuming a training state](docs/reproduce.md#resuming-a-training-state).

## Installation

```bash
git clone https://github.com/swallow-llm/swaflow-suite.git
cd swaflow-suite
python -m venv .venv && . .venv/bin/activate
pip install torch==2.12.0 torchao==0.17.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-swaflow.txt
pip install -e .
```

All runs in the paper used Python 3.14, CUDA 12.6 and nodes of 8 × H200. The tokenizer is that of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

Set the paths in [`scripts/launch/env.sh`](scripts/launch/env.sh) (data, outputs, tokenizer). To check the installation on one node, train a 2-layer model for 100 steps on the sample data bundled with torchtitan, and run the unit tests:

```bash
scripts/launch/train.sh torchtitan/models/swaflow/train_configs/debug_flow.toml install_check
python -m pytest torchtitan/models/swaflow/tests
```

## Reproducing the paper

[`docs/reproduce.md`](docs/reproduce.md) lists every run with its config and command, in order:

1. data: build plan, download, tokenization and the order log ([`scripts/data/`](scripts/data/README.md));
2. the window pilot that sets α_max for each size;
3. pretraining of the flow models and their autoregressive twins;
4. clean-prefix mid-training branches and the final 3B model;
5. post-training on Dolci Instruct and TinyGSM, and post-hoc EMA;
6. evaluation: per-α loss profiles on three α grids, α_sol, α_geo, α_eq, generation (collapse rate, Gen-PPL), multiple choice, the prefix probe and GSM8K.

## Data order

Every run reads the same build of Dolma 3 Mix with the same global batch, so step N reads the same data in every run, and the order follows from the data config shipped here. The data order log (the build plan and each run's record) will be released on the Hugging Face Hub; with it, [`scripts/data/order_log.py`](scripts/data/README.md) expands any step of any run into the documents it read, without rebuilding the data.

## Repository layout

| Path | Contents |
|---|---|
| `torchtitan/models/swaflow/` | model, loss, noise schedule, sampler, data loaders, checkpoint schedule and training entry point |
| `torchtitan/models/swaflow/ar/` | the autoregressive twins |
| `torchtitan/models/swaflow/train_configs/` | one config per run in the paper |
| `scripts/data/` | data pipeline and order log |
| `scripts/launch/` | single-node and multi-node launchers |
| `scripts/eval/` | evaluation harness |
| `scripts/external/` | scoring released checkpoints of other models (S-FLM, SFCD) with the same harness |
| `scripts/analysis/alpha_window.py` | sets α_max from the window pilot |
| `scripts/posthoc_ema.py` | post-hoc EMA weight averaging |

Everything else is torchtitan v0.2.2; its original README is in [`docs/torchtitan_README.md`](docs/torchtitan_README.md).

## Acknowledgements

SwaFlow was built by AIST and Institute of Science Tokyo under FRONTia, a Japanese national program led by the Ministry of Economy, Trade and Industry of Japan (METI) and the New Energy and Industrial Technology Development Organization (NEDO), aimed at developing a domestic multimodal foundation model for AI robots and physical AI.

## License

The code in this repository that is not part of torchtitan is released under the [Apache License 2.0](LICENSE). torchtitan is under the [BSD 3-Clause License](LICENSE-torchtitan), and parts of the sampler and of the GSM8K scorer are adapted from [S-FLM](https://github.com/jdeschena/s-flm) (Apache License 2.0); see [NOTICE](NOTICE). Model weights are released under the Apache License 2.0. Dolma 3 Mix is licensed under ODC-By 1.0 by the Allen Institute for AI.

## Citation

```bibtex
@misc{swaflow2026,
  title  = {SwaFlow: A Fully Open Suite for Studying Riemannian Flow Language Models Across Training and Scale},
  author = {Oba, Daisuke and Mizuki, Sakae and Takamura, Hiroya and Okazaki, Naoaki},
  year   = {2026},
  note   = {coming soon}
}
```
