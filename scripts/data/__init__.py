# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Pretraining data pipeline.

- `config.py`             TOML source definitions with mixture weights
- `indexed_dataset.py`    uint16 indexed dataset (.bin / .idx / .meta.json)
- `permutation.py`        deterministic shuffle (Feistel; O(1) map, no stored array)
- `plan.py`               slot -> block resolution (mixture, held-out, epochs)
- `tokenize_corpus.py`    CLI: download -> tokenize -> shards -> plan
- `order_log.py`          CLI: expand a step / checkpoint into document references
- `verify_order.py`       CLI: round-trip, rebuild and held-out checks
- `make_dolma3_config.py` generates the Dolma 3 Mix configs
- `build_sft.py`          builds SFT blocks (Dolci, TinyGSM)

This package does not depend on torchtitan; the training-side loader
(`torchtitan/models/swaflow/indexed_data.py`) imports it.
"""
