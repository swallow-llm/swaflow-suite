# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# Paths used by the launch scripts. Edit the three paths below, or copy this file
# and point SWAFLOW_ENV at the copy. The launch scripts source it on every node,
# so also load your modules and Python environment here if your site needs it.

export SWAFLOW_DATA_ROOT="${SWAFLOW_DATA_ROOT:-$PWD/data}"            # data builds (scripts/data/README.md)
export SWAFLOW_OUTPUTS="${SWAFLOW_OUTPUTS:-$PWD/outputs}"             # one dump folder per run
export SWAFLOW_TOKENIZER_DIR="${SWAFLOW_TOKENIZER_DIR:-$PWD/assets/hf/SmolLM2-135M}"  # HuggingFaceTB/SmolLM2-135M tokenizer files

export PYTHONPATH="${SWAFLOW_REPO:-$PWD}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
