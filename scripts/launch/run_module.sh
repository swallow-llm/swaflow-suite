#!/bin/bash
# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# Run a Python module under torchrun on this node, e.g. an evaluation harness:
#
#   scripts/launch/run_module.sh scripts.eval.ckpt_eval --run flow_150m ...
#
# For several nodes, start it once per node with NNODES, NODE_RANK and
# MASTER_ADDR set, as for train.sh.

set -euo pipefail
cd "$(dirname "$0")/../.."
. "${SWAFLOW_ENV:-scripts/launch/env.sh}"

MODULE=${1:?usage: run_module.sh <python.module> [args ...]}
shift
exec torchrun \
    --nnodes "${NNODES:-1}" --node_rank "${NODE_RANK:-0}" \
    --nproc_per_node "${GPUS_PER_NODE:-8}" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29500}" \
    -m "$MODULE" "$@"
