#!/bin/bash
# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# Start one SwaFlow training run on this node.
#
#   scripts/launch/train.sh <config.toml> <run_name> [torchtitan overrides ...]
#
# The dump folder is $SWAFLOW_OUTPUTS/<run_name>; rerunning the same command
# resumes from the latest checkpoint in it. For a multi-node run, start this
# script once per node with NNODES, NODE_RANK and MASTER_ADDR set (see
# scripts/launch/pbs_example.sh and slurm_example.sh). The global batch is fixed
# by the config, so any node count works for a fresh run, but a resumed run
# must keep the data-parallel size it was started with.
#
# Data paths are filled in from $SWAFLOW_DATA_ROOT by config family; set
# SWAFLOW_DATASET to point --training.dataset_path elsewhere.

set -euo pipefail
cd "$(dirname "$0")/../.."
. "${SWAFLOW_ENV:-scripts/launch/env.sh}"

CONFIG=${1:?usage: train.sh <config.toml> <run_name> [overrides ...]}
RUN=${2:?usage: train.sh <config.toml> <run_name> [overrides ...]}
shift 2

PRETRAIN_BUILD="$SWAFLOW_DATA_ROOT/production_6t"
case "$(basename "$CONFIG")" in
    sft_dolci_*)   DATASET="$SWAFLOW_DATA_ROOT/dolci_sft_2048" ;;
    sft_tinygsm*)  DATASET="$SWAFLOW_DATA_ROOT/tinygsm_pad512" ;;
    pilot_window*) DATASET="$SWAFLOW_DATA_ROOT/dolma3_1b_val"; PRETRAIN_BUILD="$DATASET" ;;
    debug_*)       DATASET="" ;;
    *)             DATASET="$PRETRAIN_BUILD" ;;
esac
DATASET="${SWAFLOW_DATASET:-$DATASET}"

case "$(basename "$CONFIG")" in
    ar_*|sft_dolci_ar*|debug_ar*) MODULE=torchtitan.models.swaflow.ar.train ;;
    *)                            MODULE=torchtitan.models.swaflow.train ;;
esac

ARGS=(--job.config_file "$CONFIG" --job.dump_folder "$SWAFLOW_OUTPUTS/$RUN")
if [ -n "$DATASET" ]; then
    ARGS+=(--model.hf_assets_path "$SWAFLOW_TOKENIZER_DIR"
           --training.dataset_path "$DATASET"
           --swaflow.eval.build_dir "$PRETRAIN_BUILD")
fi

mkdir -p "$SWAFLOW_OUTPUTS/$RUN"
exec torchrun \
    --nnodes "${NNODES:-1}" --node_rank "${NODE_RANK:-0}" \
    --nproc_per_node "${GPUS_PER_NODE:-8}" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29500}" \
    -m "$MODULE" "${ARGS[@]}" "$@"
