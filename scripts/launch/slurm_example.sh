#!/bin/bash
# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# Example multi-node job for Slurm: runs CMD once per node via srun.
#
#   CMD="scripts/launch/train.sh torchtitan/models/swaflow/train_configs/flow_150m.toml flow_150m" \
#       sbatch --export=ALL scripts/launch/slurm_example.sh
#
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=7-00:00:00

set -euo pipefail
export NNODES=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29400 + SLURM_JOB_ID % 1000))

export CMD="${CMD:?set CMD}"
srun bash -c 'NODE_RANK=$SLURM_NODEID $CMD'
