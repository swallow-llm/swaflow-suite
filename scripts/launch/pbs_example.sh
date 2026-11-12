#!/bin/bash
# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# Example multi-node job for PBS Professional: runs CMD once per node via pbsdsh.
# Adjust the resource lines to your site (project, queue, node type).
#
#   qsub -v CMD="scripts/launch/train.sh torchtitan/models/swaflow/train_configs/flow_150m.toml flow_150m" scripts/launch/pbs_example.sh
#   qsub -v CMD="scripts/launch/run_module.sh scripts.eval.ckpt_eval <args>" scripts/launch/pbs_example.sh
#
# qsub -v splits on commas, so CMD must not contain any. The other nodes do not
# inherit this environment; set SWAFLOW_ENV to pass a site env file to them.
#
#PBS -l select=32
#PBS -l walltime=168:00:00
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:?}"

# one entry per node, the first being this node
mapfile -t NODES < <(awk '!seen[$1]++ {print $1}' "$PBS_NODEFILE")
export NNODES=${#NODES[@]}
export MASTER_ADDR=$(getent ahostsv4 "${NODES[0]}" | awk '{print $1; exit}')
export MASTER_PORT=$((29400 + ${PBS_JOBID%%[!0-9]*} % 1000))

pids=()
for i in "${!NODES[@]}"; do
    cmd="cd $PBS_O_WORKDIR && SWAFLOW_ENV=${SWAFLOW_ENV:-} NNODES=$NNODES NODE_RANK=$i MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT ${CMD:?set CMD}"
    if [ "$i" -eq 0 ]; then
        bash -c "$cmd" &
    else
        # pbsdsh -n takes the index of the node's first line in $PBS_NODEFILE
        idx=$(awk -v h="${NODES[$i]}" '$1 == h {print NR - 1; exit}' "$PBS_NODEFILE")
        pbsdsh -n "$idx" -- bash -c "$cmd" &
    fi
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
