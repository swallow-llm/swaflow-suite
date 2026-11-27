#!/bin/bash
# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# Prepare a branch: a new run that resumes from a full training state of an
# existing run (used for the clean-prefix mid-training branches).
#
#   scripts/launch/branch.sh <source_run> <branch_run> <step>
#
# Links $SWAFLOW_OUTPUTS/<source_run>/checkpoint/step-<step> into the branch's
# dump folder and copies the matching random-number state when it was kept
# (only the last 20 full states of a run keep it). Without it, the branch
# resumes model, optimizer, schedule and data position exactly but draws a
# different, identically distributed alpha/noise stream.
# Then start the branch with scripts/launch/train.sh <midtrain config> <branch_run>.
# The branch may run on a different number of GPUs than the source run, but the
# copied random-number state reproduces the source run's draws only on the same
# number.

set -euo pipefail
cd "$(dirname "$0")/../.."
. "${SWAFLOW_ENV:-scripts/launch/env.sh}"

SRC=${1:?usage: branch.sh <source_run> <branch_run> <step>}
DST=${2:?usage: branch.sh <source_run> <branch_run> <step>}
STEP=${3:?usage: branch.sh <source_run> <branch_run> <step>}

SRC_CKPT="$SWAFLOW_OUTPUTS/$SRC/checkpoint/step-$STEP"
DST_CKPT="$SWAFLOW_OUTPUTS/$DST/checkpoint/step-$STEP"
[ -f "$SRC_CKPT/.metadata" ] || { echo "branch.sh: no full state at $SRC_CKPT" >&2; exit 1; }
mkdir -p "$SWAFLOW_OUTPUTS/$DST/checkpoint"
[ -e "$DST_CKPT" ] || ln -s "$SRC_CKPT" "$DST_CKPT"

SRC_RNG="$SWAFLOW_OUTPUTS/$SRC/rngstate/step-$STEP"
DST_RNG="$SWAFLOW_OUTPUTS/$DST/rngstate/step-$STEP"
if [ -d "$SRC_RNG" ]; then
    mkdir -p "$SWAFLOW_OUTPUTS/$DST/rngstate"
    [ -d "$DST_RNG" ] || cp -r "$SRC_RNG" "$DST_RNG"
    echo "branch.sh: $DST starts from $SRC step $STEP (with its RNG state)"
else
    echo "branch.sh: $DST starts from $SRC step $STEP (RNG state not kept; fresh alpha/noise stream)"
fi
