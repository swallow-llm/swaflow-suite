# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# Per-size settings shared by the SFT and evaluation launchers.
#   size_params <150M|500M|1B|3B>  sets FLAVOR, ALPHA_MAX, ALPHA_GRID (the
#   8-point training grid over [0, ALPHA_MAX]), SFT_LR (Dolci SFT) and AC_MODE.

size_params() {
    case "$1" in
        150M) FLAVOR=150M; ALPHA_MAX=0.15;  SFT_LR=6e-5; AC_MODE=none
              ALPHA_GRID="0.01875 0.0375 0.05625 0.075 0.09375 0.1125 0.13125 0.15" ;;
        500M) FLAVOR=500M; ALPHA_MAX=0.125; SFT_LR=3e-5; AC_MODE=none
              ALPHA_GRID="0.015625 0.03125 0.046875 0.0625 0.078125 0.09375 0.109375 0.125" ;;
        1B)   FLAVOR=1B;   ALPHA_MAX=0.105; SFT_LR=3e-5; AC_MODE=selective
              ALPHA_GRID="0.013125 0.02625 0.039375 0.0525 0.065625 0.07875 0.091875 0.105" ;;
        3B)   FLAVOR=3B;   ALPHA_MAX=0.10;  SFT_LR=3e-5; AC_MODE=selective
              ALPHA_GRID="0.0125 0.025 0.0375 0.05 0.0625 0.075 0.0875 0.10" ;;
        *) echo "size_params: unknown size '$1' (expected 150M, 500M, 1B or 3B)" >&2; return 1 ;;
    esac
}
