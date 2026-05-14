#!/usr/bin/env bash
set -eo pipefail

source "$HOME/lcls2/setup_env.sh" >/dev/null
set -u

python "$HOME/lcls2/psdaq/psdaq/debugtools/epixquad1kfps/diagnose_per_pixel_gain.py" \
  --exp ued1015980 \
  --runs 2,3,4 \
  --map /sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/roi_ft_ued1015999_r185_t300_v2/roiFromAboveThreshold_r185_c0_calib_expand2.npy \
  --csv /sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/roi_ft_ued1015999_r185_t300_v2/expand2_gainbit_fp_fn.csv
