#!/usr/bin/env bash
set -eo pipefail

source "$HOME/lcls2/setup_env.sh" >/dev/null
set -u

python "$HOME/lcls2/psdaq/psdaq/debugtools/epixquad1kfps/diagnose_per_pixel_gain.py" \
  --exp ued1015980 \
  --runs 5,6,7 \
  --map /sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/roi_ft_ued1015999_r185_t300_v2/roiFromAboveThreshold_r185_c0_calib_expand4.npy \
  --csv /sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/roi_ft_ued1015999_r185_t300_v2/expand4_gainbit_fp_fn.csv
