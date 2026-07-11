#!/bin/bash
set -e
cd /sdf/scratch/users/a/ajshack/lcls2
set +u; source setup_env.sh; set -u
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
echo "=== default-path gate (also validates node.py edit) ==="
python psana/psana/gpu/test_jungfrau_calib.py -e mfx101572426 -r 47 -n 20 --dir $FFB
echo "=== seg-h2d gate (wait-split numeric path) ==="
python psana/psana/gpu/test_jungfrau_calib.py --seg-h2d -e mfx101572426 -r 47 -n 20 --dir $FFB
