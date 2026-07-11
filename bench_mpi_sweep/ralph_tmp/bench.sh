set +u
cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh
set -u
FFB=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx101572426/xtc
NPROC=$1
EXTRA=$2
mpirun --bind-to none --oversubscribe -n $NPROC \
    python psana/psana/gpu/bench_calib.py \
    -e mfx101572426 -r 47 -n 500 --warmup 10 --dir $FFB $EXTRA
