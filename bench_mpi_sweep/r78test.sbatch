#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:data
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep/slurm-%j.out
#SBATCH --job-name=gpu_r78_test

set -x

cd /sdf/scratch/users/a/ajshack/lcls2
source setup_env.sh

OUTDIR=/sdf/scratch/users/a/ajshack/lcls2/bench_mpi_sweep
LOG=${OUTDIR}/r78_correctness.log
: > "${LOG}"

echo "== detnames for mfx100852324 r78 ==" >> "${LOG}"
DETNAME=$(python - <<'EOF' 2>> "${LOG}"
from psana import DataSource
ds = DataSource(exp='mfx100852324', run=78, max_events=1)
r = next(ds.runs())
import sys
print("detnames:", r.detnames, file=sys.stderr)
cands = [d for d in r.detnames if 'jungfrau' in d.lower()]
print(cands[0] if cands else "")
EOF
)
echo "jungfrau candidate detector: '${DETNAME}'" >> "${LOG}"

if [ -z "${DETNAME}" ]; then
    echo "NO JUNGFRAU-LIKE DETECTOR FOUND -- stopping without running test" >> "${LOG}"
    echo "=== R78 MARKER: exit_code=99 (no jungfrau detector) ==="
    exit 0
fi

T0=$(date +%s)
python /sdf/scratch/users/a/ajshack/lcls2/psana/psana/gpu/test_jungfrau_calib.py \
    -e mfx100852324 -r 78 -n 20 --det "${DETNAME}" >> "${LOG}" 2>&1
RC=$?
T1=$(date +%s)

echo "=== R78 MARKER: exit_code=${RC} wall_seconds=$((T1 - T0)) ==="
echo "=== r78 test complete ==="
