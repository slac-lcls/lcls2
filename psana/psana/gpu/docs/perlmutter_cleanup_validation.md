# GPU parser cleanup validation on Perlmutter

Date: 2026-09-14. Source baseline: `2631e30d0` on
`codex/psana2-gpu-xtc-parser`, fetched from `origin` into a dedicated worktree.

## Environment and reproducibility

Checkout: `/global/u2/m/monarin/lcls2_worktree/codex/psana2-gpu-xtc-parser`.
The detached `psana2-gpu-two-phase-retire` checkout and its dirty files were
left untouched. No production parser, routing, pinning, or lifetime code was
changed for validation.

Build on a login node:

```bash
source ~/goodstuffs/bashrc
source ~/psana-nersc/activate_psana_build_env.sh ~/.conda-envs/psana-build
./build_psana.sh -j 8
source ~/activate_psana_gpu.sh codex/psana2-gpu-xtc-parser
```

Two job-local corrections are required with the current helper and this branch:

```bash
# The helper refers to a CUDA shim that is absent on this host. The default
# toolkit is CUDA 13.2, while installed CuPy needs CUDA 12 NVRTC.
module load cudatoolkit/12.9
export CUDA_PATH="$CUDA_HOME"
# CalibConstants appends /calib_ws/ itself on this branch.
export LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws
unset SIT_PSDM_OFFSITE
```

The calibration endpoint returned the experiment's detector collections.
A CPU preflight read produced Jungfrau shape `(32, 512, 1024)`, gain-bit
values `[0, 1, 2, 3]`, and 16,768,391 nonzero calibrated pixels at timestamp
`4806054694728932589` with common mode disabled. Missing Kerberos credentials
emitted warnings but did not prevent public calibration reads.

Both the smoke and pixel-exact acceptance use staged `mfx100848724` run 51:

```bash
export PSANA_GPU_TEST_EXP=mfx100848724 PSANA_GPU_TEST_RUN=51
export PSANA_GPU_TEST_DIR=/pscratch/sd/p/psdatmgr/psdm/mfx/mfx100848724/xtc
export PSANA_GPU_TEST_SMD_GLOB="$PSANA_GPU_TEST_DIR/smalldata/mfx100848724-r0051*.smd.xtc2"
export MPICH_GPU_SUPPORT_ENABLED=0
```

Run 77 was absent from its expected staged path. The transport smoke accepts
any verified Jungfrau run; the nonzero acceptance reference remains run 51.

## Results

GPU job `58317405` ran on `nid001532` with four NVIDIA A100-SXM4-40GB
GPUs. Runtime: Python 3.9.20, Cray MPICH 9.0.1.498, CuPy 13.6.0,
CUDA runtime/NVRTC 12.9, and KvikIO 24.08.02. Both
`kvikio.defaults.compat_mode() == True` and
`DriverProperties.is_gds_available == False` confirmed CPU fallback.

| Check | Result |
| --- | --- |
| Fresh native build and imported API | PASS; `config_names()` present, `raw_descriptors()` absent, no installed `dgram_layout.py` |
| GPU unit suite (CPU-only) | 165 passed in 6.54 s |
| Fast CUDA suite | 5 passed, 6 slow cases deselected, in 15.83 s |
| Six-rank smoke, 50 events, batch 5, pool depth 2 | PASS, exit 0; 50 unique timestamps; all 4 BDs active |
| One-event smoke, batch 1 | Expected INCOMPLETE, exit 2; 1 unique timestamp, 1 of 4 BDs active |
| Invalid `--max-events 0` | Rejected before collectives; `srun` returned exit 2 |
| Real-data pixel-exact suite | 6 passed, 2 fast cases deselected, in 364.40 s |

All 176 pytest cases passed without skips. Slurm recorded job `58317405`
as `COMPLETED`, exit `0:0`, elapsed `00:10:31`. Steps `.3` and `.4` are
recorded as `FAILED 2:0` because their nonzero statuses were intentional; the
parent script checked those exact statuses and printed `VALIDATION PASS`.

Measured smoke participation:

| World rank | Role | CUDA PCI bus | Events |
| --- | --- | --- | --- |
| 0 | SMD0 | none | 0 |
| 1 | EB | none | 0 |
| 2 | BD | `0000:03:00.0` | 15 |
| 3 | BD | `0000:41:00.0` | 15 |
| 4 | BD | `0000:82:00.0` | 10 |
| 5 | BD | `0000:C1:00.0` | 10 |

Initial job `58317342` stopped at all five fast CUDA tests because
`libnvrtc.so.12` was unavailable with the default toolkit and missing shim.
This was resolved by the job-local CUDA 12.9 setup above; its original logs
are retained in `validation/perlmutter-20260914/attempt-58317342/`.
Job `58317324` was canceled while pending to correct the calibration URL,
before consuming an allocation.

## Commands and saved evidence

The complete submitted script, runtime probe, and full logs are retained at
`validation/perlmutter-20260914/` in the checkout above. Generated build and
validation artifacts are not committed. Test commands:

```bash
python -m pytest -q psana/psana/tests/gpu/unit
# On the allocated node, one rank, PS_PARALLEL=none:
python -m pytest -q -rs -m 'gpu and not slow' psana/psana/tests/gpu/integration
python -m pytest -q -rs -m slow psana/psana/tests/gpu/integration/test_pixel_exact.py
```

Slurm allocation: `-A lcls_g -C gpu -q debug -N 1 --gpus-per-node=4
--ntasks-per-node=6 --cpus-per-task=2 --time=00:30:00`.
Single-rank commands use `srun --mpi=cray_shasta -N1 -n1 --ntasks-per-node=1
-c2 --gpus-per-node=4 --gpu-bind=none`.

For the MPI checks, set `PS_EB_NODES=1 PS_SRV_NODES=0 PS_EB_NODE_LOCAL=0
PS_PARALLEL=mpi SLURM_GPUS_ON_NODE=4`, then launch:

```bash
srun --mpi=cray_shasta -N1 -n6 --ntasks-per-node=6 -c2 \
  --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 --time=00:10:00 \
  bash -c '
    rank=${SLURM_PROCID:?}
    if (( rank >= 2 )); then
      export CUDA_VISIBLE_DEVICES=$((rank - 2))
    else
      export CUDA_VISIBLE_DEVICES=""
    fi
    exec "$@"
  ' gpu-smoke python psana/psana/gpu/scripts/gpu_multi_rank_smoke.py \
  --max-events 50 --batch-size 5
```

Repeat with `--max-events 1 --batch-size 1` for expected exit 2 (INCOMPLETE),
and with `--max-events 0` for expected exit 2 (argument rejection). The saved
batch script captures each `srun` exit status and fails if any differ from
expectations. The SDF-specific `run_multi_gpu_test.sh` is not used.

## Scope

The CPU tests cover shared-stream routing, segment order, parser metadata,
budgets, and transition/lifetime contracts. The five fast CUDA cases cover
tracked-fixture field parsing, stream-specific Names resolution, parser-slot table
reuse/accounting, raw gathering/calibration, and float32 passthrough. The six real-data
cases compare 13 events each against CPU raw and exact float32 calibration,
including general fields, slot reuse, partial tails, three D2H chunk policies,
and hybrid stream mirroring.

These checks do not close the existing multi-EB ownership, result-lease fan-out,
fixed-allocation accounting, or other entries in [known_issues.md](known_issues.md).
They do not establish true GDS, GPU sharing, throughput, the user-task C ABI, or
real-device transition stress beyond the acceptance cases described above.
