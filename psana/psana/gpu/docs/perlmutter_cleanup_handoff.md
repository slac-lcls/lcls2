# GPU parser cleanup: Perlmutter handoff

Date: 2026-09-14. Continue branch `codex/psana2-gpu-xtc-parser` from `origin`.
The SDF implementation is ready for review; Stage 3 still needs real MPI/GPU
validation. Do not merge or delete this branch as part of the handoff.

## Completed work

- Stage 1, `55b69ab11`: removed unused `dgram_layout.py`, its sole unit test,
  and the CPU `Dgram.raw_descriptors()` implementation/private helpers.
  `Dgram.config_names()` remains required and intact.
- Stage 2, `7718f8622`: removed the old integration-only parser-status scan,
  clarified API/adapter/budget test contracts, and replaced the generated
  xpptut dependency with the tracked `test_data/chunking` fixture.
- Stage 3, the commit containing this handoff: narrowed the manual MPI smoke
  check to event counts, timestamp uniqueness, raw-result availability,
  measured device identity, BD participation, and completion. Removed pixel
  sanity checks and benchmark reporting. Hardened the Slurm launcher and
  added CPU regression checks for reporting/argument/exit-status handling.

The parser, DataSource routing, detector processing, and lease/budget runtime
were not changed by Stages 2 or 3. Keep GPUBAT1 read descriptors, Configure
tables, field locators, and all numerical/lifetime acceptance coverage.

## Validation on SDF

- Native psana rebuild after Stage 1: passed; verified the imported extension
  still exposes `config_names()` and no longer exposes `raw_descriptors()`.
- Main psana suite after Stage 1: 219 passed, 10 skipped, 8 deselected, one
  previously observed unrelated failure in
  `test_extract_subset_xtc2.py::test_generated_smalldata_can_be_read_back`.
  Calibration startup messages precede the correct event count `2` on stdout,
  while the test expects stdout to equal exactly `2`.
- Byhand MPI suite after Stage 1: 4 passed.
- After Stage 2: 153 GPU unit cases, 5 fast A100 device cases, and all 6 slow
  DataSource pixel-exact cases passed. Slow tests used public
  `mfx100848724` run 51 and KvikIO CPU fallback, not GDS.
- After Stage 3: 165 GPU unit cases passed (including 12 new smoke-harness
  cases). Bash syntax, Python compilation, and `git diff --check` passed.
  Total GPU pytest inventory is now 176 cases: 165 CPU-only, 5 fast CUDA,
  6 slow CUDA/external-data. The actual multi-rank smoke run is separate.
- SDF Slurm job `38221915` eventually received resources on `sdfampere019`,
  but failed before application launch with `Socket timed out on send/recv
  operation`. This is not a smoke-test PASS or an observed psana failure.
  SDF logs were `/tmp/gpu_cleanup_stage3_mpi.log` and
  `/tmp/gpu_cleanup_stage3_unit.log`; these are not portable artifacts.

## Resume on Perlmutter

1. Use the `psana-perlmutter` and `psana2-gpu` skills. Inspect existing
   worktrees/dirty files before fetching and checking out this branch. Use a
   dedicated worktree; do not overwrite another task's checkout or install.
2. Build on a login node using the checkout-local `build_psana.sh -j 8`.
   A native rebuild is required when coming from before Stage 1. Activate
   the matching checkout-local `install_psana`, not another worktree's prefix.
   Follow the Perlmutter skill's conda/compiler/CUDA-shim setup.
3. Verify `psana.__file__`, `psana.dgram.__file__`, MPI library, CuPy/CUDA, and
   visible GPUs. An incremental install can leave a stale installed
   `gpu/dgram_layout.py`; check and remove only that obsolete installed copy
   if present (do not clean unrelated directories).
4. Run CPU unit tests and the five fast CUDA tests first. The xpptut fixture
   is tracked and requires no experiment staging.
5. Run the manual multi-rank smoke check, then an intentionally short run to
   exercise INCOMPLETE reporting. Verify failures propagate to the caller.
6. Run the six slow acceptance cases if run-51 data/calibration are available.
   Report absent data separately from failures; do not substitute run 77 for
   the nonzero pixel-exact reference.

```bash
python -m pytest -q psana/psana/tests/gpu/unit
# On a GPU compute node:
python -m pytest -q -rs -m "gpu and not slow" psana/psana/tests/gpu/integration
python -m pytest -q -rs -m slow psana/psana/tests/gpu/integration/test_pixel_exact.py
```

Use `PSANA_GPU_TEST_EXP`, `PSANA_GPU_TEST_RUN`, and `PSANA_GPU_TEST_DIR` for
slow acceptance. Use `PSANA_GPU_TEST_SMD_GLOB` for the manual MPI smoke test.
Verify staged data under `/pscratch/sd/p/psdatmgr/psdm/<instr>/<exp>/xtc`;
the exact available experiment/run paths have not been checked from SDF.
Keep `SIT_PSDM_OFFSITE` unset and use the calibration URL provided by the
Perlmutter activation helper.

## Launch portability: important

`run_multi_gpu_test.sh` currently uses SDF's `ampere` partition, `lcls`
account, and `--mpi=pmix`. Even its existing-allocation mode uses PMIx.
Do not run that wrapper unchanged on Perlmutter with Cray MPICH.

For initial validation, execute `gpu_multi_rank_smoke.py` directly inside a
GPU allocation using the site's verified MPI plugin (`srun --mpi=list`;
normally `cray_shasta`). Alternatively, make a small follow-up launcher
portability change and extend the existing launcher tests; do not change
production GPU pinning/routing merely to make the smoke test pass.

Example inside a one-node, four-GPU allocation, after runtime activation:

```bash
export PS_EB_NODES=1 PS_SRV_NODES=0 PS_EB_NODE_LOCAL=0 PS_PARALLEL=mpi
export MPICH_GPU_SUPPORT_ENABLED=0
export SLURM_GPUS_ON_NODE=4
# Set PSANA_GPU_TEST_SMD_GLOB to a verified staged run-77 SMD glob first.
srun --mpi=cray_shasta -N 1 -n 6 --ntasks-per-node=6 -c 2 \
  --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 --time=00:15:00 \
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

This is one SMD0, one EB, and four BD/GPU workers. A PASS requires every
requested BD to process events on a distinct measured device. Exit 2 means
delivery checks passed but participation was incomplete; increase events or
reduce batch size. `--max-events 1 --batch-size 1` should be INCOMPLETE.
Slurm may add termination diagnostics for nonzero results; preserve them.

The single-node smoke check does not validate multi-EB scheduling, GPU
sharing, true GDS, performance, or the proposed user-task C ABI.

## Design context and remaining scope

The user-task C ABI is proposed, not implemented. Cleanup did not remove C
ABI tests: none existed. Retain the result/lease, BeginStep, budget, field,
and numerical tests that protect its underlying infrastructure.

Before implementing tasks, reconcile the two-argument entry point in
`features/psana2-gpu-d2h-pipeline`'s `gpu_task_c_abi_design.md` with the local
`docs/proposals/user_gpu_pipeline.md`, and resolve automatic-D2H/device-copy
semantics. Current `on_gpu` rejects a released device slot; the older proposal
promises device-copy access with prefetch. Arbitrary task output/scratch
arenas, invocation/error handling, ABI compatibility, and output D2H need
their own future implementation/tests, not restoration of retired layout code.

Preserve unrelated untracked SDF benchmark files. Do not restore deleted
prototype modules, create a new parser ABI, or expand Stage 3 into runtime
changes without review.
