# GPU parser cleanup: Perlmutter handoff

Date: 2026-09-14. Continue branch `codex/psana2-gpu-xtc-parser` from `origin`.
The SDF implementation and Perlmutter cleanup validation are complete. See
[the Perlmutter validation report](perlmutter_cleanup_validation.md) for all
176 passing pytest cases, four-GPU MPI participation, expected nonzero exit
checks, and the required job-local environment corrections. Do not merge or
delete this branch as part of the handoff.

## Completed work

- Stage 1, `55b69ab11`: removed unused `dgram_layout.py`, its sole unit test,
  and the CPU `Dgram.raw_descriptors()` implementation/private helpers.
  `Dgram.config_names()` remains required and intact.
- Stage 2, `7718f8622`: removed the old integration-only parser-status scan,
  clarified API/adapter/budget test contracts, and replaced the generated
  xpptut dependency with the tracked `test_data/chunking` fixture.
- Stage 3, `8d66f344a`: narrowed the manual MPI smoke
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
   Follow the Perlmutter skill's conda/compiler setup. The current CUDA shim
   path is absent on Perlmutter; after activation load `cudatoolkit/12.9` and
   set `CUDA_PATH="$CUDA_HOME"` as recorded in the validation report.
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
Run 51 is verified at
`/pscratch/sd/p/psdatmgr/psdm/mfx/mfx100848724/xtc`; run 77 is absent at its
expected staged path. Both the smoke and pixel-exact runs used run 51.
Keep `SIT_PSDM_OFFSITE` unset. After activation, override the helper with
`LCLS_CALIB_HTTP=https://pswww.slac.stanford.edu/ws`: this branch appends
`/calib_ws/` itself, so the helper's older value duplicates that suffix.

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
# Use the staged run-51 dataset verified on Perlmutter.
export PSANA_GPU_TEST_SMD_GLOB="/pscratch/sd/p/psdatmgr/psdm/mfx/mfx100848724/xtc/smalldata/mfx100848724-r0051*.smd.xtc2"
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

## Full parser integration: implemented baseline

The parser is already integrated into the DataSource event loop, not just the
standalone driver. The remaining work below is validation, hardening, API
completion, and performance work; it is not another raw-addressing migration.

- CPU Configure parsing builds stream-indexed `GpuStreamConfigTable` tables
  and name/field handles. `GpuXtcBatchPool` uploads the configuration once and
  shares it across execution slots.
- EventBuilder supplies GPUBAT1 event/stream file-offset and size descriptors.
  `KvikioGpuReader` reads whole dgrams into a slot's `data_gpu`. CPU metadata
  supplies logical event/stream indexing and device destinations, not field
  offsets derived from parsing event payloads.
- Each batch uploads initial dgram records. The GPU walker fills header/status
  columns and ShapesData references; field decoding produces device locators.
  `evt.gpu.dgrams[stream_id]` links the event to those slot-owned tables.
- General detector field access resolves detector, segment, algorithm, and
  field handles. Fields may have different shapes across segments; dense
  ordering/shape adjustment and calibration belong to detector adapters, not
  the reader. Existing calibration consumes parser-derived addressing.
- `gpu_det` keeps selected streams GPU-exclusive; `hybrid_det` explicitly
  mirrors selected streams to both paths, paying duplicate I/O. Multiple GPU
  detectors may share a stream. Do not remove these ownership rules merely
  to optimize reads.

Source map: [architecture](architecture_overview.md),
[parser details](gpu_xtc_parser.md), `gpu_events.py`, `gpu_kvikio_read.py`,
`gpudgram/config.py`, `gpudgram/batch.py`, `gpudgram/parser.py`, and
`gpu_input.py` (source paths relative to `psana/psana/gpu`).

## Remaining integration roadmap

These are proposed follow-up stages, separate from the completed cleanup
Stages 1–3. Do not implement them all as one change. Keep the verified
[issue register](known_issues.md) synchronized as issues are closed.

### A. Establish the Perlmutter baseline

Complete the validation above before changing runtime behavior. Exercise
general fields and calibrated results, shared-stream GPU detectors,
multi-stream/segment detectors, hybrid CPU/GPU event identity, partial tails,
and transition drains. Retain the tracked xpptut coverage and nonzero real-data
pixel comparison. Record actual I/O mode; fallback success is not GDS proof.

### B. Close lifetime and resource-accounting gaps

- Make result leases wait for every consumer stream, as input leases already
  do. Register the actual `on_gpu` copy stream or enforce its documented stream.
- Reserve calibration/geometry fixed allocations explicitly, including correct
  CUDA-IPC ownership. Reject oversized events before starting slot allocation;
  resolve the unsupported subbatch-budget override and estimate floor.
- Before multi-EB scaling, replace EB-local GPU identity/leader/budget decisions
  with node-wide BD and per-device coordination. Test uneven sharing and more
  than one EB group. Single-EB smoke success does not close this issue.

These are integration safety issues, not reasons to restore CPU field offsets.
Focus changes in `context.py`, `gpu_input.py`, `gpu_stream.py`,
`gpu_budget.py`, `gpu_events.py`, `gpu_mpi.py`, and `psexp/mpi_ds.py` as appropriate.

### C. Bulk reads for small detectors

**Integration update (2026-09-14):** Stages 1 and 2 of the
[bulk-read plan](proposals/bulk_read_plan.md) are implemented. Setting
GPU reads now coalesce adjacent ranges by default within an existing execution
subbatch, with immutable file/chunk resolution and pending-I/O cleanup.
CPU validation and all 14 Stage 2 GPU integration cases passed; GPU validation
used KvikIO CPU fallback with GDS unavailable. Independent fast/slow
input ownership and resident-fast scheduling remain later stages.

**Comparison mode (`gpu_bulk_read=False`):** `KvikioGpuReader.issue_batch()`
submits one `pread` per
nonempty dgram descriptor. `_build_desc_table()` packs their payloads back to
back. KvikIO's `task_size` is not a psana-level coalescing plan.

**First implementation design:** introduce a bounded read planner between
GPUBAT1 descriptor resolution and KvikIO submission, within one execution
subbatch/slot. Keep two distinct structures:

- Physical read ranges: resolved file/chunk identity, file offset, read length,
  and destination offset. Merge adjacent ranges first; allow bounded gaps only
  under an explicit over-read policy. Never merge across files/chunks.
- Logical dgram records: preserve event/stream identity and dgram size, but
  rebase each device offset into the containing physical range. For example,
  a dgram at file offset `f` inside a range starting at `r`, loaded at device
  offset `b`, has device offset `b + (f - r)`.

The GPU should walk the known logical dgrams, not interpret gap bytes as selected
events. Physical read ordering may change without changing logical event rows.
Detectors sharing a stream must reuse that stream's dgram, not issue duplicate
reads. CPU planning uses SMD offsets/sizes only; no payload parsing or parser
metadata round trip is needed.

Update admission and buffer sizing to charge **physical fetched bytes**,
including gaps/padding, plus parser tables and detector allocations. Do not
derive total input capacity from the last logical row once physical and logical
ordering differ. Keep `PendingBatch`, file handles, futures, input bytes, and
parser tables alive until all I/O and CUDA consumers finish, including partial
submission, short-read, and exception paths. Audit the current stream-keyed
file cache for chunk changes. Preserve BeginStep/EndRun drain boundaries.

Start in `gpu_kvikio_read.py`; adjust subbatch admission in `gpu_events.py` and
the descriptor-to-record boundary in `gpudgram/batch.py` only where necessary.
The field-access API and GPUBAT1 wire format need not change for this first step.

Acceptance: compare coalesced versus per-dgram reads byte-for-byte for contiguous,
gapped, interleaved-stream, shared-stream, missing-data, chunk-boundary, and tail
cases. Test short reads, memory limits, and delayed consumers. Then benchmark
small-detector-heavy data separately from large-detector data, recording request
count, useful/fetched bytes, issue-to-completion time, end-to-end throughput,
CPU overhead, and peak device/host memory. Validate fallback and true GDS
separately; the current wait-only I/O timer is not total read latency.

### D. Finish user-facing event and host-handoff behavior

- Add GPU `RunParallel.steps()` through the unified event/step envelopes.
  `run.events()` already handles BeginStep; a second independent GPU event path
  would duplicate lifetime and transition logic.
- Support `smd_callback` only after callback selection produces coherent CPU
  and GPU packets. The current explicit rejection is intentional protection.
- Generalize automatic D2H beyond dense float32 calibrated results using
  declared output shape/dtype and a pinned-host byte budget. Define whether
  raw parser inputs remain available or are released after host handoff;
  eagerly attached inputs currently prevent fully host-backed early retirement.
- Device-kernel consumers can already use device locators without a host copy.
  Python `GpuFieldResult` access still fetches a small locator row to construct
  dynamically shaped arrays. Document that boundary, and optimize/cache or add
  a device-oriented consumption path only after profiling; do not claim every
  convenience API is metadata-D2H-free.

### E. Measure parser parallelism and broader detector workflows

Keep the current iterative one-thread-per-dgram walker as the correctness and
performance baseline; different dgrams already run in parallel. Compare it with
cooperative ShapesData/block/warp approaches and multi-stream scheduling using
the same data and output checks. Measure parser-only and end-to-end costs before
choosing a design. Retain malformed-XTC, depth/capacity, missing-field, and shape
validation; do not trade explicit errors for silent truncation.

Image publication, additional detector calibration adapters/common-mode, and
user-task C ABI execution are separate workflow extensions, not prerequisites
for general raw-field parsing. Prioritize them by actual consumers after the
baseline and safety work. Bulk-read work can precede these extensions.

## User-task design context

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
