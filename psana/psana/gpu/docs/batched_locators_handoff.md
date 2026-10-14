# Batched GPU field location: task handoff

Historical handoff retained below. As of 2026-09-18, locator implementation,
validation, and matched A/B/B+ measurement are complete; the original status
statements below describe preparation on 2026-09-17 only. See the
[implementation review](batched_locators_review.md) and the current
[batched canonical gather plan and bulk integration note](batched_canonical_gather_plan.md).

Prepared 2026-09-17 on SDF. This is a proposed implementation plan, not an
implemented optimization. Only the new branch/worktree and this document have
been created. No build, tests, benchmark submission, commit, or push was done
for this handoff.

## Objective and agreed workflow

Reduce CPU orchestration of GPU field location by grouping configured field
handles by XTC stream and decoding them with batched GPU launches. Keep this
first change separate from batched gathering and from bulk-read integration.

The user agreed to: **B-first development and measurement, followed by a merge
into the current bulk-enabled history.** Do not rewrite the published branch
or replay the bulk commits merely to obtain a linear history.

- New worktree: `/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-batched-locators`
- New branch: `codex/psana2-gpu-batched-locators`
- Base/current HEAD: `803a70011d18168200927e279cbeaca90568e13f` (benchmark B)
- Existing bulk worktree: `/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline`
- Bulk branch: `codex/psana2-gpu-xtc-parser`
- Bulk HEAD at handoff: `8f94e3c7bc4643309d022cb5c1dd7e85805d3ac4`
- This handoff is intentionally uncommitted. No optimization exists yet.

The existing bulk worktree has many unrelated untracked logs, benchmark tools,
and reports. Leave them untouched. In particular, do not modify the frozen
benchmark A/B source worktrees or their installation prefixes.

## Evidence motivating this change

Read the full report in the existing bulk worktree:

`psana/psana/gpu/docs/performance/parser_warm_ab_timing_sdf.md`

Artifacts and scripts live there under:

`validation/perf-acceptance-20260916/`

Successful measurement: `job-38478311-warm-ab/`. A is legacy addressing at
`f52e90cc6`; B is the integrated parser without bulk read at `803a70011`.

Measured workload: real `mfx101210926`, run 387, 10,000 events, Jungfrau 32
segments across streams 5-9, batch size 20, pool depth 1, 8 GiB GPU budget,
one SMD0/one EB/one BD/one A100. KvikIO compatibility ON (CPU fallback, not
GDS), eight reader threads, 1 MiB task size. Warm node-local data, no user D2H.

- Clean median elapsed: A 21.768 s; B 33.451 s (11.683 s additional).
- Steady host field-location scope: B 5.179 s.
- Steady host field-access/dependency/gather scope: A 0.844 s; B 7.009 s.
- Separate trace: 500 XTC walker launches take only 0.0123 s GPU time total.
- B locates 192 handles per subbatch: 96,000 location kernels, 96,000 status
  fill kernels, and 96,000 locator memsets across the run.
- GPU location plus initialization totals approximately 0.905 s in that trace.
- Calibration GPU time is essentially unchanged; raw read task counts match.

Host scopes are elapsed submission scopes, not pure CPU utilization. GPU times
come from separate captures and overlap host work. Do not subtract or add these
numbers to promise an exact speedup. Nsight reported possible collection gaps;
expected main operation counts were audited. Use clean runs for throughput.

## Current B implementation and source map

Paths below are relative to this new worktree unless otherwise stated.

- `psana/psana/gpu/gpu_events.py`: constructs the configured GPU-detector field
  handle list and `GpuXtcBatchPool`.
- `psana/psana/gpu/gpudgram/config.py`: Configure tables and `GpuFieldHandle`.
- `psana/psana/gpu/gpudgram/batch.py`: reusable parser storage, `parse()`,
  metadata allocation, and memory estimates.
- `psana/psana/gpu/gpudgram/parser.py`: walker, `GpuEventBatch.locate()`,
  device locators, and CUDA field-location code.
- `psana/psana/gpu/gpu_input.py`: event/stream views and field access.
- `psana/psana/gpu/gpu_detector.py`: canonical gathering and calibration;
  leave these algorithms unchanged in the first optimization.

Configure supplies types, ranks, field order, Names identity, and detector/
segment/algorithm membership. Event ShapesData supplies actual dimensions.
Stream IDs are active psana stream indices, not CUDA stream identifiers.

One walker launch covers all input dgrams, one thread per dgram. Afterwards,
`parse()` loops over all configured GPU-detector event field handles. Each
uncached `locate(handle)` initializes a locator buffer, launches one field
kernel, records a ready event, and caches a locator object. The list spans
configured GPU streams, even when some are absent in the current input.
It does not automatically include every CPU-only detector.

The existing location kernel parallelizes over dgram/ShapesData-reference
pairs for one handle, uses atomics for duplicate detection, and computes the
target offset by traversing preceding fields. Later gets use cached locators;
they do not decode the field again. All decoded offsets stay on the device.

## Proposed first implementation

1. Build/upload a run-scoped numeric handle table grouped by XTC stream, with
   stream-to-handle ranges and stable CPU handle-to-output indexing. Include
   the new device tables in fixed-cost budget accounting.
2. Use reusable owner-local locator backing storage, conceptually
   `[handle, dgram, locator_columns]`, exposing compatible per-handle views.
   Preserve logical handle identity and window-local dgram rows. Capacity
   strides must be explicit when buffers are reused for smaller inputs.
3. One initialization kernel fully initializes locator rows and sets absence
   status. A separate decoding kernel follows on the same CUDA stream; avoid
   unsafe cross-block initialization/decoding races within one launch.
4. One combined decoding launch distributes work across the relevant handles
   and input references. Use stream-grouped handles so a dgram does not scan
   unrelated streams' fields. No GPU-to-CPU metadata readback for scheduling.
   Correctness comes before tuning block/warp geometry.
5. Preserve existing error/duplicate/bounds semantics. Initially retain the
   existing offset algorithm; decoding all fields of a Names block in one pass
   is a possible later optimization, not required in this stage.
6. Record one shared locator-ready CUDA event after decoding; per-handle
   locator objects expose their compatible slices and the shared dependency.
   Preserve supported lazy lookup behavior and cross-stream correctness.

Target sequence per nonempty parsed input: walker kernel, initialization
kernel, decoding kernel, then locator-ready event record. The event record is
not a fourth kernel. Empty/no-handle cases should avoid unnecessary launches.
Gathering/calibration are separate and unchanged. Do not enlarge EB batches,
execution subbatches, pool depth, or admission budgets to obtain the speedup.

Initially dense output can still initialize absent-stream rows to maintain
the API. Compacting storage is optional later work. Do not confuse avoiding
unrelated decoding with eliminating every unused output row.

## Bulk-read compatibility: design now, integrate later

There are 13 commits after B through the recorded bulk HEAD. Inspect them with
`git log --reverse --oneline 803a70011..8f94e3c7b`. Notable changes:

- `a5f07ee38`: input-window ownership independent of execution slots.
- `763a8df1b`: pre-I/O admission and full replacement-allocation accounting.
- `694b9ff2b`: resident streams read/parsed once across execution subbatches.
- `4a26bc640`: mean nonempty dgram-size admission priority.

The parser kernel is unchanged across this range; `parser.py` changes only its
ownership documentation. `gpudgram/batch.py` adds `parse_window()`, ownership,
failure draining, `allocation_requirements()`, and free-buffer trimming.

Required integration invariants:

- Storage and completion events belong to each parsed InputWindow, not to an
  execution slot reused while resident input remains live.
- Resident and transient windows have independent `data_gpu` bases and row
  numbering. Preserve descriptor-derived byte offsets after read coalescing;
  never derive them from stream order or file offsets alone.
- One parse sequence per newly read input window, not universally per execution
  subbatch. For five execution subbatches: no residency usually means five
  parses; mixed residency may mean one resident plus five transient parses;
  all-resident input can need only one parse.
- Keep `GpuEventDgrams.from_windows()` composition and consumer leases intact.
- Update allocation estimates, full old-plus-new growth reservations, memory
  reporting, and trimming for combined locator storage. Do not double-count
  per-handle views or undercount run-scoped handle/work tables.
- `InputWindow` currently collects individual locator readiness events. Adapt
  this to avoid duplicate waits for the shared event, while retaining all
  genuinely separate lazy-parser and downstream consumer dependencies.
- Preserve failure ownership if completion cannot be established. No release
  on generator advancement and no early slot/input reuse.

Later integration should use a new branch from the then-current bulk branch
and merge this optimization branch, with explicit conflict review. Preserve
published hashes. Do not perform that merge or publish without coordination.

## Validation and environment

Start by reading applicable psana2-gpu, psana-xtc-config, gpu-cuda-python, and
psana skills. Current source is authoritative where older skill prose differs.

This fresh worktree has NOT been built or activated. Establish a separate,
worktree-local installation and verify imported `psana`/GPU module paths.
Do not accidentally run the existing bulk install or edit the frozen B build.
The parent benchmark activates `setup_env.sh` and `install_psana/activate.sh`
in the parent worktree; do not assume those local/generated files exist here.

Correctness first:

- CPU unit tests and GPU locator tests; compare exact offset/size/type/shape/
  status results against the existing parser/CPU reference.
- Multiple streams/detectors, absent streams/fields, sparse/noncontiguous
  dgram rows, reused Names IDs across streams, scalar/array fields, damaged or
  duplicate ShapesData, bounds failures, empty inputs, tails, and buffer reuse.
- Cross-stream consumption, delayed consumers, allocation failure/growth, and
  parser storage retention through completion.
- Pixel-exact raw/calibration tests and unchanged event counts/order.
- Run `pytest psana/psana/tests/` and `pytest psana/psana/tests/byhand_*` for
  psana core changes; distinguish environmental failures from regressions.

Tests already at B are under `psana/psana/tests/gpu/unit/` and `integration/`,
including `test_gpudgram.py`, `test_gpudgram_device.py`, `test_gpu_input.py`,
`test_gpu_result_lifetime.py`, and `test_pixel_exact.py`. Bulk integration later
also needs the input-window, admission, residency, and bulk-read test suites
present on the bulk branch. Do not restore obsolete legacy-layout tests.

Benchmark B versus B+optimization, warm only, identical workload/settings,
alternated repeated clean runs; use CPU/NVTX and Nsight separately to verify
launch counts and locate remaining cost. No hardware thresholds in pytest.
Use a new harness/output directory: the existing `run_warm_ab.py` hardcodes
historical A/B prefixes, and its sbatch script changes to the parent worktree.
AST timing hooks also depend on old function structure. Adapt a separate
copy deliberately; merely running the old script will not measure this branch.

Existing scripts, reports, calibration references, and logs remain readable in
the parent validation directory. They are mostly untracked and were not copied
here. Previous node-local data may no longer exist; verify/restage within an
allocation, never purge shared data caches. Record actual import paths and
commit/dirty state. No GPU jobs were launched for this handoff.

## Next concrete step and review boundary

Audit B's `parse()/locate()` and bulk-head ownership/accounting methods, then
implement only stream-grouped batched field location with focused correctness
tests. Prepare an isolated B/B+ warm measurement. Report results before moving
on to batched gathering or merging bulk history. Keep implementation, evidence,
and later integration in reviewable commits when the user authorizes commits.
