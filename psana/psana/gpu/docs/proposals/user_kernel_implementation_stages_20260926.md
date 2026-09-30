# User-kernel implementation stages

2026-09-26. Implementation tracking; the initial planning checkpoint was
`480f7074c`. See [Stage 1 findings](../user_kernel_stage1_findings_20260926.md)
for the raw-preparation extraction and validation evidence.

- Task branch: `codex/psana2-gpu-user-kernels`.
- Parent branch: `codex/psana2-gpu-bulk-batched-integration`.
- Branch point: `6e5b8d0d096c769f9538064635db72a0b72d4bca`.
- Active worktree: `/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline`.
- The existing `codex/psana2-gpu-user-callback` worktree is not a dependency.
- Stage 1 implementation checkpoint: `7d0b5941e`. Existing untracked experiments,
  logs, and validation directories remain outside task changes.

The [canonical proposal](user_gpu_pipeline.md) defines the contract. This file
tracks implementation order and review gates; it does not introduce a second
design. The [source note](calib_azint_callback_sources_20260926.md) pins the first
calibration/azimuthal-integration example. The [handoff](user_kernel_integration_handoff_20260926.md)
records the inherited validation baseline, which is not callback acceptance.

## Priorities and boundaries

Simple structure, minimum viable code addition, then measured optimization.
Extend the existing producer, leases, and host delivery. Preserve batched parser
and gather launches. No registry, task graph, second scheduler, managed user
device arenas, or advance output schemas. The user chooses output size, shape,
dtype, and publication cadence. Psana retains registered owners and uses CUDA
events for completion; host access waits for D2H completion as well as production.

The first public API remains `GpuTask(function, inputs, calibconst)` passed as
`DataSource(gpu_fn=task)`. Plan to export `GpuTask` from `psana.gpu` with no CUDA
initialization at import or declaration time. Following the user's post-Stage-1
direction, remove built-in calibration completely from the GPU runtime.
`gpu_fn=None` retains input routing/parsing and explicit parsed-field access,
with no automatic calibration or output D2H. Keep callable state in user code;
no setup/teardown framework is needed
for the first example. Device initialization belongs on the assigned BD.

The [removal dependency audit](calibration_runtime_removal_20260926.md) finds
no architectural blocker. Stage 1b removes the legacy path before continuing
task wiring; CPU/hybrid calibration and source dictionaries remain available.

## Stages and exit gates

Each stage is a reviewable unit and may need several commits. No stage is
complete on CPU mocks alone when its contract depends on CUDA lifetimes.

| Stage | Deliverable and primary source areas | Exit gate |
|---|---|---|
| 1. Raw preparation boundary | Extract requested dense raw/presence preparation from `gpu_detector.py`; separate legacy calibration setup in `gpu_events.py`, `gpu_calib.py`, and MPI setup | Raw-only preparation works without pedestals or legacy calibration/geometry allocations; default calibration remains pixel-exact; existing batched gather structure is preserved |
| 1b. Remove built-in calibration | Delete automatic GPU calibration/geometry setup, producer, refresh, fixed-pair IPC, synthetic outputs, and image-only D2H; migrate fixtures | Serial/MPI input processing works without pedestals or legacy work; CPU/hybrid calibration and collectives remain correct; no-callback mode has parsed inputs but no synthetic outputs |
| 2. Task declaration and requested constants | Small task configuration type, public export, `ds_base.py`/Run/MPI plumbing; selective original-value uploads with existing admission accounting | Empty requests upload nothing; gain-only requests need no pedestals; dtype/shape/segment mapping survive; refresh drains prior users; CUDA exists only on assigned BDs |
| 3. Producer dispatch and owner retention | Producer context and callback invocation in `EventPool.submit()`; integrate selected event identity and existing leases | Exactly one call per eligible selected event, including max-events tails; no-output and scratch-only work are safe; failures drain or retain owners; no repeated per-field parser/gather submissions |
| 4. Generic publication and host delivery | Extend delivery in `gpu_events.py` and result access in `context.py` using publication-specific byte extents and metadata | Scalars, empty arrays, mixed/changing shapes and dtypes, conditional/every-N outputs, exact names, retained events, and delayed copies work; aggregate pinned cap and synchronous fallback prevent self-deadlock |
| 5. User calibration plus azimuthal integration | Adapt Amanda's CUDA algorithms as a user callable; psana schedules it through `DataSource` and delivers the histogram | Calibration and integration match stated references; geometry/segment mapping, missing-segment counts, and concurrent scratch lifetimes are correct; no kernels or normal D2H are launched from the public loop |
| 6. Lifecycle and integration acceptance | Exercise serial/MPI, transitions, multiple BDs, input modes, failures, and early close; update runnable examples | Required psana suites and real-device acceptance pass with recorded provenance; baseline launch structure is preserved; remaining limitations and measured submission costs are documented |

## Completed Stage 1 extraction

Stage 1 established the dense preparation boundary before public callback
dispatch. The following records that checkpoint, not a requirement to retain
the calibration path after Stage 1b:

1. Extract the existing batched gather/presence work into a reusable preparation
   operation. Keep `GPUDetector.process_batch()` using it before the current
   calibration launch, preserving default results and existing storage leases.
2. Establish the supported Jungfrau raw layout from detector/Configure metadata
   without using pedestal shape. Expose canonical segment identity and reject
   unsupported layouts explicitly. Generic fields retain descriptor access.
3. Exercise raw-only preparation internally without creating legacy calibrated
   slots, prepared constants, geometry, or calibration kernels. Keep its owned
   buffers in existing byte admission and trimming.
4. Isolate legacy setup/refresh calls so Stage 2 can select the raw-only path
   consistently on serial and MPI entry points. Do not skip a collective on
   only some ranks or remove derived work needed by CPU/hybrid consumers.

The internal Stage 1 gate exercised preparation directly. End-to-end removal
is checked in Stage 1b and task setup again after Stage 2 wiring; Stage 1 alone
does not claim a usable `gpu_fn` path. Stages 2 and 3 must reject unsupported or
unfinished delivery rather than silently ignoring a task/publication. The first
usable public callback milestone is the integrated Stage 4 path.

## Decisions to settle where they are needed

- Stage 1: exact supported Jungfrau layout/segment mapping and a simple
  callback-independent subbatch default; do not derive it from calibration cost.
- Stage 1b: no calibration compatibility mode. Retain parsed-field access without
  a task; retire nonzero image-count `gpu_d2h_chunk_size` requests with an explicit
  error. Preserve numerical reference code in example/test support and migrate
  automatic-calibration acceptance to an explicit callback in Stages 2–4.
- Stage 2: constructor validation and unsupported-path errors. Source calibration
  values stay unchanged; user code owns any derived constants and their storage.
- Stage 4: choose and document a finite configurable aggregate pinned byte cap
  per BD. This bounds psana host staging only. Use byte-oriented buffers and
  publication records; do not create image-shaped pools per result name.
- Stage 5: explicit geometry, mask, gain-zero, invalid-gain, q-range, and count
  policies. Start without common mode. Use fresh registered scratch to establish
  correctness before considering user pooling.

These are implementation choices within the agreed design, not prerequisites
for another design approval. Batch callbacks, fused kernels, native-library
loading, generic CUDA IPC constant sharing, and performance tuning are follow-ups.

## Validation and progress

- Preserve focused unit tests and real-device parser/gather, ownership, and
  retirement checks. Migrate pixel-exact checks to explicit user calibration;
  Stage 1's default-path results remain the historical numerical baseline.
- Core DataSource/Run changes require both `pytest psana/psana/tests/` and
  `pytest psana/psana/tests/byhand_*` in a verified built environment.
- Add focused acceptance for varying publications, missing inputs, transitions,
  delayed kernels/copies, and failure after submission. Record launch counts as
  structural diagnostics; no throughput thresholds in pytest.
- Compare input-only and callback-path host submission time and launch/copy
  counts separately. Small output volume is not evidence of low launch overhead.
- Store generated validation artifacts on scratch and concise findings in the
  repository. Do not reuse historical performance numbers as new acceptance.

Status: Stages 1 and 1b implemented and validated on CPU and A100, including
four-rank MPI exclusive/hybrid acceptance. See
[Stage 1b findings](../user_kernel_stage1b_findings_20260926.md).
Stage 1b performance acceptance is recorded in the
[matched regression report](../performance/user_kernel_stage1_regression_20260926.md).
The historical Stage 1 calibration-path timing finding does not gate the intended
Stage 1b runtime.

Stage 2 declaration, selective constant uploads, and serial/MPI task setup are
implemented and validated. See the [Stage 2 findings](../user_kernel_stage2_findings_20260927.md)
for the review fixes, ownership contract, and CPU/A100/MPI acceptance results.
The [Stage 2 performance report](../performance/user_kernel_stage2_regression_20260927.md)
records the accepted completed A/A and A/B controls; no repeatable slowdown above
the 5% investigation threshold was observed.

Stage 3 internal producer dispatch, publication registration, and owner retention
are implemented and validated on CPU and A100, with MPI input/task setup regression
checks. See the [Stage 3 findings](../user_kernel_stage3_findings_20260927.md).
Stages 4–6 remain pending. Next: generic publication copies and host delivery;
declared tasks still reject public event processing until Stage 4 is available.
