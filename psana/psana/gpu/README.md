# psana2 GPU Documentation

The psana2 GPU path moves selected detector streams through a GPU-oriented
EventBuilder batch, reads bigdata into device memory, parses XTC on the GPU,
and exposes lease-aware parsed detector fields through normal `psana.Event` objects.

The documents are grouped by status. Current-design documents describe this
branch and should be kept synchronized with code. Proposal documents are for
review and are not API commitments. Performance documents record measurements
and the configurations that produced them.

## Current design

Stages 1–4 are implemented: `GpuTask` runs once per nonempty selected execution
subbatch, and named publications receive batched host delivery. Start with the
[task and results guide](docs/user_task_results.md) or the
[input-only example](examples/input_only.py). Set `batch_size` explicitly;
the task default is one. Memory admission can split an EventBuilder batch.

The runtime provides input preparation, requested original calibration constants,
completion tracking, and bounded output staging. User code owns calibration,
geometry algorithms, kernels, scratch, and device outputs. With no task there
is no automatic calibration or output D2H. CPU/hybrid calibration remains available.
The default output pinned-memory cap is 64 MiB per BD; nonzero
`gpu_d2h_chunk_size` remains retired.

[Stage 4 findings](docs/user_kernel_stage4_findings_20260928.md) record correctness
and performance. Subsequent fixes cover
[noncollective geometry cache misses](docs/geometry_cache_fallback_20260928.md)
and [serial iterator cleanup](docs/serial_gpu_close_20260928.md).
Use `with closing(run.events())` for deterministic cleanup on early exit.
The [external batched calibration example](docs/user_kernel_stage5a_20260928.md)
is implemented and validated in Stage 5a on CPU/A100 and serial/MPI, including
byte-for-byte default CPU-v3 calibration matches with matching constant selectors. Azimuthal integration
and combined performance (Stages 5b/5c), plus final consolidated acceptance
(Stage 6), remain pending; see the
[stage tracker](docs/proposals/user_kernel_implementation_stages_20260926.md).


- [Architecture overview](docs/architecture_overview.md): components,
  boundaries, routing modes, and supported scope.
- [Event flow and lifetimes](docs/event_flow_and_lifetimes.md): CPU/GPU MPI call
  paths, Run/Event ownership, transitions, and result delivery.
- [GPU XTC parser](docs/gpu_xtc_parser.md): Configure tables, device parsing,
  field locators, detector bindings, and general field access.
- [Memory backpressure and results](docs/memory_backpressure_and_results.md):
  execution slots, byte budgets, asynchronous D2H, leases, and result access.
- [Known problems and limitations](docs/known_issues.md): verified implementation
  gaps, their impact, and the intended direction for follow-up work.

## Proposals

- [User GPU kernel support](docs/proposals/user_gpu_pipeline.md): `GpuTask`
  input/constant declarations, internal BD submission, user-owned buffers,
  named output publication, and asynchronous host delivery.
- [User-kernel preparation handoff](docs/proposals/user_kernel_integration_handoff_20260926.md):
  master merge validation and provenance for the canonical proposal.
- [AMI integration](docs/proposals/ami_integration.md): possible psana2 GPU and
  AMI integration; retained for evaluation.

## Performance evidence

- [User-kernel scaling and batch scheduling](docs/performance/user_kernel_scaling_20260928.md):
  full JF / partial JF+feespec reruns and measured per-event versus batched kernels.

- [Code-size simplification handoff](docs/simplification_baseline_20260925.md):
  committed baseline, completed JF results, mixed-detector campaign and invariants.
- [Jungfrau single-node scaling](docs/performance/jungfrau_single_node_sdf.md):
  the historical 10,000-event cold/warm 1/2/4-GPU, multi-BD matrix, including
  the 1-GPU/4-BD cold and 4-GPU/8-BD warm results.

- [Current Jungfrau scaling campaign](docs/performance/jungfrau_current_scaling.md):
  pre-user-kernel multi-GPU/BD results and completed mixed-detector comparison.
- [User-kernel Stage 1/1b regression check](docs/performance/user_kernel_stage1_regression_20260926.md):
  matched JF-only comparisons on one GPU with 1–4 BDs.
- [JF + feespec one-GPU scaling](docs/performance/jf_feespec_single_gpu_scaling.md):
  batch-20 cold/warm, bulk off/on comparison with 1, 2 and 4 BDs.
- [GPU pipeline baseline](docs/performance/gpu_pipeline_baseline.md): initial
  CPU/GPU throughput comparison and bottleneck observations.
- [D2H bandwidth](docs/performance/d2h_bandwidth.md): measured D2H sampling and
  NIC-bandwidth behavior.

## Status convention

Each document must identify itself as one of:

- **Current:** describes behavior implemented on this branch.
- **Proposed:** describes an interface or architecture still under review.
- **Measured:** records an observation tied to a dataset, software revision,
  topology, and runtime configuration.

Superseded designs are removed from the working tree rather than kept in a
second archive. Git history remains the archive.
