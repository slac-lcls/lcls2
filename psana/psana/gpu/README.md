# psana2 GPU Documentation

The psana2 GPU path moves selected detector streams through a GPU-oriented
EventBuilder batch, reads bigdata into device memory, parses XTC on the GPU,
and exposes lease-aware detector results through normal `psana.Event` objects.

The documents are grouped by status. Current-design documents describe this
branch and should be kept synchronized with code. Proposal documents are for
review and are not API commitments. Performance documents record measurements
and the configurations that produced them.

## Current design

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

- [User GPU pipeline](docs/proposals/user_gpu_pipeline.md): user CUDA or CuPy
  work scheduled inside psana's BD batch pipeline.
- [User-kernel preparation handoff](docs/proposals/user_kernel_integration_handoff_20260926.md):
  master merge, design history, current integration boundary, and open decisions.
- [AMI integration](docs/proposals/ami_integration.md): possible psana2 GPU and
  AMI integration; retained for evaluation.

## Performance evidence

- [Code-size simplification handoff](docs/simplification_baseline_20260925.md):
  committed baseline, completed JF results, mixed-detector campaign and invariants.
- [Jungfrau single-node scaling](docs/performance/jungfrau_single_node_sdf.md):
  the historical 10,000-event cold/warm 1/2/4-GPU, multi-BD matrix, including
  the 1-GPU/4-BD cold and 4-GPU/8-BD warm results.

- [Current Jungfrau scaling campaign](docs/performance/jungfrau_current_scaling.md):
  current-code multi-GPU/BD rerun, validation gates and job status.
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
