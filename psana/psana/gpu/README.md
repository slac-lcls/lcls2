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
- [AMI integration](docs/proposals/ami_integration.md): possible psana2 GPU and
  AMI integration; retained for evaluation.

## Performance evidence

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
