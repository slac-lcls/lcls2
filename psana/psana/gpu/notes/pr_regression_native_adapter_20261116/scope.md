# Native regression adapter implementation

Base: 5a76c182f7d56a86d4dba43f5baf749400366bcf, results/psana2-gpu-scaling.

Implement a source-only Perlmutter adapter for the existing six run-51 raw
CPU/GPU cases and the current features/psana2-gpu API (199575205). A candidate
revision must retain that API and independently qualify its pinned installation.
The adapter launches separate synchronous warmup/measured MPI steps within an
existing one-node allocation. It verifies runtime, role/affinity/GPU identity,
independent event/byte references, private-input cache state, and records native
I/O, NIC, lifecycle/resource and diagnostic call-count evidence. Preserve the
controller/profile defaults and all historical harnesses. No SDF launcher,
calibration/output/full-node/multistream extension, framework change, data copy,
job submission, or performance claim in this implementation step.

Acceptance: offline failure-path/aggregation/cache tests; existing 20 controller
tests and archive checker; independent Claude review on the exact code commit,
maximum three rounds. Do not duplicate framework unit/device assertions.

Qualification remains a separate step: native API smoke, inputs/reference and
private-copy preparation, two isolated installs and provenance, compute-access
and retention/cross-site receipts, instrumentation-overhead measurement, qualified
component bounds, scheduler cancellation test, then correctness and A/A. No
qualification flags are enabled by implementation or offline tests. Missing
native metrics must be null/unavailable, not invented zeroes; component review
cannot be replaced with the loop-time verdict.
