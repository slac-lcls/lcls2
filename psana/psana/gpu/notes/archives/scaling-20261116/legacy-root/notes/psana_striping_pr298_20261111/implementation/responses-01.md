# Response to implementation review 1

IMPL-R1-01: Fixed Python buffered payload reads: buffering=0, explicit pread(24), POSIX_FADV_RANDOM. Metadata validation stays on the allocated compute node, outside measured cases, rather than doing intensive I/O on a login node or adding another allocation. This is a disputed placement recommendation, not a closed finding. The small native diagnostic will measure validation and cold-gate cost, with the existing 1200-second validation timeout and 45-minute total cap; no screen launches. Before measured evidence, budget is 600 seconds GPU tests + 1200 seconds metadata validation + 900 seconds remaining pilot. It can fail within that cap; no successful-fit claim yet.

IMPL-R1-02: Empty BDs allowed under the exact global partition rule. For small pilots only, a BD with fewer callbacks than requested streams reports its coverage without falsely failing; at least one BD must exercise the requested depth. Full screen retains the strict nonempty-BD rule. No overlap claim.

IMPL-R1-03: Every BD compiles all four parser kernels before the cold gate, without payload reads; warm-up receipts recorded.

IMPL-R1-04: Added rank wall start/completion, gate duration, per-case llite/bdi diagnostics and cross-case affinity audit. Continuous device peak telemetry deferred to later screen preparation: native snapshots and framework admission/peak outstanding bytes retained; no measured memory peak claim.

IMPL-R1-05: Audit takes expected reviewed commit independently, compares exact 31 native node IDs and source hashes with committed expected-native.json, checks both witness/ranges hashes. All native evidence still pending.

IMPL-R1-06: Deferred screen/sizing walltime launch until measured pilot and separately authorized 100k sizing case. Prepared 72-case parameter order remains unchanged. The pilot launcher is not a ready-to-submit full screen launcher.

IMPL-R1-07: Each cached lookup now verifies the returned handle object and fd equal the first fstat capture; reopens fail closed. Avoid per-request fstat overhead; first-open fd/inode checks remain.

No finding is declared reviewer-closed. Native testing is implementation validation; round 2 will review fixes and evidence together. Missing external PR internals will be copied into review evidence to remove the restricted-read limitation.
