# Capacity review closure

Claude passed `df757a35763458bca62053f8f51c86dce84ae293` with no blocking
findings. All three cases passed full CPU pixel/count acceptance. The tested
executable files are unchanged; this closure contains documentation/evidence.

| Finding | Response |
|---|---|
| CAP-001 | Accepted, optional helper generalization deferred. This campaign's interrupted job ID is correct and explicit. Closure independently checked status and hashes of BOTH jobs' controlled-tests.json and records them in closure-checks.json. A reusable recovery summarizer should load both test reports and derive the interrupted job ID instead of the fixed campaign literal. |
| CAP-002 | Accepted and addressed in documentation. Say the 200-bin case did not complete; the interrupted report has no phase checkpoint and is never acceptance evidence. Terminal observation of a retained first-bin file suggested verification had begun, but the report itself does not prove the phase. Per-phase journaling remains optional hardening for a future run. |
| CAP-003 | Accepted and addressed. Warm expansion interval sum is described as about 165 ms rather than a tightly rounded range. |
| CAP-004 | Accepted and addressed. gpu_phase_memory covers baseline through finished export, excluding close() and CPU validation. README labels the host column through-export and explicitly defines this window. |
| CAP-005 | Accepted and addressed by documenting the supported reset contract and limitation of negative-delta detection. No evidence of an intra-call reset in the frozen reader; all case counters reconcile. Epoch tracking is a future requirement if reset semantics change. |
| CAP-006 | Accepted, optional test deferred. The derived task inherits _identify unchanged; existing Stage3A identity tests exercise duplicates and this campaign validates native identities. Add a direct duplicate-add/no-mutation test before broadening task-input support. Not required by the reviewer for this fixed campaign. |

The prior F-008 third-stream gap is covered in this derived task's device tests:
RuntimeError is required and both banks/seen must remain unchanged. Old Stage3A
source/review history was preserved.

Reviewer inspected code and targeted sections of the large reports, not every
per-bin/sample row, and did not execute tests/hashes. Coordinator checked all
reported source hashes against Git blobs, byte-identical report copies, every
retained file hash, all native counts and every final pixel's numerical bound.
The interrupted original 200 case is excluded; recovery ran on a different node,
so these timings do not establish same-node scaling or sustained throughput.
