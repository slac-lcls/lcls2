# Response to implementation review round 1

Reviewed manifest: `4e3f8732e035e2e0e149facda8f5764f855671f55dc3df6dde3058029f001bde`.
Verdict: changes requested. No job submitted. Corrections require re-review.

| Finding | Disposition | Correction / evidence |
| --- | --- | --- |
| IMPL-R1-001 (blocking) | Fixed, awaiting re-review | StageTask.prepare uses admission_from_budget, which calls committed() and allocation_snapshot(). Unit test requires both calls. Host native check exercises the pinned _GpuBudget with reserve_allocation/commit/release for all eight stage/topology combinations, and the actual StageTask.prepare path with a host CUDA mock. |
| IMPL-R1-002 | Fixed, awaiting re-review | Native-derived seed offsets and per-record XTC-header/Shapes comparisons bound sampled reads without trusting equal record sizes. Timestamp, service, file identity and physical panel membership remain checked. Sixteen reference tests cover parity and malformed/truncated/changed layout. Three actual native events match full expansion bitwise across all panels; warm requested reads are 2,932 bytes/event after one full witness per stream, versus 3,358,200 bytes/event. This is requested-byte evidence, not physical-I/O or walltime proof. |
| IMPL-R1-003 | Fixed, awaiting re-review | Every BD probes a one-element CuPy ndarray weakref and verifies release at setup, recording version and result. This remains unexecuted on CUDA until allocation. |
| IMPL-R1-004 | Fixed, awaiting re-review | S now performs both entry/exit memory queries, matching E/EC/ECC. Processing schema explicitly labels the two queries per callback and distinguishes untraced timing from zero monitoring overhead. All safety checks remain. |
| IMPL-R1-005 | Implemented, awaiting re-review | Checks all four optional shared-memory/cache attributes; per-rank output records their absence and both environment flags. |

Round 1 explicitly accepted the inherited source/native compatibility strategy
for preparation only, with its meson.build mismatch and mandatory allocation
acceptance retained. It did not claim a fresh rebuild or measured GPU fit.

Reference evidence: `native-reference-parity.json`. Original detailed logs and
timings are under shared scratch `reference-r1-002-rckuHw`. This optimization
does not establish the reference time ceilings: many small preads remain and
exhaustive acceptance/pilot reference costs are unchanged. The conservative,
unmeasured operational budget and fail-closed timeout gates remain explicit.
The first expanded-suite invocation in the plain venv failed on missing native
psana imports. The final suite activates the pinned source/native overlay;
that environmental failure is retained, not counted as a passing test run.
