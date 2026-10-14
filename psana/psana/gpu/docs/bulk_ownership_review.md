# Ownership changes: pre-commit review

2026-09-23, branch `codex/psana2-gpu-allocation-ownership`, bulk base
`8f94e3c7bc4643309d022cb5c1dd7e85805d3ac4`.

Reviewed the cumulative Stage 1–4 changes: allocation owners, admission tokens,
reader/parser/output/fixed storage, public access guards, consumer completion,
iterator cleanup, failure handling, diagnostics, tests, and documentation.
The B++ merge and allocation-reuse optimization remain separate work.

## Findings fixed

### P1: failed setup uploads could be freed by garbage collection

The budget held quarantined arrays, whose allocation charges referenced the
budget. That cycle became collectible if setup raised and the caller lost the
budget. Collecting it could return backing to CuPy despite unproven completion.

A bounded real-CuPy probe reproduced this: after dropping the external budget
reference, 512 bytes remained live; after `gc.collect()`, the budget and its
512-byte backing disappeared. The probe constructed the failed-work ownership
graph with an unproven-completion token; it did not inject a hardware fault.
Evidence: job **38897548**, `device-38897548.log` under the review artifacts.

Fix: `_GpuBudget.quarantine_upload()` stores the work under the budget lock and
adds a process-level strong reference to the budget. Successful explicit
draining removes that reference. Repeated drain failure preserves it. No CUDA
calls were added to destructors. A real-device regression test drops the budget
owner, forces GC, verifies the live charge/backing, retries failure, and checks
that successful draining releases both the storage and the safety reference.

### P2: parsed-field context exit could not retry failed completion

An input child lease closes before asking its owner to drain. If that drain
failed, retrying context exit recorded a second event and attempted to register
it on the closed lease. This prevented recovery through the context API.

Fix: the context remembers when completion registration is finished and retries
only the outstanding owner drain. Both normal event recording and the
same-stream fallback path have regression tests. The two tests fail against
the Stage 4 install (`field-retry-before.log`).

### P2: field CPU-copy event-record failure leaked a child reference

`GpuFieldResult.on_cpu` had its own cleanup block. Failure while recording its
terminal event bypassed child-reference release, leaving the input window
occupied even when the copy stream could be drained.

Fix: CPU field copies use the shared field-view context on the current stream.
Its recording-failure fallback drains that stream and releases the child use.
The regression verifies reference counts, eventual window release, and the
readable cached CPU copy. It fails against the Stage 4 install
(`field-host-before.log`).

## Cleanup

- Restored `__slots__` on result leases and result-view contexts, avoiding an
  instance dictionary on these frequently retained objects.
- Removed unused dataclass imports after the facade conversions.
- Updated the allocation helper documentation to cover all pipeline storage.
- Replaced the outdated manual reserve/release example with allocation-owned
  usage, and documented failed-upload retention and retry behavior.
- Updated the known-issues link to the completed Stage 4 acceptance.
- Updated two CPU field tests to supply a fake current CUDA stream, preserving
  CPU-only execution while exercising the unified copy/context path.

No changes were made to the cache-trimming/residency policy or to core MPI/event
iteration during this review. The measured 99,507 versus 206 allocation calls
remain a later performance target. Reusing those caches requires its own
admission/lifetime review and measurements.

## Validation

| Gate | Result |
|---|---|
| CPU unit suite | **317 passed**, including three new field-cleanup cases |
| A100 device suite | **21 passed**, including quarantine/GC regression |
| JF review acceptance | **4 cases passed; 26,598 reconciled checkpoints** |
| Main psana suite | **384 passed, 28 skipped, 10 deselected** |
| MPI byhand group | **4 passed** |
| Python compilation and diff whitespace | Passed |

GPU/JF job **38897698** completed in **4m 12s** on `sdfampere038`.
Core/MPI job **38897699** completed in **4m 01s** on `sdfmilan266`.
Both jobs exited `0:0`. Test groups overlap and their counts must not be added.

Each JF case delivered 1,003 ordered events, checked 15 raw/calibrated image
pairs against the Stage 4 independent CPU reference, and ended with zero live
device allocations while the selected facades remained saved. The two cases
without automatic D2H additionally checked direct parsed-field CPU images
against the canonical raw images at all 15 sample points.

| JF case | Batch / depth | Budget GiB | D2H chunk | Peak live MiB |
|---|---|---:|---:|---:|
| Bulk off, all facades saved | 20 / 1 | 8 | 0 | 3,201.874 |
| Bulk on, all facades saved | 20 / 1 | 4 | 0 | 3,201.874 |
| Bulk off, partial tail | 13 / 2 | 8 | 7 | 3,970.476 |
| Bulk on, partial tail | 13 / 2 | 8 | 7 | 2,305.267 |

These peaks match Stage 4. The 10,000-event campaign was not repeated: the
review fixes target failure retention, field-access cleanup, and object layout;
the short matrix covers the affected paths and the previously accepted long
run remains documented separately. No throughput claim is made.

Artifacts: `validation/ownership-review-20260923/`. Tests use an isolated install
with hashes recorded in `build.json`; all installed runtime sources match the
worktree. Historical Stage 1–4 reports retain the exact results and scope of
their original builds. The review acceptance checks direct parsed-field CPU
images in addition to canonical raw/calibrated outputs.

## Commit scope

Changes remain uncommitted. `commit-files.txt` in the review artifact directory
lists the proposed source, test, and documentation files. Generated installs,
CuPy caches, raw traces, Slurm logs, and unrelated pre-existing untracked files
are excluded from that list and remain on disk. The cumulative `commit.patch`
includes the new allocation helper and regression tests, which plain
`git diff` alone would omit while they are untracked.

**Review disposition:** the three reproduced findings are fixed and the
validation gates pass. No further commit blocker was identified in this scope.
Multi-BD/IPC scaling, true GDS, and allocation-reuse optimization retain their
separate acceptance requirements.
