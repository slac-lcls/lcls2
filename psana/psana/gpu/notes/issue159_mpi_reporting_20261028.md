# Issue159 PR246: combined MPI field lifetime and memory reporting

October 28, 2026. **The focused MPI/reporting matrix passed** on the integrated
Stage4 runtime. Job **42250696**, `sdfampere001`, completed all 16 field/reporting
cases, two independent real Jungfrau calibration pixel comparisons, and the
six-rank import-origin preflight. This is correctness evidence, not a performance
measurement or full Stage5/final acceptance. Independent review and matched
cold/field performance remain pending. No production code changed.

## Scope and tested identities

- Integration runtime: `31c4c2f214de14628820cfdfdf309b74367620e6`.
- Source branch at preparation: `111922650` plus the new
  [`mpi_issue159_fields.py`](../../tests/gpu/integration/mpi_issue159_fields.py)
  acceptance driver, frozen by hash before submission. The frozen driver and
  committed driver must have identical hashes.
- Existing independent install:
  `/sdf/scratch/users/m/monarin/gpu-validation/issue159-target-merge-review-20261028-r1/trial/install`.
- Artifact root:
  `/sdf/scratch/users/m/monarin/gpu-validation/issue159/pr246-mpi-20261028-r7`.
- Six processes: one Smd0, one EB, four BDs on the same physical A100. The driver
  asserts observed roles, PCI identity, peer count, one owner, unique event IDs,
  and that every BD exercised the field path. Slurm supplied the GPU mask.
- KvikIO compatibility mode ON, eight threads, 1 MiB tasks; MPI OB1. Native and
  Python imports resolve to the frozen install, and input hashes pass before
  and after the campaign. No native install was moved or modified.

## What the combined test proves

The driver uses real psana MPI event delivery, GPU discovery, actual IPC
owner/importer allocations, manager budgets, and the delivered event's actual
input lease. Synthetic packed fields force unalignment independently of the
dataset. They contain a 257-element uint64 array, a scalar, and an absent segment.
The fixture's source/output buffers are outside the manager budget; the lazy
aligned copies use the real event lease and manager budget.

Two consumer streams read those fields AND the actual device constants. The
first materialization is deliberately delayed; the cached access on the second
stream must observe its copy-ready dependency. The test requires both consumers
to be pending, checks exact output values after close, and intercepts constant
close only to assert completion and charge retirement before calling the real
collective teardown. It does not manually join consumers before advancing or
closing the real event iterator. The existing open-child device tests remain
separate coverage; this MPI case exits view contexts with GPU work still pending.

Every BD made exactly **two byte copies** (array and scalar), totaling **3,072
rounded backing bytes**. The second access reused both pointers and readiness
events, made no extra copy, and added no budget charge. The absent segment did
not allocate. Snapshotting did not change limits, committed bytes, or holds.
The field charge appeared in `retained` and the `field-copy` ledger, never in
imported `borrowed` constants. All consumers completed and all field-copy
charges/weak references disappeared before constant teardown.

| Constant mode | Owned constants per rank | Imported borrowed bytes | Field-copy charge |
| --- | --- | --- | --- |
| Shared owner | 136 bytes | 0 | 3,072 bytes |
| Shared importer (three ranks) | 0 | 136 bytes each | 3,072 bytes each |
| Sharing disabled | 1,024 pooled backing bytes each | 0 | 3,072 bytes each |

The synthetic constants are a uint64 scalar, a 16-element uint64 vector, and an
empty `(0, 3)` array. Actual shared bytes are 8 + 128; private nonempty arrays
occupy two 512-byte pool blocks. Owned/imported counts and constant ledger
entries all clear after real constant close. Automatic limits honor owner versus
importer sizing; explicit limits remain 3 GiB per BD. The test checks those
values rather than inferring budget correctness from pixels alone.

## Case results

Each case is a fresh MPI process group. A zero return code and an explicit
rank-validated success record are both required. Each process group has a
600-second deadline; timeout kills the process group and is never accepted.
Normal cases deliver 40 unique events; early cases stop after one event per BD.

| Case | Events delivered | BDs exercising fields | Result |
| --- | ---: | ---: | --- |
| fields-share0-budget0-bulk0-early | 4 | 4 | passed |
| fields-share0-budget0-bulk0-normal | 40 | 4 | passed |
| fields-share0-budget0-bulk1-early | 4 | 4 | passed |
| fields-share0-budget0-bulk1-normal | 40 | 4 | passed |
| fields-share0-budget3-bulk0-early | 4 | 4 | passed |
| fields-share0-budget3-bulk0-normal | 40 | 4 | passed |
| fields-share0-budget3-bulk1-early | 4 | 4 | passed |
| fields-share0-budget3-bulk1-normal | 40 | 4 | passed |
| fields-share1-budget0-bulk0-early | 4 | 4 | passed |
| fields-share1-budget0-bulk0-normal | 40 | 4 | passed |
| fields-share1-budget0-bulk1-early | 4 | 4 | passed |
| fields-share1-budget0-bulk1-normal | 40 | 4 | passed |
| fields-share1-budget3-bulk0-early | 4 | 4 | passed |
| fields-share1-budget3-bulk0-normal | 40 | 4 | passed |
| fields-share1-budget3-bulk1-early | 4 | 4 | passed |
| fields-share1-budget3-bulk1-normal | 40 | 4 | passed |

The unchanged `mpi_calib_pixel_exact.py` then passed sharing ON and OFF with
**20 unique events each**, one EB/four BDs, and an explicit 3 GiB budget. These
compare real GPU Jungfrau calibration pixels against the CPU result from the
same event. They use the unmodified calibration setup/transport path; the
synthetic constant injection described below applies only to the field driver.

## Separate host scalar-shape finding

The initial field driver exposed an existing host-calibration limitation:
`CalibXtcConverter._build_datadef` in `psexp/calib_xtc.py:98` applies
`np.ascontiguousarray()` to a zero-dimensional NumPy array, changing `()` to
`(1,)` before the GPU manager receives it. The shared GPU callback consequently
received `(1,)`, as recorded by failed job 42250579. This is distinct from
PR187's GPU IPC scalar-shape preservation and issue159's scalar event fields.

The serializer has identical blob `aeb20556f3a287fb0d32853ee9f3bf15e0cf7c51`
at both original baseline `90c05965c` and updated target `8cf05e86a`.
No serializer fix is included here. The final driver records the host-transport
shapes, then reinstates the intended synthetic constant arrays immediately
before constructing the GPU manager. Thus the successful scalar IPC check is
explicitly **from the GPU-manager input boundary**, not a claim that the entire
host calibration transport preserves scalar shape. Review this separate finding
for follow-up; do not silently count the failed whole-path scalar check as passed.

## Preserved development attempts

| Attempt / job | Outcome |
| --- | --- |
| r1 / 42250082 | Launcher preflight failed: clean PATH omitted srun and Slurm exposed one task slot. No MPI test ran. |
| r2 / 42250132 | Six-rank imports passed; a teardown coverage assertion obscured an earlier failure. |
| r3 / 42250242 | Added field-entry traceback; failure occurred earlier, in the callback. |
| r4 / 42250496 | Moved coverage assertion out of teardown so it no longer masks the initiating error. |
| r5 / 42250579 | Full exception diagnostics identified the inherited host scalar promotion to `(1,)`. |
| r6 / 42250647 | New host-shape diagnostic assumed `(array, metadata)` entries; transport returned bare arrays. Corrected the diagnostic. |
| r7 / 42250696 | All 16 field/reporting cases, two real pixel checks, and provenance checks passed. |

Attempts are retained separately under
`/sdf/scratch/users/m/monarin/gpu-validation/issue159/pr246-mpi-20261028-rN`.
Changes between attempts affect launch configuration, fixture scope, and test
diagnostics only. No production change, weakened lifetime assertion, skipped
case, or sanitizer suppression was used. This MPI job was not run under
sanitizer; the earlier selected 315-test device memcheck is separate evidence.

## Reproduction and remaining work

The frozen `run.sbatch`, `run.py`, `metadata/runtime-env.json`, and input hashes
describe the exact allocation and 16-case Cartesian product. A representative
case, in that verified installed environment, is:

```bash
PSANA_GPU_SHARE_CONSTANTS=1 mpirun --bind-to none -np 6 python -B \
  tests/gpu/integration/mpi_issue159_fields.py --exit early --bulk 1 --budget-gb 3
```

The allocation requests one A100, six tasks/four CPUs each, 128 GiB host memory,
and a one-hour time limit. The controller preserves job logs, per-case rank
snapshots, return codes, timings and final status. Compact durable evidence is
in [the JSON record](evidence/issue159_mpi_reporting_20261028.json).

Next: independent review of the implementation/integration and this test driver,
then the already planned matched cold A/B and field-view timings. No Stage5
implementation or PR merge occurred during this validation.
