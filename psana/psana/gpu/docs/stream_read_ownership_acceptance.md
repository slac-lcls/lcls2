# Retained-view and tight-budget GPU ownership acceptance

2026-09-25, source commit `d4cd86d10`, with a new device test module and no
production runtime changes. The broader ownership matrix passes: **18 new
A100 cases plus 20 existing regression cases**, **38 passed** in 12.13 seconds.
Job **39095559** completed on **sdfampere011**, exit `0:0`, elapsed 25 seconds.

## New coverage

| Matrix | Cases | Checks |
|---|---:|---|
| Depth 1/2 × 4/8 MiB working budget × automatic D2H off/chunk 7 | 8 | Two EB packets with 300 and 203 events, ordered delivery of all 503 timestamps, selected public field pixels, all five calibrated slow events, partial D2H tail, input retirement and idempotent close |
| Depth 1/2 × 4/8 MiB working budget × resume/close recovery | 8 | Retain an open public field context while advancing delivery; next-batch admission reports stream-credit pressure; cache trimming and failed close preserve both event input owners and pixels; context release permits processing or close to recover |
| Depth 1/2 with two independent CUDA consumers | 2 | Exit two public field contexts after enqueueing copies on separate streams; complete the fast consumer while the earlier slow consumer still runs; polling reclaims the later group without waiting; both copies retain exact original pixels |

The budget is the fixture's already-committed fixed allocations plus 4 or 8 MiB.
The admission capacity is respectively 4 or 8 MiB. Every successful allocation
reservation and admission hold is observed, checking committed plus held bytes
against the limit. Tests also require zero outstanding admission holds, no
pending I/O or input pins, and released parsed windows after successful close.
These are explicit owned-allocation checks, not CuPy memory-pool usage checks.

The retained context reads the fast stream's field at event 99, when a slow
stream dgram is also present. The current context conservatively retains both
input owners. That behavior is tested as the current safety contract; selective
source-owner leases remain an optional future change.

The delayed consumer copies the complete field after a device-side delay.
The test requires its CUDA event to remain incomplete before and after polling;
it does not infer nonblocking reclamation from host timing. A second consumer
on another stream completes first. The delayed completion is synchronized
explicitly before final pixel comparison and cleanup.

## Runtime, fixture, and regression coverage

- One A100, PCI bus `0000:C1:00.0`; eight allocated CPUs, 32 GiB host memory.
- Python 3.9.20, CuPy 13.6.0, KvikIO 24.08.02, compatibility mode ON.
- Real KvikIO reads, GPU XTC parsing, canonical gather/calibration, CUDA streams,
  and public `GpuEventState.detector(...).field(...).on_gpu_view(...)` contexts.
- The existing `xpptut15-r0014-s000-c000.xtc2` Configure/L1Accept fixture supplies
  real XTC metadata and pixels. Private generated files contain per-event pixel
  sentinels, one fast input per event, and a padded slow input every 100 events.
- Existing regression modules cover batched parsing, BeginStep/EndRun draining,
  partial parser construction failures, independently reusable raw slots,
  multi-owner calibration transitions, bulk-off byte parity, and legacy-path
  lifecycle failures/early exit with and without automatic D2H.
- Source and fixture/test hashes passed before and after all 38 tests. Current
  production Python files matched the previously validated runtime before
  freezing. `git diff --check` passed.

This completes the requested retained-view/tight-budget/delayed-consumer test
expansion. It is single-process GPU correctness evidence, not an MPI scaling,
live-data, true-GDS, or throughput measurement. It does not authorize removing
legacy coverage: each old policy fixture still needs its replacement assertions
mapped before deletion. Throughput work remains deferred.

## Reproduction and artifacts

Test module:
`psana/psana/tests/gpu/integration/test_gpu_group_ownership_device.py`

SHA-256:
`e6e342990ddb981941e6271227d8c791ff358b37cccdbe0e68913d931342d362`

Frozen campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-ownership-20260925`

The campaign contains `run.sbatch`, `device.py`, `source-commit.txt`, frozen
`tests/gpu`, `test-hashes.json`, `builds.json`, and `job-39095559.log`.
The runner verifies hashes before and after executing the new module and the
five recorded regression modules. Runtime/native dependencies remain at the
paths hashed in `builds.json`.

```sh
sbatch /sdf/scratch/users/m/monarin/gpu-validation/stream-read-ownership-20260925/run.sbatch
```
