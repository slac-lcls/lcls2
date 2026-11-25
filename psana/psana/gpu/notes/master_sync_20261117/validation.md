# Master synchronization: validation record

**November 25 status:** see the [current overview](current_status.md) for the
combined PR326/EndRun fix, passing selected verification and GitHub CI. The
callback MPI deferral below is historical and has been superseded by a pass on
the combined code. Other failures/skips and performance limits remain recorded.

Companion to [merge_decisions.md](merge_decisions.md) for
[PR #331](https://github.com/slac-lcls/lcls2/pull/331).
Consolidated November 23, 2026 from the saved SDF records and the Perlmutter
snapshot taken November 21 at 01:44:53 UTC.

**November 24 update:** the [completed Perlmutter report](performance_20261124.md)
records all five raw A/B cases, reproducible sample data, the agreed narrower
performance scope, CPU-crash disposition and #299 correctness follow-up. Three
loop comparisons are equivalent and two inconclusive; all five execution reviews
have passed for the scoped evidence claims. This is not full performance acceptance or a merge-ready claim.

The sections below preserve the November 23 consolidation and November 21
Perlmutter snapshot. Their pending-campaign statements are historical; use the
November 24 report for the latest campaign outcome.

## Tested revisions and environments

| Role | Identity |
| --- | --- |
| Tested integration implementation | `5bd5c7ba300cf0a0894b5b904d3975d5ba0b3cad` |
| Pre-merge GPU baseline | `546764e0338225d026faa29627297a4a390d3396` |
| Master parent | `785f7ef5dd2428bc0f1fb87c0ff1ebbab4d37935` |
| Additional master comparison | `9635c4fe1722f52e67aadaab864a39b02359c6e7` |
| Reviewed scaling-template anchor | `9f809fb08468c232e208dcfeb4d8a1a745f5fc96` on `results/psana2-gpu-scaling` |
| SDF correctness environment | Python 3.9 / `ps_20241122`, release build, Open MPI 5, real A100 hardware, KvikIO compatibility mode |
| Perlmutter pair | Separate debugoptimized installs; Python 3.11.14, CuPy 14.0.1, KvikIO 26.10.00, CUDA runtime 12090, driver API 13000 |

The documentation commit containing this report is not a newly tested runtime.
Both Perlmutter revisions use the same dependency environment. Source/install
integrity and imports were verified; SDF release and Perlmutter debugoptimized
timings are not compared to each other. Compatibility mode is not true GDS.

## Correctness and scoped review evidence

| Check | Recorded result | Evidence / interpretation |
| --- | --- | --- |
| Clean build/import provenance/CUDA smoke | Passed | Initial SDF job 45574973; 18 focused checks passed, including six adapter cases. |
| Initial default suite | 2,134 passed, 5 skipped, 1 timeout; 8 slow deselected | The repeated-run MPI callback case timed out. This is not a green full suite. |
| Conflict 3 review | Scoped pass, two rounds | 63 selected single-run callback tests, 11 focused interface/transport tests, two mocked communication checks, completed byhand evidence. These overlap; do not sum as unique cases. |
| Conflicts 1/2/4/5 review | Scoped pass, one round | Job 45702920: setup/CUDA smoke plus 94 focused tests, zero skips. |
| GPU directory including slow cases | 1,960 passed, including 561 GPU-marked cases; no GPU failures/skips | Verbose-log outcomes from job 45956052 before a CPU shared-memory test hung; not a completed full-suite JUnit. Includes real-data raw/hybrid pixels, slot reuse/tails and bulk/non-bulk routing. |
| Non-GPU remainder | 180 passed, 5 skipped, 1 failed, 2 deselected | Job 45958415. Isolated the already hung shmem case and the explicitly deferred callback case. |
| Reconciled full collection | 2,140 passed, 5 skipped, 2 failed/hung, 1 deferred = 2,148 | Across disjoint partitions/attempts, not a single successful pytest run. Six of eight slow tests passed. |
| Byhand MPI tests | 5/5 passed | Corrected eight-task launcher, job 45960569; earlier task-slot/binding failures retained. Prior successful run also in job 45702004. |
| MPI CPU/GPU calibrated pixels | Pixel-exact for 53 events per mode | Job 45955820: automatic and explicit 2 GiB/BD budgets in fresh processes, two BDs sharing one A100; no duplicate/lost events or pixel mismatches. Constant sharing off, so not IPC-constant coverage. |
| Two-GPU transport smoke | 100 unique events, 50 per BD | Job 45958415, distinct devices; separate transport fixture, not pixel or performance evidence. |
| Baseline attribution of two slow failures | Both reproduced on master parent and then-current master | Jobs 45982854/45982855, isolated release builds; detailed signatures below. |

The full-suite harness explicitly cleared the default `not slow` selection and
collected 2,148 tests. It deselected only the explicitly deferred callback case
initially; a later remainder run additionally isolated the known hung shmem case.
No production assertion was weakened and no skip/xfail was added to hide failures.

### Failures, skips and explicit deferral

- `test_callback_stream.py::test_fresh_callback_per_run[mpi]`: explicitly
  deferred by Mona on November 19. Unresolved and unattributed; neither passed
  nor proved inherited. Other lifecycle tests were not waived by that decision.
- `test_live_transfer.py::test_live_mode_detects_completed_transfers`: fixture
  replaces logbook lookup with `{}`, while the live path requires a stable
  nonempty file list. Both master revisions reproduced the retry/failure. This
  is not evidence of a logbook-service outage.
- `test_calib_prefetch_cache_shmem.py::TestCalibPrefetchCacheShmem::test_detector_cache_shmem`:
  three clients fail with `(32,512,1024)` versus `(6,512,1024)` calibration-array
  broadcasting errors; the parent waits indefinitely on its server. Both
  master revisions reproduced it with a bounded 360-second timeout. Killing
  the server is not a passing test. These two CPU repairs are tracked separately
  in [#351](https://github.com/slac-lcls/lcls2/issues/351); inheritance does not
  turn either result into a pass or automatically waive it.
- Five skips: absent calibration pickle (`test_calib_xtc`), absent roentdek
  library (`test_psalg`), missing calibration constants (`test_timetool`), and
  two explicitly skipped `test_xtc` callback/timestamp-filter cases.

## Perlmutter evidence available at the transfer snapshot

| Stage / job | Recorded outcome | Acceptance limit |
| --- | --- | --- |
| Runtime capture, 61621117 | Both captures and pair check passed; scheduler COMPLETED 0:0; 133 controller offline tests passed during preparation | Compatible pair, not native-workload acceptance. |
| Reuse audit, 61631312 | COMPLETED 0:0; source/install/dependency/native-library integrity and all selected input-range hashes passed; review passed | Reuse evidence; not cold-cache or timing evidence. |
| 1k native smoke, 61659990 | 12/12 cases passed, both pins across six cases; COMPLETED 0:0; independent execution review passed | 1,000 measured events after 200-event warmups, exact identity/bytes and placement; not statistical A/B or full-EndRun coverage. |
| 10k full-prefix, 61664705 | Baseline 6/6 passed; candidate four event-iteration cases passed; candidate CPU steps/1BD segfaulted; steps/2BD not run | Qualification failed. Ten accepted cases independently audited as partial evidence; no waiver of failed/missing cases. |
| CPU-step diagnostic, 61679475 | Scheduler/driver check: all six observer-mode cells and all 12 warmup/measured steps completed 0:0, elapsed 57m51s | Snapshot only; final audit/review not received here. Successful retry does not establish cause or absence of an intermittent fault. |

Original fault: rank 2, the CPU BigData worker, in measured step 61664705.35.
Adapter return 143 records termination after the fault. Periodic Python stack
samples are not the faulting native backtrace. The diagnostic compares both
revisions under full, count-only, and boundary-only instrumentation; the last
mode does not observe event count. Do not pool diagnostic timings into A/A/A/B.

### Raw workload and controller identity

- Six cases: CPU events/steps with one and two BDs; GPU events with one and two
  BDs sharing one selected A100, one EB.
- Jungfrau `mfx100848724`, run 51, first 10,000 events, streams 3/5/7/8/9;
  separate 200-event warmup per sample; 335,571,760,000 selected datagram bytes.
  Event digest: `2034208363befd6a797f889cbbff997a4c28d7297cf136de15e6b775a0597f17`.
- Batch 20; one stream/BD; 8 GiB/BD; bulk target 1 MiB; KvikIO four workers,
  1 MiB tasks, 16 MiB bounce buffer; compatibility ON/GDS off; client-cold.
- GPU work prepares/gathers dense raw input and records callback metadata.
  It does not measure calibration, a user kernel, payload D2H or output.
- Full-prefix task-local controller preserves validated scratch CuPy/CUDA cache
  paths. The diagnostic additionally enables fatal capture. These overlays
  have separate manifests; do not identify them with unchanged template hashes.
- The exported controller metadata initially names `fd33a36f7445001d7f17e13f010ed26988e529bd`;
  later reuse checks matched its Python sources to reviewed anchor `9f809fb08`.
  The final campaign report must identify its actual frozen controller/overlay.

All ten accepted full-prefix cases had exact reference identity/bytes, double
zero-residency checks, and HNI RX coverage at least 0.9 times selected bytes.
Whole-node HNI coverage is not storage-only throughput or physical-media proof.
The original qualification's failed scheduler state remains part of the record.

## Historical performance handoff requirements (November 23)

No statistical PR331 A/A or A/B results had been recorded at the transfer
snapshot. The Perlmutter task is now continuing the campaign; this historical
statement must not be interpreted as a live assertion that no later run exists.
When its final report arrives, incorporate:

1. Exact tested runtime revisions and frozen controller/profile/observer hashes;
   explain any runtime changes after the original implementation.
2. Original CPU-step fault disposition, completed missing qualification, and
   audit/review outcomes, preserving unsuccessful attempts.
3. A/A noise results before paired A/B; complete per-case samples, counts,
   95% intervals, cache/NIC/placement checks and final scheduler accounting.
4. Component checks as pass/breach/unavailable; unresolved required observations
   or bounds; instrumented-performance limitations.
5. Final conclusion and explicit remaining acceptance decisions, with stable
   artifact paths, essential hashes and reproducible commands/configuration.

The reviewed raw screen uses six balanced pairs/case and ±5% equivalence:
an interval wholly inside [-5%, +5%] is equivalent; wholly above +5% is a
regression; wholly below -5% is an improvement; otherwise inconclusive.
Positive change means slower. A/A inconclusive is not a pass, and a zero exit
status from A/B/summarize does not imply equivalence.

The completed template observer study was inconclusive at ±5%. Equal observer
code does not prove equal overhead; do not subtract uncertain point estimates
or claim uninstrumented production throughput. Saved contracts also retain
retention/cross-site, native-adapter and required component-measurement gaps.
Reconcile these against the final campaign evidence rather than assuming they
were closed. Raw results alone do not satisfy calibrated/task or broader scaling
performance, nor prove uncapped event/step/EndRun and pressure-recovery coverage.

## Evidence locations and identity

Small working reports, scripts, review receipts and hashes are retained on SDF:

```text
/sdf/home/m/monarin/lcls2_worktree/psana2-gpu/notes/psana2_gpu_master_sync_20261117/
```

Generated SDF evidence is under
`/sdf/scratch/users/m/monarin/agent-work/lcls2/gpu-master-sync-20261117/`:
`remaining-45956052`, `calibration-45955820`, `remaining-45958415`,
`review-conflict-3/validation-callers-45960569`, and `master-slow-comparison/`.
Its checked source was the isolated `source/` worktree, not the unrelated home
checkout used to retain notes. The saved `validate_remaining.sbatch`,
`validate_mpi_calibration.sbatch` and `compare_master_slow.sbatch` record commands,
selection and environment; reruns must use fresh output directories.

Original Perlmutter evidence is under:

```text
/pscratch/sd/m/monarin/agent-work/lcls2/gpu-master-sync-20261119-attempt02/
```

Relevant subdirectories: `capture-61621117`, `reuse-check-02`,
`step2-contracts-02`, `step3-smoke-01/job-61659990`,
`step4-full-prefix-01/job-61664705`, `step4-crash-01/job-61679475`.
The first two runtime manifests are `capture-61621117/{baseline,candidate}-runtime.json`.
The reviewed profile/component draft lives in `step2-contracts-02`; it was
intentionally blocked and is not itself an executable campaign.

Private input directory:
`/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx100848724-r0051-pr-regression-20261117-attempt02`.
Its manifest/reference/preparation receipts identify the selected data; the
older fixture input specification must not replace the actual PR runtime pair.

These artifact locations require site access; they are not GitHub-hosted
attachments or a retention guarantee. Preserve essential final evidence before
scratch expiration. The concise results and identities here travel with the PR.

| Source report in the SDF notes directory | SHA256 at consolidation |
| --- | --- |
| `validation_20261119.md` | `53944d2790fbe77c109c8525463010a194161986b58ce0e6b61255eb481e7194` |
| `master_slow_comparison.md` | `5ccfe72031a9cbff74d0abb7c02a23aaef64e60bc95d963646c5a807f3b169fd` |
| `performance_step3_smoke.md` | `617cdfb828b5da10e6df6911197a5d6df73e88995f2149b4e0fc6335a3476199` |
| `performance_step4_full_prefix.md` | `68b540ff3698d4ce9ec253d158618f09ee82ff5971364f338e36aa347c31f5a4` |
| `perlmutter_performance_transfer_20261120.md` | `54b074b08d2188b8cbcfd80f06962e3b7ae714d13451049b71a34658cce317e8` |

Selected primary evidence hashes from the saved receipts:

| Artifact | SHA256 |
| --- | --- |
| SDF `remaining-45956052/logs/full-including-slow.log` | `02a799b2fe8461c3aa50e8d26ef2c7a88189fbb0b0896713ea4d8785ce52e4f0` |
| SDF `calibration-45955820/automatic.log` | `3de97aefc559fc7b0c02ca48d50fa4dffe95b54c58ebe58d9338031056d08c2a` |
| SDF `calibration-45955820/explicit.log` | `f43a0eac71f93bd9cc779961b8e6b72d1994238187367dc639b4c3124042ad40` |
| Perlmutter baseline runtime manifest | `a21a495e360bfbacdd1ef8136cafa15a13103a804a4af6c300000d70d14bb458` |
| Perlmutter candidate runtime manifest | `31d893bf02585ac3c04137457d526a41253aba0c1b72d2abd2f5001f47801506` |
