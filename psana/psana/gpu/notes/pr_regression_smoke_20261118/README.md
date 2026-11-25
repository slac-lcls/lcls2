# Six native run51 smoke cases — 2026-11-18

All six cells passed on Perlmutter job **61503769**, **nid001765**. The allocation
COMPLETED/0:0 in **5:10**. All four fresh placement probes and all twelve workload
steps (six warmup + six measured) completed 0:0. In-job direct audit showed only
batch/extern; the final direct query found no remaining allocation steps.
Final independent Claude review **PASS**, with no blocking findings. Default
campaign qualification flags stay false.

| Cell | BDs / GPUs | Measured events per BD | Result |
|---|---|---|---|
| cpu-raw-events-bd1 | 1 / 0 | 1000 | PASS |
| gpu-raw-events-bd1 | 1 / 1 | 1000 | PASS |
| cpu-raw-events-bd2 | 2 / 0 | 500 + 500 | PASS |
| gpu-raw-events-bd2 | 2 / 1 | 500 + 500 | PASS |
| cpu-raw-steps-bd1 | 1 / 0 | 1000 | PASS |
| cpu-raw-steps-bd2 | 2 / 0 | 500 + 500 | PASS |

Every cell used a separate 200-event warmup and then the same **1,000-event**
prefix: **33,557,176,000 bytes (31.253 GiB)** across streams 3/5/7/8/9 and all
32 Jungfrau segments. Prefix preparation matched 5,000 selected headers to the
independent SMD plan and hashed exact disjoint private ranges. CPU steps matched
the native step index 0. Two BDs each delivered 500 events. GPU callbacks used
20-event batches; total GPU requested/read/useful bytes all matched the reference.
This is raw datagram/event validation, not calibrated or pixel-exact validation.

The prepared private copy and original 10k manifests were reused unchanged;
new smoke-prefix manifests live only in this job's output. The known-API runtime
was `1995752058bb9aaa3f16ccd86c22e84153257538`, not a newly selected PR baseline.
No production psana package, dependency installation or calibration was changed.

Settings: batch_size 20, one GPU stream, 8 GiB budget per BD, bulk target 1 MiB,
KvikIO 4 workers/1 MiB tasks/16 MiB bounce, compatibility ON, GDS false, automatic
direct I/O disabled. GPU work includes dense input preparation/gather plus a
metadata-checking callback, with no user kernel, output or payload D2H. Both GPU
cells used the same A100 UUID `GPU-a626c6b0-fa28-960d-be23-0d212e7bb1ad`.
Ranks bind four distinct physical cores each; step reservation is 16 physical
CPUs/rank. Requested 64 CPUs/one GPU/128 GiB; allocation actually reserved 128
logical CPUs (64 physical), four A100s and 128 GiB. Runtime/import/file/settings
checks passed before and after each workload.

## Cache and counters

Both mincore observations before every measured loop reported **zero selected
resident bytes** after private-range eviction. This establishes the adapter's
client-page-residency check for the 1k prefix; it does not establish storage-cold
reads, control SMD/server caches or replace 10k cache/accounting qualification.

The `hsn0..3` kernel network-device counters were readable and monotonic, but
reported only **0.408–1.254 MB RX** and **1.179–2.207 MB TX** per measured case,
despite 33.56 GB of selected datagrams. They do **not account for the workload's
data traffic**. RDMA/storage-traffic coverage is unqualified. Do not derive
storage/NIC bandwidth or a link ceiling from these counters or claim physical
read volume from the event-byte denominator. Counter coverage is a required
follow-up before performance interpretation. This smoke criterion checked
availability, not traffic attribution, and no bandwidth result is published.

Per-BD RSS high-water observations were approximately 283–552 MiB; GPU pool end
snapshots were 0.625–1.252 GiB used/reserved per BD. These are not GPU/pinned peaks.
Slurm MaxRSS was inconsistent for short steps (0 or 512K) and differed markedly
from per-process ru_maxrss; retain the raw accounting but do not treat its zeros
as measured zero memory. Peak GPU/pinned use and other missing component counters
remain unqualified. Diagnostic loop/process timings are retained in the JSON;
a single 1k correctness pass is not an overhead-qualified throughput comparison.

## Evidence and review

Production source: `2a13789d6c2d03d8b7c7dd32050bbce7644e76fd`.
[Hardware evidence](hardware-evidence.json) contains per-case identities, bytes,
cache outcomes, placement, timing, BD distributions, callback/read/delivery
counters and scheduler receipts. Coordinator revalidated all **243 indexed
files**, including **64 placement capture files**, all input/runtime pins, every
executed source against Git, independent prefix derivation, immutable input
metadata/SMD hashes, and final scheduler records. No large XTC rehash ran on login.
Offline final suite: **99 tests PASS** in 23.469 seconds on the executed code.

Successful launch: [smoke-02.sbatch](smoke-02.sbatch), frozen-02 sources.
[Source hashes](frozen-02-SHA256SUMS) and [input pins](inputs-01.json) are committed.
[Final accounting](job-61503769-final-accounting.txt) and
[final step audit](job-61503769-final-step-audit.json) are retained.
Raw evidence: `/pscratch/sd/m/monarin/psana2-gpu/pr-regression-smoke-20261118/`.
One compute attempt completed successfully; no failed hardware cohort was discarded.

[Round1 review](review-01.json) passed; [response](response-01.md) describes
additional input-association, allocation-audit and orchestration-test checks.
[Round 2 review](review-02.json) passed on
`56c2a74f6250f40195702d057c5327aa69213cdc`, confirming all four prior findings
resolved. [Response](response-02.md) records the remaining documentation
clarification. Closure changes only documentation and review records; executed
source remains unchanged from the production commit above. The reviewer had
read-only access to committed compact evidence; raw scratch hashes/tests were
verified by the coordinator, not independently rerun by Claude.

Each workload process logged one MPICH “pending 1 references” message at exit
(12 occurrences), while all steps exited 0. Its cause has not been established;
this report does not declare it harmless or claim complete communicator-lifetime
qualification. Inspect it before defining a cleanup/lifetime regression gate.
Empty-calibration warnings are consistent with explicit skip_calib_load. Stream 0
fallback messages are consistent with the private five-stream selection, which
passed exact opened-file and event/byte checks. GPU CPU-fallback messages match
the configured compatibility ON path.

## Next qualification

Use [SMOKE.md](../../scripts/performance/regression/SMOKE.md) to repeat these checks
on a future compatible runtime. New allocations require new placement manifests.
The driver deliberately creates correctness requests without claiming campaign
readiness; saved blockers include native_adapter and all qualification flags
remain false. The component contract is explicitly unqualified.

Next: validate traffic-counter coverage and inspect the MPI-reference warning;
qualify full 10k cache/byte accounting, instrumentation overhead and component
bounds; then timed A/A before PR A/B. The prefix covers one native step and an
artificial max_events cutoff, not multiple steps or the dataset's full EndRun.
