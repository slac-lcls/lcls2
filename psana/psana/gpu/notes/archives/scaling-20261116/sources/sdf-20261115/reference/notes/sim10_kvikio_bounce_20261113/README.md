# Sim10 KvikIO bounce-buffer screen — November 13, 2026

**Complete — job 44870463, exit 0:0.** All 31 GPU tests, 4 pilots, 8 full runs,
and the final audit passed; the frozen re-audit is byte-identical and staged
inputs were removed. [Recorded results and plot-ready data](RESULTS.md) show
64/16 MiB mean-rate changes of −2.37% at 1 BD and +1.28% at 4 BDs. Retain the
16 MiB control: two repeats do not demonstrate a consistent benefit. Results
have not undergone independent interpretation review. Final accounting is
in `results/accounting.psv`; `evidence/accounting.psv` preserves the original
submission-time pending snapshot. The preparation plan below is historical.

The user explicitly requested preparation, review, and submission of this
16/64 MiB screen. It follows the [task-size result](../sim10_kvikio_tasksize_20261113/RESULTS.md)
and is linked to the [single-node report](../sim10_single_node_scaling_20261113/README.md).
This is a controlled confirmation screen, not a claim that bounce size is a
demonstrated bottleneck. Typical requests (524724, 629664, 734604 bytes) already
fit inside either buffer size, so unchanged chunking is expected.

| BDs | Workers/BD | Batch | Bounce MiB | Task MiB | Bound physical cores |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 50 | 16, 64 | 1 | 17 |
| 4 | 4 | 200 | 16, 64 | 1 | 34 |

Four traced pilots (1000 events per BD) precede eight untraced full runs
(100000 global events each), in forward/reverse order: 810000 events total.
Use one A100-SXM4-40GB and one CUDA stream per BD, exclusive Ampere allocation,
6 tasks × 6 CPUs = 36 physical cores requested, 160G host memory, 90 minutes.
The exclusive allocation can reserve all node CPUs/GPUs; only one GPU is used.
Each case has a 900-second timeout. The prior identical-sized task screen took
34m50s, but completion within the time limit is not guaranteed.

## Fixed controls and acceptance

Pinned PR298: `67c327a148a08b2aaf972e67e1e02e367c2b298f`. No framework changes.
Existing accepted five-JF Sim10 100k fixture, 512 KiB bulk target, SMD1000,
6 GiB input budget per BD, KvikIO 26.10 CPU fallback ON, auto-direct OFF.
Staging-only GpuTask validates descriptors and records callback metadata;
no expansion/calibration/cube kernels or per-event publication D2H.

Set `KVIKIO_BOUNCE_BUFFER_SIZE` before process startup and check the effective
default before I/O and after the loop. Keep task size at 1 MiB; reject a
conflicting framework task override. Freeze eight installed KvikIO identities,
including the POSIX staging and buffer-pool headers, alongside the accepted
native-build compatibility receipt (same documented meson exception).

Require 31 native GPU tests, exact global timestamps/read/byte/callback counts,
no pressure drains or CPU payload reads, all slot/reader/budget owners drained,
aggregate GPU memory headroom, disjoint CPU affinity and interleaved NUMA policy.
Each full run reads 500000 datagrams / 335820000000 bytes, with 2000 callbacks
at batch 50 or 500 at batch 200. After DataSource setup, fsync/fadvise/mincore
must show zero initial payload residency. Idle-adjusted physical NIC RX must
cover at least 90% of payload. This corroborates network-served Weka FFB input,
not cold backend storage. Reap owned MPI groups before stage cleanup; the
inherited staging-copy-thread termination-grace limitation remains.

## Host memory and diagnostic evidence

Installed `detail/posix_io.hpp::posix_device_io` acquires a reusable pinned
bounce buffer, uses `min(buffer_size, remaining_bytes)`, and synchronizes each
H2D before reusing it. `bounce_buffer.hpp` describes the reusable process pool.
For this synchronous POSIX path, a planning estimate of one active buffer per
worker gives the following aggregate demand across BDs:

| Profile | 16 MiB | 64 MiB |
|---|---:|---:|
| 1 BD × 8 workers | 128 MiB | 512 MiB |
| 4 BDs × 4 workers | 256 MiB | 1024 MiB |

These are source-based estimates, not measured pinned allocation counts or
total process-memory bounds. Require MemAvailable to exceed the aggregate
estimate plus 8 GiB before the loop. Record every rank's VmRSS, VmHWM, VmLck,
VmPin and host MemAvailable at startup, before the loop and after timing ends.
VmHWM is the lifetime process peak; RSS and kernel locked/pinned counters do
not isolate KvikIO or reliably account for all CUDA pinned memory. Do not
report their difference as exact pinned usage. No allocator interposition or
continuous sampling is added to timing runs.

Pilots retain payload-filtered per-BD strace with exact byte coverage and saved
event-loop windows. The audit recomputes histograms from raw traces; failures
stop subsequent cases. POSIX call counts do not directly measure native task
counts, H2D timings, or buffer allocations. Pilot tracing overhead means pilot
rates are excluded from throughput comparisons. Full runs bypass strace and
require TracerPid zero. Observation windows assume no wall-clock step.

## Review, reporting and exclusions

Maximum three independent preparation-review rounds on exact commits. Continue
the independent in-session Codex reviewer used for earlier screens after
automatic approval review blocked external Claude source/context disclosure.
Freeze the reviewed harness/PR, pass scheduler test-only, journal and submit
the saved command once. Never infer submission from a client reply alone.

Host tests cover settings, corrupt evidence, host memory, trace parsing and an
actual local strace child process, CPU demand, cleanup and native receipt gates.
CUDA-under-strace, actual FFB/cache/NIC and native GPU tests remain allocation
gates. Same-allocation 16 MiB controls are the comparison denominators; two
full repeats are screening evidence, not confidence intervals. Save this as a
new cohort, grouping by bounce size as well as BD/worker/batch/task settings.
No extra GPU-count, worker, batch, task-size, application or multi-stream axes.
Multi-stream/forced-split concurrency remains deferred under #308 acceptance.

## Submission

Submitted as **44870463** after independent preparation review passed on `de7513219568bc0acee99ffe3d3b61287e088f77`. Host acceptance: 63 tests passed; native compatibility and scheduler preflight passed. The job subsequently completed in **34m 43s** on **sdfampere018**; all allocation-time gates passed. See [results](RESULTS.md), [status](status.json), [preparation review](review/round-01.json), and [submission receipt](evidence/submission.json). Raw artifacts: `/sdf/scratch/users/m/monarin/gpu-validation/sim10-kvikio-bounce-20261113/attempt01`.
