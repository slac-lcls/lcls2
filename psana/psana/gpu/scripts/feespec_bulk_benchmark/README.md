# Feespec on Weka FFB

## Quick cold baseline

`quick.py --directory PRIVATE_FFB_XTC --constants SNAPSHOT --jf-cpu-log CPU_LOG`
reuses the verified six-stream input for 1,000-event cold samples in order
E-off, E-on, E-on, E-off. Batch 100, depth 1, and budget 8 GiB are unchanged.
Each fresh MPI process validates 200 warmup feespec arrays and three JF
raw/calibrated samples before evicting input. The measured loop has no
full-array D2H. It accumulates the existing reader byte/request/wait and
request-to-ready counters with one lightweight wrapper per read window.

SMD-derived extents bound `mincore` checks to the first 1,000 events in each
file. Cold eviction covers the private files; page residency must be <=1%
for each measured prefix. Tier checks require SSD-only Weka backing before
and after the sweep. Every measured run checks event identity and feespec
GPU sums. `--events 2000` supports the longer fallback if 1,000 is unstable.
Results and a four-sample summary are saved in `job-SLURM_JOB_ID/` on scratch.
This mode retains fresh processes and setup per sample; it does not yet reuse
initialized GPU workers across cases.

## Native fallback trace for the short cold baseline

Add `--fallback-trace` to `quick.py` to run eight cold samples: bulk off/on,
two rounds, each with a control and a trace. Round two reverses variant and
trace/control order. Copy `kvikio_fallback_trace.py`, its C source, and
`summarize_kvikio_fallback.py` from the parent scripts directory into the
scratch campaign. Compile the C source with the command in its header as
`fallback.so`. The required `stage.py`, `page_cache_residency.py`, `memory_state.py`, and
`warm_cache.py` helpers are included in this directory; copy them too.

Tracing is active only during the measured loop and buffers records in RAM.
The tracer observes native POSIX reads, H2D calls, and existing stream waits;
it adds no CUDA synchronization. Each trace must pass byte, operation-count,
triplet, and interval audits. Controls have no tracer preload. Both modes keep
the 200-event correctness warmup, measured-prefix eviction verification,
network traffic check, and before/after Weka SSD-tier check.

`summary.md` reports the numerator and denominator for the single-file
percentage: wall time with exactly one file in POSIX reads divided by wall
time with any POSIX read active. Concurrent calls on the same file count once;
idle gaps are excluded. Files are identified by stable open handles for this
single-chunk workload. Each trace also produces `.audit.json`, `.json`, and
`.bin` artifacts; `control-summary.md` retains the untraced baseline report.

## Combined JF + feespec mode

Submit on SDF with account `lcls:data` and QoS `normal` (the Ampere association
is `lcls:data@ampere`). Bare `lcls` selects the default preemptable account.
Request one A100; whole-node exclusivity is not required for this harness.

Pass `--include-jf --constants /path/to/cpu-calibration.pkl.gz
--jf-cpu-log /path/to/cpu-check.log` to `run.py`. This stages physical streams
0 and 5–9 and runs the same 12-case matrix with batch size 100, depth 1, and
an 8 GiB GPU budget. Reserve sufficient host RAM for the roughly 336 GB input
to remain in the page cache (the submitted job reserves 700 GiB).

A keeps `gpu_det='jungfrau'`, calibrates JF on the GPU, and extracts feespec
on the CPU followed by per-event H2D and GPU sum. E uses
`gpu_det=['jungfrau', 'feespec']`; its bulk switch applies globally to both.
JF calibration executes in all cases; clean runs do not export JF images.
This measures the current combined behavior, including admission changes,
and does not isolate feespec-only coalescing or its incremental cost.

Drop `field`, `segments`, and `values` after the field-view context and before
advancing the event iterator. A Python `with` block leaves those variables
bound; the feespec slice can otherwise retain the entire mixed input buffer
through a later buffer replacement. The maintained consumer explicitly clears
them. See `docs/performance/jf_feespec_admission_failure.md` under `psana/gpu`
for the reproduced failure and allocation accounting.

Validation runs check every feespec array and three JF raw/calibrated arrays
against the existing CPU baseline. The frozen JF calibration snapshot is
shared read-only and hashed before/after. Diagnostic read counts are broken
down by physical file. Warm preparation uses `warm_cache.py` under NUMA
interleave; residency is verified separately for all six files. That helper is included in this directory.

## Feespec-only mode

Proof-of-concept comparison of historical A with per-event CPU field extraction
and H2D, current E bulk off, and current E bulk on. Feespec only; JF is absent.
Current E is frozen at ac87a93b2. No production runtime module is modified.

Workload: mfx101210926 run 387 s000, first 10000 events; hproj is an 8 KiB int32
array. Preserve complete shared-stream datagrams and SMD offsets. A uses the
normal CPU path, including any CPU read coalescing. E uses a process-local
exclusive routing exception for feespec's stream and reads all its bytes once.
Both produce per-event GPU int64 sums. E's public on_gpu_view API includes
locator metadata D2H and consumer lifetime bookkeeping in the timing.

Batch 100, execution depth 1, 8 GiB per-BD GPU budget for E. A has no detector
GPU pipeline in this small-detector-only case; its user loop does cp.asarray
and the identical GPU sum. One BD plus EB/SMD0, one A100, KvikIO CPU fallback
with 8 workers and 1 MiB tasks. Three validation runs copy every array back
for exact CPU-reference checks and collect API read counts. Twelve clean runs
contain no read hooks and no full-array D2H. End-of-loop GPU synchronization
is timed; checksum retrieval, initialization before the loop, and cache
preparation are excluded. Lazy setup inside run.events remains timed.

Two rounds reverse variants and warm/cold order. Warm >=99% page residency;
cold <=1% before the loop plus physical Ethernet receive counters. FFB copies
must have full SSD coverage and no object/remote backing, checked before/after.
Server-side caches are not flushed; node background traffic can contribute
to NIC counters. This is a small-input latency test, not a network saturation test.

Copy all scripts in this directory to a new shared-scratch campaign directory. Provide builds.json with
A and E frozen prefixes and complete file hashes. The controller stages a
private bounded prefix under /sdf/data/lcls/drpsrcf/ffb/users/monarin on the
compute node and writes logs/results/summary to shared scratch.
