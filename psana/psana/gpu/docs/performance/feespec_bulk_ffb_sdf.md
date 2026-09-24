# Feespec: A with event-loop H2D versus E bulk off/on

Measured 2026-09-24, Slurm job **38989141**, `sdfampere019`, one A100.
The job completed successfully in 3m32s. All 12 clean samples and three
separate full-array validation runs passed.

## Workload and implementation

- Feespec only, `mfx101210926` run 387, stream s000, first 10,000 events.
  Jungfrau is absent from this experiment.
- `raw.hproj`: 2,048 int32 values, 8,192 bytes/event. Complete shared-stream
  datagrams total 97,354,744 bytes; array payload totals 81,920,000 bytes.
- Batch size 100, one BD plus EB/SMD0, GPU depth 1, 8 GiB GPU budget for E.
  KvikIO CPU fallback, eight workers, 1 MiB task size; GDS is disabled.
- A uses historical build `f52e90cc66d4c8c175b7689922c7e441e78b367f`:
  normal CPU `Detector('feespec').raw.hproj(evt)` followed by
  **`cp.asarray(host_values)` in every event**, then an int64 GPU sum.
  This gives 10,000 user H2D uploads. A has no detector GPU pipeline here.
- E uses frozen integrated build `ac87a93b2`, with bulk off or on. The public
  `evt.gpu.detector('feespec').field('raw', 'hproj').on_gpu_view(...)` API
  provides the input to the same per-event int64 GPU sum. Its locator-metadata
  D2H and consumer lifetime bookkeeping are included.
- A process-local benchmark exception permits E to route feespec's entire
  shared stream to the GPU. Original Configure and datagram bytes are retained;
  diagnostic checks verify no duplicate CPU bigdata reads. Calibration service
  lookup is bypassed for this uncalibrated field in all variants.
- Maintained harness: `psana/psana/gpu/scripts/feespec_bulk_benchmark/`.
  No production runtime module was changed.

## Measured throughput

Two rounds in one allocation, reversing variant and cache order in round 2.
Aggregate rate is 10,000 divided by the median of the two elapsed times.

| Variant | Cold R1 / R2 events/s | Cold aggregate events/s | Warm R1 / R2 events/s | Warm aggregate events/s |
|---|---:|---:|---:|---:|
| A + event-loop H2D | 3,408.8 / 3,388.8 | **3,398.8** | 3,420.6 / 3,429.3 | **3,424.9** |
| E bulk off | 2,989.7 / 3,100.4 | **3,044.1** | 3,140.6 / 3,113.6 | **3,127.0** |
| E bulk on | 3,041.9 / 3,039.0 | **3,040.5** | 3,057.1 / 2,977.6 | **3,016.8** |

| Variant | Cold seconds / 10,000 events | Warm seconds / 10,000 events |
|---|---:|---:|
| A + event-loop H2D | 2.9422 | 2.9198 |
| E bulk off | 3.2851 | 3.1979 |
| E bulk on | 3.2890 | 3.3148 |

Each sample uses a fresh MPI process and a 100-event warmup. The timer includes
the event loop and final GPU completion; the reported elapsed time is the
maximum across the three ranks. Cache preparation, initialization before the
loop, final sum retrieval, and teardown are excluded. Lazy setup inside
`run.events()` remains timed. Clean samples have no read-count hooks and no
full-array D2H. Every clean sample checks timestamps and GPU sum hashes.

## Read counts and correctness

Counts below come from the three separate warm validation runs. Each validates
all 10,000 complete arrays against the CPU reference, including A's GPU copies.
These instrumented runs are excluded from the throughput table.

| Variant | CPU BigData `_read` calls | KvikIO API reads | GPU input windows | KvikIO requested bytes |
|---|---:|---:|---:|---:|
| A + event-loop H2D | 192 | 0 | 0 | 0 |
| E bulk off | 0 | 10,000 | 100 | 97,354,744 |
| E bulk on | 0 | 182 | 100 | 97,354,744 |

Bulk on reduces E's KvikIO API requests by **54.9x (98.18%)**, with identical
requested bytes. Batch boundaries and noncontiguous extents leave 182 requests
across 100 windows. A already combines CPU input reads: per-event field access
and H2D do not imply one CPU read request per event. The CPU and KvikIO counts
refer to different software APIs; neither is a count of POSIX syscalls or
physical storage operations.

Diagnostic read submission/wait totals were 0.2044/0.1613 seconds for E-off
and 0.1310/0.0428 seconds for E-on. These are illustrative measurements under
full-array validation, whose synchronization changes overlap. They must not
be subtracted from clean elapsed times or treated as a clean phase breakdown.

All arrays matched SHA-256
`b86c897effd2cfdb64f1ecc7f95b16e7ce367156d49eeecbe6ef36ede5cce3e5`.
Five harness tests passed, covering matrix order and Weka tier validation.

## Storage and measurement checks

The staged prefix is under
`/sdf/data/lcls/drpsrcf/ffb/users/monarin/feespec-bulk-38989141/xtc`.
Before and after the sweep, Weka reported complete SSD write-cache coverage:
114,191,264 bytes for bigdata and 3,126,064 bytes for SMD, with **zero object
or remote storage bytes**. The bigdata prefix includes trailing staging margin.

All cold samples started at 0% node page residency; all warm samples started
and ended at 100%. Physical Ethernet RX was 138.0–140.1 MB per cold sample
and 18.6–21.3 MB per warm sample. NIC counters include filesystem overhead and
background traffic. Weka server caches were not flushed: cold means node page
cache cold on FFB, not cold at every storage layer.

The assigned GPU's sampled memory peak across each complete process, including
warmup and setup, was 429 MiB for A and 431 MiB for each E variant. Sampling
was every 250 ms; these are device occupancy observations, not owned-allocation
peaks. Build/script hashes and rank CPU affinity were checked by the controller.

## Interpretation

Bulk coalescing works, but **this workload shows no end-to-end throughput gain**:
E-on is 0.1% below E-off cold and 3.5% below warm. Relative to A with per-event
H2D, E-off is 10.4% lower cold and 8.7% lower warm; E-on is 10.5% and 11.9%
lower. Two rounds provide a descriptive comparison, not statistical certainty.

The test demonstrates that avoiding thousands of small API reads alone does
not make the current E user path faster. It does not isolate the cost of
parser work, field metadata access, leases, or per-event consumer launches.
Those remain candidates for focused timing. Historical A and E are complete
different builds, so their difference cannot be attributed solely to the parser.

This small input reaches roughly 0.03 GB/s of useful datagram throughput and
does not establish a network bandwidth limit. A combined JF + feespec workload,
shared-budget admission, and selective per-detector bulk scheduling remain
separate questions.

## Artifacts

Scratch campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/feespec-ffb-20260924`.
The workspace link `validation/feespec-ffb-20260924` points there.
`job-38989141/summary.md` and `summary.json` contain the generated summary;
`results.json`, `diagnostics.json`, `provenance.json`, `reference.json`, per-case
logs, and GPU CSVs preserve the evidence. Frozen benchmark scripts, submission
manifest, and build hashes are in the campaign root.
