# Short cold baseline for GPU bulk-read scheduling

Follow-up native file-concurrency measurements are in
[the cold Weka trace report](jf_feespec_cold_quick_trace.md).

Job **39008696**, 2026-09-24, `sdfampere030`, completed successfully in
**4m25s**. Account `lcls:data@ampere`, normal QoS, one A100, one BD plus
EB/SMD0. All four samples passed the benchmark audit.

## Fixed workload

- 1,000 events from `mfx101210926` run 387, JF streams 5–9 plus feespec s000.
- Batch size 100, GPU depth 1, per-BD GPU budget 8 GiB.
- Frozen Integrated E build `ac87a93b2`, corrected benchmark view lifetimes.
- KvikIO CPU fallback, eight workers, 1 MiB tasks. Global bulk switch only.
- 33,566,911,424 useful full-datagram bytes per sample.
- Cold only; order **off R1, on R1, on R2, off R2**, same allocation.
- Each fresh MPI process validates 200 warmup feespec arrays and three JF
  raw/calibrated samples, then evicts the input before timing.
- Timed loop retains JF calibration and per-event feespec GPU sum; no full-array
  D2H. All 1,000 timestamps and feespec GPU sums are verified after timing.
- Existing reader counters are accumulated by a lightweight wait wrapper,
  without native-operation tracing or per-request instrumentation.

## Results

| Variant | R1 events/s | R2 events/s | Rate from median time | Median loop s | API requests/sample |
|---|---:|---:|---:|---:|---:|
| E bulk off | 130.10 | 133.42 | **131.74** | **7.5908** | 6,000 |
| E bulk on | 79.47 | 81.57 | **80.51** | **12.4214** | 114 |

| Read request-to-ready | R1 s | R2 s |
|---|---:|---:|
| E bulk off | 5.1543 | 5.0057 |
| E bulk on | 9.7826 | 9.4217 |

Paired on/off elapsed-time ratios are **1.637** and **1.636**. The aggregate
bulk-on loop is **63.6% longer**, or **38.9% lower event throughput**, despite
52.6 times fewer API requests and identical bytes read.

The loop difference is 4.831 seconds; the request-to-ready difference is
4.522 seconds (93.6% of that difference). This supports focusing on read
scheduling. Request-to-ready measures submission through completion and can
overlap other work; it is not a separate additive phase or proof of a specific
filesystem/inode mechanism.

## Controls

All six measured input prefixes had **0% resident pages** before every timed
sample. Prefix bounds came from the first 1,000 SMD event descriptors; cache
checks did not dilute residency over the larger 10,000-event staged files.
All samples requested/read exactly 33,566,911,424 bytes. Physical NIC RX was
36.2–37.6 GB, including filesystem overhead and possible background traffic.

Weka reported full SSD coverage with zero object/remote backing before and
after the sweep. Cold means the node page cache was evicted; server caches
were not flushed. Build/script hashes, rank affinity, correctness warmups,
event identity, and result hashes passed. Hardware/context initialization,
cache preparation, checksum retrieval, and teardown are outside loop timing.

## Use as an iteration baseline

This four-sample run reproduces the substantial cold regression with nearly
identical paired ratios. **Use 1,000 events as the working baseline for
scheduling experiments**, holding all settings and input bytes fixed.
The measured complete-job turnaround was 4m25s, faster than the initial
10–15 minute estimate; it is an observation, not a guaranteed future duration.

These two rounds establish a useful local comparison, not precise universal
rates across nodes or storage load. Repeat the baseline alongside each
candidate in one allocation. If the gap becomes comparable to variability,
increase to 2,000 events or additional pairs. After a promising change, check
warm behavior and the existing 10,000-event acceptance workload.

## Artifacts and entry point

Maintained controller: `psana/psana/gpu/scripts/feespec_bulk_benchmark/quick.py`.
Seven CPU harness tests passed before submission. No production runtime module
was changed for this baseline.

Scratch campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-cold1k-20260924`.
`job-39008696/` contains `summary.md`, `summary.json`, `results.json`,
`provenance.json`, measured/warmup manifests and references, per-case logs,
and GPU CSVs. The workspace symlink `validation/jf-feespec-cold1k-20260924`
points to the campaign.
