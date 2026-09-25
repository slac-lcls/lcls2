# Four MiB bulk target and KvikIO tasks, pool depth 1

## Configuration and submission

Job **39038746**, account `lcls:data`, one A100, 48 CPUs, 128 GiB host RAM,
35-minute limit. Submitted for the requested current bulk off/on comparison.
Completed on `sdfampere002` in 19m26s, Slurm exit `0:0`. All 14 samples
validated and the driver recorded COMPLETE. Results recovered 2026-09-25
from the completed campaign after the original task history became unreadable
through the app's database reader.

## Completed results

Rates below use the two control rounds only, calculated from median loop time.
Native traces and pipeline diagnostics are separate samples.

| Bulk | Cache | R1 / R2 events/s | Median-time events/s | Median loop seconds |
|---|---|---:|---:|---:|
| Off | Cold | 121.35 / 122.43 | 121.89 | 8.2043 |
| On | Cold | 94.62 / 112.57 | 102.82 | 9.7260 |
| Off | Warm | 135.94 / 189.26 | 158.23 | 6.3199 |
| On | Warm | 119.02 / 155.76 | 134.93 | 7.4111 |

Bulk on took **18.5% longer cold** and **17.3% longer warm** than bulk off.
Warm results still vary substantially between rounds. This comparison does
not isolate a causal 4 MiB versus 1 MiB effect across allocations.

Cold traces show single-file time of 1.62% / 1.54% with bulk off and
2.09% / 1.79% with bulk on, as a fraction of POSIX-active wall time.
Time without POSIX reads was 35.33% / 39.29% of the BD loop with bulk off,
versus 48.48% / 54.46% with bulk on. That includes useful GPU/H2D work and
setup/tail; it is not a measurement of GPU idle time.

Pipeline diagnostics recorded 20 versus 29 execution subbatches and
28 versus 1,914 reservation calls for off versus on. Peak charged memory
was 7,302.275 versus 7,307.970 MiB, within the 8 GiB budget; this is owned
backing, not total CUDA process memory. Profiling setup, allocation/reuse,
and retirement remains the next investigation, not an established cause.

## Run settings

Both variants use:

- `gpu_bulk_target_bytes=4194304`, `KVIKIO_TASK_SIZE=4194304`.
- `n_gpu_streams=1` (execution pool depth), eight KvikIO workers, CPU fallback.
- 1,000 real JF+feespec events, batch 100, 8 GiB GPU budget, D2H chunk 0.
- The same private Weka FFB input and frozen calibration/reference data as
  the previous depth comparison; no object-store input.
- Two cold/warm control rounds, with reversed ordering in round 2.
- Separate two warm pipeline diagnostics and four cold native traces.

Every sample checks warmup arrays, measured timestamp/feespec checksums,
exact payload and API request totals. Cold requires <=1% page residency per
prefix and physical NIC RX >=98% of payload; warm requires >=99% residency
before and after. SSD placement and code/reference hashes are verified
before and after the sweep. Traces/diagnostics do not determine control rates.

## Real-data plan

The 4 MiB SMD preview still gives 5,019 bulk-on API requests: 19 feespec
bulks and 5,000 JF reads. Bulk off uses 6,000 requests. Every variant reads
33,566,911,424 payload bytes. Feespec grouping is constrained by batch bounds
and nonadjacent offsets before the byte limit. All JF datagrams exceed 4 MiB
and remain singleton requests. Thus this workload does not exercise a larger
feespec bulk after the target increase; it exercises larger KvikIO tasks.

The native traces confirm 10,000 JF POSIX operations: each of the 5,000 JF
reads has a 4 MiB chunk plus its remainder. Total POSIX operations were
11,000 with bulk off and 10,019 with bulk on, including feespec. The previous
1 MiB configuration used 37,000 JF operations.

## Code and checks

Runtime: `c8f6b6cdf` plus a frozen patch adding the public positive-integer
`gpu_bulk_target_bytes` parameter and forwarding it to `GroupReadSchedule`.
Default remains 1 MiB. No native rebuild was needed. The benchmark verifies
the requested task size; trace auditing now reads task size from metadata.

CPU unit suite: 387 passed initially; one new parameterized test had an
outdated assertion that the first parser submission always contains two
inputs. With a deliberately tiny target the first execution boundary changes.
Corrected that test to check total parsed inputs and preserve the original
first-submission assertion for the default target. All three controller
cases then passed. Benchmark harness: 12 passed. `git diff --check` passed.

## Artifacts

Campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-size4m-20260925`

- `read-plan.json`: complete real-SMD plan at 4 MiB.
- `python/`, `source.patch`, `builds.json`: frozen runtime and hashes.
- `run.sbatch`, `job-39038746.log`: allocation and driver log.
- `job-39038746/summary.md`, `results.json`, `provenance.json`: completed
  validated results; `summary.json` records 14 accepted samples and completion.

Compare previous 1 MiB measurements as historical context only: they came
from another allocation, and warm depth-1 performance varied substantially.
