# JF + feespec on Weka FFB: corrected comparison

Measured 2026-09-24, job **38997826**, `sdfampere031`, one A100,
`lcls:data@ampere`, normal QoS. Completed successfully in **54m49s**.
All 12 clean timing samples and three separate validation runs passed.

## Configuration

- `mfx101210926`, run 387, first 10,000 aligned events: feespec s000 plus
  JF streams s005–s009. Useful full-datagram input: 335,669,114,744 bytes.
- Batch size 100, GPU depth 1, 8 GiB per-BD GPU budget, one BD plus EB/SMD0.
  KvikIO CPU fallback, eight workers, 1 MiB tasks; GDS is disabled.
- A: historical build `f52e90cc66d4c8c175b7689922c7e441e78b367f`, JF GPU
  calibration pipeline plus CPU feespec extraction and one 8 KiB H2D upload
  per event. E: integrated build `ac87a93b2`, GPU parsing for both detectors,
  with bulk off or globally on.
- Every variant computes the same per-event feespec GPU int64 sum. JF
  calibration executes in all variants; clean runs do not export JF images.
  E includes public field-access locator-metadata D2H and lease costs.
- E's shared-stream routing exception remains benchmark-local and preserves
  complete input datagrams. Frozen JF calibration is shared read-only.
- Corrected benchmark releases borrowed feespec aliases before advancing the
  event iterator. See `jf_feespec_admission_failure.md`. Production runtimes
  are unchanged; original cancelled/failed attempts are excluded.
- Fresh MPI process per sample, 100-event warmup. Timer covers the event loop
  and final GPU synchronization, using maximum elapsed time across ranks.
  Setup outside the loop, cache preparation, final sum retrieval, and teardown
  are excluded. Lazy setup inside the loop remains included.

## Rates and elapsed times

Two rounds in one allocation; variant/cache order reversed in round 2.
Aggregate rate is 10,000 divided by median elapsed time across the two rounds.

| Case | Cold R1 / R2 events/s | Cold aggregate events/s | Warm R1 / R2 events/s | Warm aggregate events/s |
|---|---:|---:|---:|---:|
| A + feespec event-loop H2D | 157.3 / 158.5 | **157.9** | 292.2 / 311.8 | **301.7** |
| E bulk off | 130.1 / 157.9 | **142.7** | 295.9 / 281.4 | **288.4** |
| E bulk on | 81.6 / 81.7 | **81.7** | 252.6 / 313.1 | **279.6** |

| Case | Cold seconds / 10,000 | Warm seconds / 10,000 | Cold useful GB/s | Warm useful GB/s |
|---|---:|---:|---:|---:|
| A + feespec event-loop H2D | 63.333 | 33.150 | 5.300 | 10.126 |
| E bulk off | 70.096 | 34.672 | 4.789 | 9.681 |
| E bulk on | 122.435 | 35.767 | 2.742 | 9.385 |

## Read counts from separate validation runs

| Case | JF KvikIO requests | Feespec KvikIO requests | Feespec CPU BigData `_read` calls | Feespec user H2D uploads |
|---|---:|---:|---:|---:|
| A | 50,000 | 0 | 182 | 10,000 |
| E bulk off | 50,000 | 10,000 | 0 | 0 |
| E bulk on | 910 | 182 | 0 | 0 |

E bulk-on reduces total KvikIO API requests from **60,000 to 1,092** (54.9x),
with identical requested bytes. Each of six streams has 182 bulk requests.
A's CPU reader already combines feespec datagrams into 182 calls. Its
per-event H2D does not imply per-event CPU I/O. Zero user H2D uploads for E
does not mean zero transfers: KvikIO fallback performs its own CPU-to-GPU
transfers. These are API counts, not POSIX or physical storage operations.

Each validation run compares all 10,000 feespec arrays and three JF
raw/calibrated samples against CPU references. All clean runs verify exact
timestamps and feespec GPU sum hashes. Validation rates are excluded above.

## Measurement checks

Private staged files reside on Weka FFB. Tier checks before and after the
sweep found full SSD coverage and zero object/remote storage backing.
All six cold samples began at **0% node page residency**. All warm samples
began and ended at **100%**. Cold physical NIC RX was 360.8–380.4 GB per sample;
warm RX was 0.186–2.610 GB. Counters include filesystem overhead and background
traffic. Weka server caches were not flushed. Thus cold refers to network
reads from SSD-backed FFB with the node page cache evicted, and warm refers
to node DRAM; this is not a local-NVMe comparison.

Assigned-GPU memory peaks sampled across complete clean-case processes were
7,729 MiB (A), 7,735 MiB (E-off), and 7,741 MiB (E-on). These include setup
and allocator/runtime storage and are sampled device occupancy, not exact
owned-allocation peaks. Build/script hashes and CPU affinity checks passed.
The allocation reserved one GPU, 48 CPUs, and 700 GiB host RAM without whole-node
exclusivity; node-wide NIC observations may include other activity.

## Interpretation

- **Cold bulk-on is consistently slower:** 81.6 and 81.7 events/s versus A's
  157.3 and 158.5. The aggregate is **48.3% below A** and **42.7% below E-off**.
  Its event loop takes 122.4 seconds versus 63.3 for A.
- E-off's aggregate is **9.6% below A cold**, but its two cold rounds vary
  substantially (130.1 versus 157.9). Round 2 approximately matches A.
- Warm aggregates are **4.4% below A for E-off** and **7.3% below A for E-on**.
  Warm E-on varies from 252.6 to 313.1 and slightly exceeds A in round 2.
  Two rounds do not establish a precise small warm-performance difference.
- Fewer API requests did not produce a combined-workload speedup. Current
  bulk-on changes JF scheduling and admission too; this is not a test of
  feespec-only bulk with JF kept on per-event reads. Earlier file-scheduling
  findings are a plausible lead for cold behavior, but this sweep contains
  no trace establishing that cause.
- There is no JF-only control in this allocation, so these measurements do
  not isolate incremental feespec overhead. They also do not establish a
  raw storage/network bandwidth ceiling.

The next targeted comparison would retain JF's per-event read scheduling
while coalescing feespec only, with an explicit bounded scheduling policy.
That requires a separate implementation or clearly labeled benchmark change.

## Artifacts

Campaign:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-feespec-ffb-20260924-fixed`.
Its `job-38997826/` directory contains generated `summary.md`, `summary.json`,
`results.json`, `diagnostics.json`, `provenance.json`, references, per-case logs,
and GPU CSVs. Frozen scripts, build hashes, and submission details are in the
campaign root.
