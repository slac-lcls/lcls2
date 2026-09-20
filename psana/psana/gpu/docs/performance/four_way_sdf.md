# Matched A / B / B+ / B+gather comparison

Completed **job 38564718**, sdfampere004, 2026-09-18. Slurm reports
**COMPLETED / 0:0**, allocation elapsed **57:46**. All **16 samples and four
CPU-reference preflights pass the audit**; no samples were excluded.
This report compares all four implementations within one allocation; earlier
node timings are context only and are not pooled into these results.

## Results: 10,000 events

| Variant | Median elapsed, s | Range, s | Events/s from median |
| --- | ---: | ---: | ---: |
| A | **28.338** | 24.890–29.476 | 352.9 |
| B | **39.063** | 33.429–40.898 | 256.0 |
| B+ | **32.388** | 29.675–33.450 | 308.8 |
| B+gather | **29.807** | 25.067–33.827 | 335.5 |

| Round | Execution order | A, s | B, s | B+, s | B+gather, s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | A → B → B+gather → B+ | 24.890 | 40.781 | 32.529 | 27.800 |
| 2 | B → B+ → A → B+gather | 27.852 | 40.898 | 33.450 | 33.827 |
| 3 | B+ → B+gather → B → A | 28.824 | 37.346 | 32.247 | 31.814 |
| 4 | B+gather → A → B+ → B | 29.476 | 33.429 | 29.675 | 25.067 |

Median comparisons:

- B → B+: **6.676 s / 17.1% less elapsed time**.
- B+ → B+gather: **2.581 s / 8.0% less elapsed time**.
- B → B+gather: **9.257 s / 23.7% less elapsed time**.
- B+gather versus A: **1.469 s / 5.2% more elapsed time**.

These percentages use ratios of the four-sample medians, not averages of paired
percentages. Gathering beats B+ in three of four rounds; round 2 is 0.377 s
slower. A and gathering ranges overlap, and gathering beats A in round 4.
Four rounds show substantial variability and do not establish statistical
significance or an invariant 5.2% gap. They do provide the requested direct,
same-allocation comparison without mixing historical timings.

## Implementations

| Label | Implementation | Revision |
| --- | --- | --- |
| A | Earlier fixed-offset detector path | `f52e90cc66` |
| B | GPU XTC parser with per-field location and gathering | `803a70011d` |
| B+ / O | Stream-grouped batched field location | `b0c9c3c02` |
| B+gather / G | Batched canonical gathering across the existing execution subbatch | Validated working tree based on `b0c9c3c02` |

Four separate installation copies were frozen before submission. A/B were
verified against the previous A/B/B+ run's recorded hashes. B+/gather were
verified against the completed B+/gather run's hashes. Per-run provenance
records source and native-extension hashes again and checks them at completion.
No production or installed-code changes are part of this rerun.

## Controlled workload

- 10,000 events, `mfx101210926` run 387, Jungfrau 32 segments, physical files
  s005–s009; shared staged inputs and CPU-reference calibration snapshot.
- GPU execution batch 20, depth 1, 8 GiB budget. SMD batch setting 1,000.
- Three MPI ranks: one SMD0, one EB, one BD; one A100; 48 requested CPUs;
  450 GiB requested host memory. Identical shared CPU mask for all ranks/variants.
- KvikIO compatibility ON, eight threads, 1 MiB tasks: CPU fallback, not GDS.
- No user D2H, bulk integration, detailed timing hooks, or profiler captures.
- Each sample follows a 100-event warmup. Cache residency must be >=99% before
  and after timing. Staging and cache preparation are outside the timer;
  detector setup and final GPU synchronization are inside it.
- Fresh CPU-reference preflights for all four variants precede measurements.

The four rounds use ABGO, BOAG, OGBA, GAOB. Each variant appears in each
position once, and every ordered adjacent pair appears once. All repetitions
are reported. This balances run order but does not eliminate shared-node noise.

The CPU mask is `1-4,9-16,40,47-57,65-68,73-80,104,111-121`.
This is not exclusive per-rank binding or GPU-local NUMA memory binding.

## Audit, placement, and earlier-run differences

All 32 before/after cache checks were **100% resident**. Initial cache
preparation required three passes (90.71%, 98.10%, 100%), all outside timing.
The common event timestamp SHA256 is
`23f9051ac9f98208310171f734251b0ccd1f40fa048d1748553cbd38cce559d6`;
payload is 335,571,760,000 bytes. Calibration snapshot SHA256 is
`c3679e0fb47bebfafe41f36750e88fe1ec0d6b71ef386814fb1eac37dc197ce9`.

Every sample uses A100 UUID `GPU-c27d62cd-6693-d20f-5167-0d5ccf9cda25`,
CuPy 13.6.0, CUDA runtime 12.9, and KvikIO 24.08.02. The audit verifies
runtime/import paths, all three ranks' CPU affinity, one GPU identity,
dataset count/order/bytes, calibration digest, and unchanged installed hashes.
The sampled device-memory peak is 3,631 MiB for A and 3,633 MiB for each parser
variant in every round. Sampling is every 500 ms at MiB resolution, not an
exact allocation or transient-peak measurement.

The earlier B+/gather job 38561649 also ran on sdfampere004, but used GPU UUID
`GPU-3e8528f8-fe59-785e-c2c4-c969ab13acda` and CPU mask
`1-4,9-16,46-57,65-68,73-80,110-121`. This rerun uses a different GPU and swaps
CPUs 46/110 for 40/104 in the mask. The builds and workload settings are
unchanged; the measured B+/gather medians here are 32.388/29.807 s rather than
37.275/28.578 s in that earlier allocation. The median gathering reduction is
therefore 8.0% here versus 23.3% there. These observed allocation differences do
not establish which factor caused the timing change. CPU/NUMA placement is
shared, and elapsed samples vary even within this allocation.

The older matched A/B/B+ job 38513845 ran on sdfampere033 with medians
21.850/33.561/28.743 s. Those values are historical context, not controls for
this four-way result. There are no new CPU phase timings or Nsight captures
in this clean rerun, so it does not attribute the remaining gap to a particular
host or device stage. Earlier launch-count evidence remains separate.

## Reproduction and evidence

```bash
sbatch validation/four-way-20260918/run.sbatch
python validation/four-way-20260918/audit.py \
  validation/four-way-20260918/job-38564718-warm-ab
python validation/four-way-20260918/summarize.py \
  validation/four-way-20260918/job-38564718-warm-ab
```

Harness: `validation/four-way-20260918/`. Generated installations and logs
remain local and ignored. `builds.json` records build identities. The audit
checks the 16 samples, four preflights, order, cache, workload hashes, runtime,
GPU identity, rank affinity, and unchanged installed hashes.

Existing [gather call-count evidence](batched_canonical_gather_sdf.md) was
collected separately and does not contribute timing samples to this report.
