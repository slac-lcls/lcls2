# PR246: Perlmutter field results and proposed performance disposition

The Perlmutter field campaign completed successfully. Mona proposes prioritizing
the batched path and deferring optimization of the alternative per-event
`on_gpu_view()` API. **Claude should review this evidence and disposition before
we close the remaining performance gate or merge PR246.** No runtime changes,
merge, or Stage 5 implementation are included here.

This note summarizes the Perlmutter agent's report, retrieved from its completed
campaign. The tables below are reported measurements, not an independent rerun.

## A/B identity and existing batched coverage

| Arm | Revision | Meaning |
| --- | --- | --- |
| A | `8cf05e86a7511f84a24f82b9c9720bc2308c471d` | Updated unfixed target, including final PR187 and PR208 |
| B | `405f868ed380c78b6009da3001f4535cb9ae647b` | Reviewed Stages 1–4 alignment fix, including review corrections |
| Harness | `b52512621744c07882a55f1a3ade339a3ecb642a` | Identical field/access-retirement scripts and fixture in both arms |

This compares the cumulative alignment fix, not the dependency merge alone.
Aligned fields support A/B; unaligned fields run only on B because A's typed
views would be unsafe.

We already measured aligned batching in the accepted SDF screen: real read,
parse and dense gathering, batch20, 4BD/1GPU, bulk off/on. Job42255812 measured
mean loop increases of **1.32% off / 1.95% on**, accepted by Mona. That workload
has no user-supplied GpuTask or calibration kernel, but it does exercise batched
input preparation where the gather alignment fix operates. The field benchmark
fills a separate per-event API gap; it does not imply missing batched-gather
coverage or require repeating that accepted screen. See the
[cold report](issue159_pr246_cold_20261028.md).

## Completed Perlmutter campaign

Job **60467642**, exclusive A100-SXM4-40GB node **nid001024**, completed **0:0**
in **15m57s** (19:51:57–20:07:54 UTC, October28). One physical GPU and CPU affinity
`[0]` were used throughout. Four process pairs ran AB, BA, AB, BA. Separate native
builds used the same dependencies and release/no-LTO settings. Source trees,
installed runtimes and measurement scripts were not patched or relocated.

The report records A 48 / B 377 passing CPU tests; both retirement preflights and
access warmups passed. All 23 GPU origin records agreed on host/affinity/device.
The 28,719-file manifest passed before and after the successful allocation.
Earlier allocations failed before timing due to Lustre manifest reads and an
origin-startup timeout. Corrections affected launch/diagnostic scripts only:
full hashes moved to login-node checks before/after, and startup timeout rose
from 90 to 600 seconds. Failed attempts are preserved and excluded from results.

Python3.11.14, NumPy2.3.5, CuPy14.0.1, KvikIO/libkvikio26.10.0, mpi4py4.1.1,
Cray MPICH9.1.0.794; native GCC13.2.1, CUDA runtime12090, driver580.178.04.
GPU clocks were not locked. Do not pool absolute timings with SDF measurements.

## Aligned access overhead

Host context entry/exit time, microseconds, mean ± sample SD of four process
medians. Each process measures 200 accesses after 20 warmups. Percentages are
means of paired B/A changes, not ratios of the aggregate means.

| Field | Access | A (µs) | B (µs) | Added mean time (µs) | Mean paired change |
| --- | --- | ---: | ---: | ---: | ---: |
| Scalar | First | 38.523 ± 2.038 | 41.850 ± 1.984 | 3.327 | +8.79% |
| Scalar | Cached | 12.199 ± 0.242 | 13.070 ± 0.206 | 0.871 | +7.18% |
| 4MiB | First | 38.756 ± 0.840 | 40.632 ± 0.404 | 1.876 | +4.87% |
| 4MiB | Cached | 12.043 ± 0.221 | 13.008 ± 0.365 | 0.964 | +8.03% |

B is slower in all 16 aligned comparisons. This is consistent local accessor
overhead, about 1–3 µs, not a measured 5–9% end-to-end throughput regression.
Four process pairs are a screen, not proof of equivalence.

## B-only unaligned access

Same aggregation; all durations in microseconds. Isolated device-copy timing
is measured separately and is not full accessor latency.

| Field | First host access | Cached host access | Isolated byte copy | Retained allocation |
| --- | ---: | ---: | ---: | ---: |
| Scalar | 71.338 ± 0.647 | 14.484 ± 0.247 | 2.603 ± 0.065 | 512 bytes |
| 4MiB | 74.266 ± 0.474 | 14.419 ± 0.234 | 8.484 ± 0.059 | 4,194,304bytes |

Aligned fields directly view their input bytes. An unaligned field is copied
once into aligned storage and cached until retirement; cache hits do not add
another materialization. Scalar budget accounting rounds the 8-byte value to
512 bytes. Borrowed `on_gpu_view()` access is not an unconditional zero-copy API.

## Delayed-consumer retirement

Materialization finishes before timing. A warmed synthetic consumer reads the
cached field after 10,000,000 clock ticks, typically about 7.12 ms. Real EventPool
begin/finish methods operate on a constructed slot; this is not full event
submission. Blocking/deferred InputWindow policies represent the ownership
behavior associated with bulk off/on, without executing the bulk reader.

| Delayed case | Host finish | State when finish returns | Subsequent drain |
| --- | ---: | --- | ---: |
| Aligned, deferred, A/B | about 12.3–12.7 µs | Consumer pending; owner retained | about 7.075 ms |
| Unaligned, deferred, B | about 7.098 ms | Consumer complete; input owner still retained | about 6.3 µs |
| Blocking windows | Wait for consumer | Consumer complete; input owner released | about 1 µs |

B's root retained-copy synchronization makes unaligned retirement wait even when
the input window is deferred. This is waiting for remaining consumer work, not
7 ms of copying or additional GPU execution. It can reduce overlap; the benchmark
immediately drains afterward and does not measure application throughput.
Earlier scalar blocking cases had longer, variable consumer durations because
fixed clock ticks do not imply fixed elapsed time. Those delay-dominated A/B
differences must not be attributed to accessor overhead.

All 1,920 delayed timed cases had pending work at finish entry. Across 3,840 timed
retirement iterations, assertions checked address alignment, exact consumer
endpoints, no premature release, zero budget charge at finish and zero owner
references after drain. Access checks full field bytes on the first warmup of
each case, not every timed access. No correctness failure was reported.

## Proposed disposition for Claude's review

Mona considers per-event `on_gpu_view()` the alternative, lower-priority API and
proposes deferring its optimization while accepted batched performance holds.
We recommend retaining this implementation and documenting its measured costs:

1. Keep alignment safety, exact values, budget accounting and consumer-lifetime
   guarantees as requirements; none are waived by performance prioritization.
2. Accept the measured small aligned-access overhead for this stage and defer
   optimization of unaligned-copy host retirement. Track the latter separately
   if accepted; no new optimization issue has been opened by this update.
3. Preserve the accepted batched input/gather screen. Do not claim coverage of a
   user-kernel/calibration workload or project field-call percentages onto it.
4. Claude should assess whether the measurements and limits support closing the
   focused performance gate under this explicit scope. The gate remains pending
   that review/acceptance; PR246 remains open and Stage 5 has not started.

The duplicate SDF field job42263681 was still pending at this update and has not
been cancelled. Its pending status is not additional performance evidence.

## Original evidence

Perlmutter root:
`/pscratch/sd/m/monarin/psana2-gpu/issue159-pr246-fields-perlmutter-20261028/attempt01`.

Read `REPORT.md` for all per-process tables; `analysis/summary.json`,
`analysis/within-process.json` and `raw/` retain detailed results. `provenance/`
contains job status, build/runtime origins, hardware, launcher correction diffs
and preserved failed attempts. `EVIDENCE_SHA256SUMS` fingerprints report outputs.

Retrieved original `REPORT.md` SHA256: `16e8c46fcc814941019e03f8f4959123800daf5e95398460d137fc8de9a53f22`.
