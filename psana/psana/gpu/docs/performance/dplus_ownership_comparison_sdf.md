# D+ ownership comparison: two-round closeout

2026-09-23. Job **38898399**, `sdfampere038`, one A100 and one BD.
The user requested closure after the first two complete warm/cold rounds.
The audited report uses 12 clean samples; later partial-round samples remain
in the artifacts but do not enter the comparison. Remaining repetitions and
trailing timestamp profiles were waived. Three CPU-reference preflights and
all three full-window pixel/profile diagnostics passed before clean timing.

## Workload and frozen implementations

JF only: `mfx101210926`, run 387, streams 5–9, 10,000 events, batch 20,
depth 1, 8 GiB budget, no automatic/user D2H in clean timing. KvikIO CPU
fallback, eight I/O threads, 1 MiB tasks. Fresh MPI processes, 100-event
compilation warmup, fixed allocation CPU mask, alternating cache order.
Warm page-cache residency was at least 99% before/after each sample; cold
residency was at most 1% before each sample, on private local NVMe input.
Cold does not mean remote-storage or controller-cache cold.

- D: original bulk implementation, `8f94e3c7bc`, bulk on.
- C+: reviewed D+ ownership fixes, bulk off.
- D+: the same ownership-fixed installation, bulk on; runtime corresponding
  to ownership commit `786cd16ca`. B++ is not included in these variants.

Frozen build hashes, native identity, timestamps, event/input counts, cache
conditions, and CPU/GPU placement passed the final audit.

## Clean results

Rates below are 10,000 divided by the median loop time, not the median of
individual rates. Timing includes GPU-manager setup and final GPU sync;
staging, cache preparation, and post-loop teardown are excluded.

| Variant | Cold median seconds | Cold events/s | Warm median seconds | Warm events/s |
|---|---:|---:|---:|---:|
| D | 100.694 | 99.3 | 48.725 | 205.2 |
| C+ | 65.525 | 152.6 | 43.467 | 230.1 |
| D+ | 98.275 | 101.8 | 48.540 | 206.0 |

Paired throughput changes, pairing by repetition/cache condition:

| Comparison | Cold | Warm |
|---|---|---|
| D to D+ | mean of two changes +2.46%; range +0.04% to +4.89% | +0.35%; range -1.93% to +2.63% |
| C+ to D+ | -33.32%; range -33.66% to -32.99% | -10.45%; range -10.80% to -10.11% |

No consistent D+ throughput regression is observed relative to D in these two
rounds. This supports proceeding with integration; it does not establish a
precise speedup or statistical equivalence. Two rounds do not balance all
variant positions. Bulk on remains slower than bulk off in both cache modes.

## Memory and I/O diagnostics

| Variant | Ledger + held peak MiB | CuPy used peak MiB | CuPy used at loop end MiB | Read requests |
|---|---:|---:|---:|---:|
| D | 3,201.8 | 8,325.5 | 5,123.7 | 2,885 |
| C+ | 3,201.9 | 3,201.9 | 0 | 50,000 |
| D+ | 3,201.9 | 3,201.9 | 0 | 2,885 |

The ownership fix removes the unexplained live-memory excess in this workload
while preserving bulk coalescing. Fewer read requests alone do not improve
throughput. These are instrumented diagnostic measurements; their timings
are excluded from the clean table. CuPy used memory, allocator cache, and
sampled device residency are distinct quantities.

## Artifacts and next comparison

`validation/dplus-performance-20260923/job-38898399/summary.md` and
`summary.json` contain the audited first-two-round result. `results.json`
preserves every accepted sample. Reproduce the closeout with:

```
python validation/dplus-performance-20260923/finalize_two_rounds.py \
  validation/dplus-performance-20260923/job-38898399
```

Next: one fresh allocation, two warm/cold rounds, D+ versus integrated bulk
off/on at the same JF settings. The integrated runtime is `ac87a93b2`, validated
through Stage 3 without further production changes. Drain/trim and overlap
costs remain profiling questions; true GDS and multi-BD scaling are not covered.
