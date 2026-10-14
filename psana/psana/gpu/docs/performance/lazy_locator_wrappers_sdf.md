# Warm A / B+gather / on-demand locator wrappers on SDF

Job **38676923**, **sdfampere034**, 2026-09-20. One A100 allocation;
**COMPLETED / 0:0**, elapsed **55:56**. All **18 clean samples**, three
CPU-reference preflights, and 33 locator/gather integration tests passed.

## Result

10,000 events per sample; six samples per variant:

| Variant | Median elapsed (s) | Range (s) | Events/s from median |
|---|---:|---:|---:|
| A | 22.464 | 22.394–26.626 | 445.1 |
| B+gather | 25.288 | 24.652–28.037 | 395.4 |
| B+gather zero wrappers | 24.824 | 24.236–25.637 | 402.8 |

Removing eager wrappers reduced the observed median from **25.288431 s** to
**24.824160 s**: **0.464271 s / 1.84% less elapsed time** (1.87% more throughput).
The zero-wrapper build was faster in **four of six rounds**, but slower in
rounds 1 and 6. Its range overlaps the eager-wrapper range. This is a modest
observed median improvement; these measurements do not establish a stable
1.84% speedup or isolate wrapper construction time.

Compared with A, the eager-wrapper median is **2.824043 s / 12.57% longer**;
the zero-wrapper median is **2.359772 s / 10.50% longer**. Thus the gap remains.
No samples from previous nodes or allocations are pooled into these values.

## Every clean sample

A = legacy addressing; G = B+gather with eager wrappers; Z = B+gather with
on-demand wrappers. Positive G−Z means zero wrappers was faster in that round.

| Round | Order | A (s) | G (s) | Z (s) | G−Z (s) |
|---:|---|---:|---:|---:|---:|
| 1 | AGZ | 22.416551 | 24.651930 | 25.377425 | -0.725495 |
| 2 | GZA | 26.625698 | 26.034585 | 24.312729 | +1.721856 |
| 3 | ZAG | 22.738005 | 24.736637 | 24.251229 | +0.485408 |
| 4 | ZGA | 22.512225 | 28.037219 | 25.636917 | +2.400302 |
| 5 | GAZ | 22.393823 | 25.718105 | 24.236390 | +1.481716 |
| 6 | AZG | 22.415886 | 24.858756 | 25.335591 | -0.476835 |

All six permutations were used. Every variant appears twice in every position;
each ordered adjacent pair within a round appears twice. Rounds are sequential,
not simultaneous, so node variation remains possible despite the matched
allocation. All samples, including slower ones, are reported.

## Build isolation

- A: frozen `f52e90cc66d4c8c175b7689922c7e441e78b367f` installation from the
  previous four-way comparison.
- G: frozen B+gather installation from job 38564718, based on B+ commit
  `b0c9c3c029beb0d54c8fecd3256da3fa8898be25` plus the validated gather changes.
- Z: an independent copy of G with only `gpudgram/parser.py` replaced. AST
  comparison limits the changes to `_locate_configured` and `locate`: remove
  eager wrapper construction, then create/cache configured views on first access.

A and G each matched all 25 installed-source/native-extension hashes recorded
by job 38564718. All installed hashes remained unchanged through this run.
G/Z differ in exactly one source file; CUDA kernels are unchanged. Unrelated
memory-reporting cleanup in the working tree is excluded from both G and Z.
The exact isolated change is recorded in `G-to-Z.patch` alongside the harness.
Z parser SHA256: `e3c2b143d5b30360d33c6b24ff2fb3126e18121c5256a9cff4364abc3142fddb`.

Canonical gathering accesses combined locator storage directly. The integration
tests verify zero wrappers after parsing and after canonical gathering. The
clean timing runs use no wrapper counters, phase hooks, cProfile, or Nsight;
they do not supply a separate measurement of wrapper-construction cost.

## Matched workload and placement

- `mfx101210926`, run 387, 10,000 events, 32 Jungfrau segments across five
  physical XTC files, streams s005–s009. Payload **335,571,760,000 bytes**.
- GPU execution batch **20**, depth **1**, GPU budget **8 GiB**; SMD batch
  environment `PS_SMD_N_EVENTS=1000`. No bulk-read integration or user D2H.
- Three MPI ranks: one SMD0, one EB, one BD. One A100-SXM4-40GB, 48 requested
  CPUs, 450 GiB requested host memory. CPU affinity is identical for all ranks
  and variants; no exclusive per-rank or GPU-local NUMA binding was introduced.
- GPU UUID: `GPU-73c5f641-1fc6-4433-4038-13df4ea1e353`.
- CPU mask: `8-19,40-41,43-52,72-83,104-105,107-116`.
- CuPy **13.6.0**, CUDA runtime **12090**, KvikIO **24.08.02**, NVIDIA driver
  **575.57.08**. KvikIO compatibility **ON**, eight threads, 1 MiB tasks;
  CPU fallback, not GDS.
- Sampled peak device memory: A **3631 MiB**, G **3633 MiB**, Z **3633 MiB**
  in every clean run (500 ms sampling; not an allocation-ledger maximum).
- All variants use the same frozen CPU-reference calibration snapshot.
  Calibration SHA256: `c3679e0fb47bebfafe41f36750e88fe1ec0d6b71ef386814fb1eac37dc197ce9`.
- Timestamp SHA256 for every sample:
  `23f9051ac9f98208310171f734251b0ccd1f40fa048d1748553cbd38cce559d6`.

The event-loop timer matches the previous comparison: detector setup and final
GPU synchronization are included. Each measured sample follows a separate
100-event warmup. Staging, cache preparation, and placement logging are outside
the timer. The benchmark AST matches the prior harness except variant choices.

All **36 before/after cache checks were 100%**. Before the first A measurement,
the cache warmer needed three passes (88.31%, 95.55%, then 100%). Those retries
were outside timing; no partially warm sample was accepted. Expected KvikIO
CPU-fallback, expired Kerberos-ticket, and pytest-asyncio warnings were logged;
the frozen calibration preflights and all measured cases completed successfully.

## Reproduction and artifacts

Harness: `validation/lazy-wrapper-perf-20260920/`.
Job artifacts: `validation/lazy-wrapper-perf-20260920/job-38676923-warm-ab/`.

```bash
source setup_env.sh
source install_psana/activate.sh
sbatch validation/lazy-wrapper-perf-20260920/run.sbatch
python validation/lazy-wrapper-perf-20260920/audit.py validation/lazy-wrapper-perf-20260920/job-38676923-warm-ab
python validation/lazy-wrapper-perf-20260920/summarize.py validation/lazy-wrapper-perf-20260920/job-38676923-warm-ab
```

`results.json` contains all samples; `summary.json` contains exact medians and
comparisons; `audit.json` verifies workload, ordering, cache, placement, runtime,
CPU-reference checks, and unchanged installed hashes. `provenance.json` records
builds, source/native hashes, environment, CPU affinity, and the working-tree
diff. Per-case logs, GPU CSV samples, and host snapshots are retained.
Eight harness tests and benchmark/order equivalence checks passed before launch.
No production code, archived benchmark build, commit, push, or bulk integration
was changed by this performance run.
