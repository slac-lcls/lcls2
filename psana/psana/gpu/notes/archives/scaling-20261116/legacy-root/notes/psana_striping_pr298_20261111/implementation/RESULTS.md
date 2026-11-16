# PR298 stripe-read pilot

Job **61133510** passed all 14 cases and the independent audit. Every Slurm record, including 18 numeric steps, is COMPLETED0:0. The premium lcls_g allocation ran 15m27s, from 2026-11-11 06:35:18 to 06:50:45 UTC (November 10, 10:35:18–10:50:45 PM Pacific). Node nid003140, one active A100-SXM4-40GB.

Exact PR298 source: `67c327a148a08b2aaf972e67e1e02e367c2b298f`. Executed harness bundle: `e7c61f05ab31923fadf8579223b2e74b020d5d79`. The source and installed build are unchanged; every loaded production module is hash-checked. Independent review passed on `0723c08e841dd26b5c42269ba2fe832c555ca1f3`, with no blocking findings; see [review-03.json](review-03.json).

The prepared sweep has 72 cases: stripe counts 1, 2, 4, 8, 16, 32 × CUDA streams 1, 2 × BD readers 1, 4 × 3 randomized blocks. [sweep.csv](sweep.csv) is the full parameter list. Every full case is 100,000 events across the same five JF files. Neither the full sweep nor the 100k sizing case has been submitted.

Fixed controls: bulk target 524,288 bytes, batch 100, SMD 1000, 256 MiB SMD chunks, 6 GiB per BD, KvikIO 8 threads per BD / 1 MiB tasks / 16 MiB bounce buffers, compat ON and automatic direct I/O OFF. One GPU is shared by the BD processes; helpers have no visible GPU. All ranks in all cases used the same 56 physical cores selected from both Slurm layouts, with NUMA interleave.

## Accepted short timing pilots

Each timing pilot reads 1,000 events at stripe count 2, 3.127567 GiB of useful datagrams. These are single short runs, including startup/fill/drain; they do not choose a stripe count or measure steady-state cube performance.

| BD readers | CUDA streams/BD | Read-loop seconds | GiB/s | Whole case seconds |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1.725 | 1.813 | 50.56 |
| 4 | 1 | 0.967 | 3.235 | 53.18 |
| 1 | 2 | 1.853 | 1.688 | 51.21 |
| 4 | 2 | 1.023 | 3.056 | 52.72 |

## Validation and timing scope

- 1,023 CPU unit tests, 31 native GPU tests, and 7 harness audit tests passed.
- Six 100-event binding cases verified actual DgramManager and CuFile descriptors for all stripe layouts.
- Four 1,003-event tail cases covered 1/4 BDs and 1/2 CUDA streams. Each compared 96,288 GPU seed values to CPU references across 32 panels; exact event/range/byte/callback coverage passed. Every depth-2 BD actually used both streams in the accepted run.
- All 30 files passed every 100k-event SMD/header check. Eight independent validation workers completed in 160.999 s. Process read_bytes increased by 46,846,976,000 bytes; this is a kernel/process counter, not validated Lustre wire traffic.
- All 30 payloads passed two zero-residency mincore scans plus the selected-five scan before each loop. Cache preparation took 45.06–60.82 s per case, outside the measured read loop. Backend/server cache state and remote byte attribution remain unverified.
- No CPU EventManager payload reads, pressure trims, unfinished reads, active slots, or held GPU budget remained. Host cleanup assertions and every Slurm exit passed. Raw logs retain the site module-logging tracebacks, expected CPU-fallback warning, and an MPICH builtin-communicator reference warning at shutdown; these are not hidden or presented as failed-rank evidence.

## Prior attempts and compute cost

| Job | Outcome | Node time |
|---|---|---|
|61131843|Stopped serial metadata diagnostic; no timing cases|15m08s|
|61132571|Failed first binding due to temporary proc-directory fd in snapshot; fixed and regression-tested|4m37s|
|61132790|All 14 local cases passed, but audit rejected unequal physical CPU sets across reader counts|15m30s|
|61133510|Accepted 14-case pilot and full audit|15m27s|

Total for this phase: 50m42s of node walltime. That leaves 5h09m18s within the planned 6-node-hour correctness/pilot/screen cap, excluding unrelated task use. Raw node time is not an Iris charged-hours calculation. Before every submission Iris showed personal/project balances above 20%; accounting can lag.

## Remaining before a full sweep

After separate authorization, run the 100k sizing case before choosing a full-sweep walltime. Add peak-device-memory telemetry and prepare/review the sizing/screen launcher within the remaining cap, as explicitly deferred by reviews 2 and 3. The pilot runner itself requests 45 minutes; it must not be mistaken for a sized full-screen submission. No stripe recommendation follows from this smoke test.

Raw scratch root: `/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111`. Accepted receipts, commands, Slurm accounting, source comparison and audit output are under [evidence/pilot04](evidence/pilot04). Earlier rejected attempts remain under their own directories.
