# Issue 159: tracing the historical cold-rate drop

The absolute rate is substantially lower: the documented September 4BD/1GPU
baseline was **341.64/351.64 events/s** (bulk off/on), while the October 22
Stage 4 screen was **177.46/177.42 events/s** using the same median-loop metric.
That is a 48.1%/49.5% throughput reduction across campaigns. The records do
**not** identify a commit that caused it. Several rate reductions already
occurred with unchanged runtime source, on different nodes/dates.

The Stage 4-versus-Stage 3 result remains valid as an incremental comparison.
It does not establish preservation of the historical 350 Hz baseline or rule
out cumulative effects before Stage 4. This broader question remains open.

## When lower rates appear in the records

All rows below are 10,000 events, JF-only dense raw gathering, 4 BDs/1 GPU,
no calibration/user kernel, CPU fallback. Rates are `10000 / median(loop_s)`
to match `read_staging.md`; the latest mean/std report uses a slightly different
aggregation and is not being silently substituted here.

| Date | Job / node | Revision | Cold off (events/s) | Cold on (events/s) |
| --- | --- | --- | ---: | ---: |
| Sep 28 | 39369005 / sdfampere035 | September `10df4c6e3` | 341.64 | 351.64 |
| Oct 16 | 40882509 / sdfampere012 | Post-stack `410ebba1c` | 391.32 | 385.84 |
| Oct 17 | 41173787 / sdfampere033 | Same post-stack `410ebba1c` | 308.50 | 299.69 |
| Oct 19 | 41310953 / sdfampere030 | Pre-PR187 `c157b490c` | 260.95 | 261.16 |
| Oct 19 | Same job/node | Integrated PR187 `90c05965c` | 258.31 | 264.39 |
| Oct 22 | 41604951 / sdfampere026 | Stage 3 `f425e601e` | 177.34 | 177.41 |
| Oct 22 | Same job/node | Stage 4 `ef0a453d4` | 177.46 | 177.42 |

The first lower-than-September cold result among these recorded campaigns is
October 17; the October 16 run was actually faster than September. A further
drop appears October 19, and the first measured ~177 Hz point here is October
22. These dates mark observations, not dates when a software defect began.

`410ebba1c` is present in both the ~390 Hz and ~300 Hz runs. Its successor
`c157b490c` changes only two documentation files, yet ran at ~261 Hz. Therefore
the fall to ~261 Hz is already visible without an intervening processing-source
change and before PR187 or issue159 fixes. Build/runtime/environment and
node/storage variability cannot be ignored.

The October 19 same-node comparison measured PR187's incremental paired changes
at +1.35% cold/off and -0.99% cold/on; it did not show a ~50% collapse. The
October 22 same-node comparison likewise shows Stage 4 did not introduce the
~177 Hz level: Stage 3 already has it. The remaining ~261-to-177 gap is about
32% in rate and spans different nodes, runtime/build revisions and harness
changes. There is no matched cold bridge for that interval. Stages 2–3 had
warm screens, so those runs cannot date a cold-specific regression.

## Workload and dependency audit

All five campaigns have identical staged file hashes and identical parsed
10k-event reference records:

- Five private files: **335,656,487,644 staged bytes**.
- Timed payload: **335,571,760,000 bytes**, **50,000 requests**, 10,000 events.
- 32 JF panels, batch 20, execution depth 1, eight KvikIO workers per BD,
  1 MiB task/group target, physical GPU 0, XFS `/lscratch` on `/dev/md0`.
- Same event-loop timing definition; explicit setup and staging excluded.

There is no doubling of payload/request count or switch to calibration in the
177 Hz result. Both bulk settings report 50,000 requests for this large-dgram
workload; the bulk label does not imply fewer requests in this dataset.

September used KvikIO 24.08.02 / CuPy 13.6.0. All four October campaigns used
official KvikIO 26.10.00 / CuPy 14.0.1, including the ~390 Hz result. The
library upgrade alone therefore does not explain why all new runs must be
~177 Hz. October 16 used shared native build artifacts; later comparisons used
independent installations. These are additional cross-campaign differences.

The budget changed from automatic ~9.87 GiB/BD to explicit 6 GiB/BD for the
PR187 comparison. Peak tracked allocations stayed ~1.252 GiB/BD throughout;
there is no recorded capacity exhaustion. This is not a controlled budget
A/B, so the budget change is retained in the evidence rather than dismissed.

The current harness also adds a zero-shift alignment allocation wrapper
(request N+1 bytes, expose N), stronger pre-loop import provenance, and explicit
MPI `pml=ob1`. Timed address audits are disabled. The cold helper itself is
unchanged between October 19 and October 22: the new private-file wrapper
validates ownership then calls the same `cache_inputs(..., 'cold', ranges=...)`
with fsync, DONTNEED and per-file residency checks. It runs outside timing.
Both future comparison arms must use the same harness/environment to avoid
confounding runtime and harness changes.

## Where the additional time appears

The bulk-off counters show the increase almost entirely in read-future waits:

| Run / selected variant | Median loop (s) | Median maximum BD read wait (s) | Effective payload GB/s |
| --- | ---: | ---: | ---: |
| Sep 28 baseline | 29.271 | 27.092 | 11.46 |
| Oct 16 post-stack | 25.554 | 23.535 | 13.13 |
| Oct 17 same post-stack | 32.414 | 30.437 | 10.35 |
| Oct 19 pre-PR187 | 38.322 | 36.400 | 8.76 |
| Oct 22 Stage 3 | 56.389 | 54.482 | 5.95 |
| Oct 22 Stage 4 | 56.351 | 54.461 | 5.96 |

`read_wait_s` accumulates time around KvikIO future completion in
`KvikioGpuReader.wait_batch()`. These are maximum-rank aggregates, not a precise
serial decomposition of the critical path. Nevertheless, non-wait loop time
remains roughly two seconds while read waits expand by tens of seconds.

**Inference:** investigate the read-completion path first. Possible contributors
include local disk service, page-cache/reclaim behavior, CPU scheduling, and
CPU-fallback transfers. The counters do not isolate physical disk service and
do not prove a throttle, disk defect, or NUMA cause. The very tight latest rates
near 5.95 GB/s are consistent with a common throughput limit, not proof of one.
The source read/controller/window modules are unchanged between integrated
`90c05965c` and Stage 3, while gather and lease handling did change.

Historical September provenance includes a two-device RAID0 `/lscratch`.
Latest screen provenance records XFS/md0 but lacks disk models, RAID membership,
I/O limits and read-ahead settings. An attempted read-only inspection of
sdfampere026 was denied by `pam_slurm_adopt` because the completed allocation
is no longer active; no bypass was attempted. Present hardware inspection
would also not reconstruct the exact historical load/state.

## Next controlled comparison

Before calling this a particular commit regression, run the saved pre-PR187
`c157b490c` installation against current Stage 4 `ef0a453d4` in one allocation,
with the **same current harness**, explicit 6 GiB budget, dependencies, MPI
settings, GPU/CPU placement, private input hashes and per-sample cold gates.
Use four balanced pairs per bulk condition and retain A/B mean/std reporting.
This tests different revisions; no additional A/A campaign is proposed.

Capture block-device/RAID topology, per-device `/proc/diskstats`, applicable
cgroup `io.max`/`io.stat`, read-ahead settings, NUMA/CPU affinity and GPU clocks
around samples, outside loop timing. Do not change host I/O controls. Reuse
existing frozen builds at their original paths; do not silently rebuild or
relabel prior results.

- If old and current code both run near 177 Hz, the historical absolute loss is
  reproduced without requiring the newer runtime; investigate I/O/environment
  and, if needed, replay the same frozen old revision on a previously faster node.
- If the old revision is substantially faster on the same node, isolate the
  cumulative change using the integrated pre-fix baseline and then Stage 2/3.
- A result near 350 Hz for both would demonstrate node/time dependence; it would
  not retroactively make the 177 Hz run invalid.

No new job was submitted during this historical audit. The current Stage 4
screen is complete, but historical throughput preservation remains unresolved.

Sources: [read/staging report](../docs/performance/read_staging.md),
[PR187 comparison](pr187_staging_comparison_20261019.md),
[Stage 3 warm screen](issue159_stage3_cache_reset_retry_20261020.md),
[Stage 4 cold screen](issue159_stage4_cold_20261022.md), and
[compact historical evidence](evidence/issue159_cold_rate_history_20261022.json).
