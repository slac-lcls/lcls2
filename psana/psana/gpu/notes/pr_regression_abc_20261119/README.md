# Three-mode instrumentation overhead

Status: **completed and recorded; overhead remains inconclusive**.
Job **61610523** completed **4h 33m 10s** on nid001045. All six preflight samples,
36 main samples and 95 scheduler rows passed. The sealed collector independently
reproduced both saved summaries. All main samples passed the two zero-residency
cache observations and NIC coverage checks. See [results and hashes](results.json)
and [scheduler accounting](job-61610523-final-accounting.txt).

Claude passed code `45eb20a4315badf240e7a114adc3d8fec6af9676` with no blockers;
133 offline tests passed. This completion update changes records only.
No further overhead jobs are planned for this task; no qualification flag changes.

Scope: **GPU events / one BD / one GPU**, same private run-51 input and native
psana runtime as the [earlier two-mode study](../pr_regression_overhead_20261118/README.md).

| Mode | Added observer work |
| --- | --- |
| A / baseline | Boundary timers/checks only; no per-event or callback observation |
| B / minimal | Event and callback counts |
| C / full | Event identity/step/byte records, delivery/reader wrappers, callback sizes |

Native psana counters remain active in all modes. A uses the same declared GPU
input and gather, with an empty callback and an event loop that only advances.
A's event/callback counts, identities and bytes are explicitly unavailable.
It is not correctness evidence and is not completely uninstrumented production.

| Comparison | Mean paired change | 95% interval | +/-5% screen |
| --- | ---: | ---: | --- |
| B/A: minimal versus baseline | +8.40% | [-3.06%, +19.86%] | Inconclusive |
| C/A: full versus baseline | +6.90% | [-8.32%, +22.11%] | Inconclusive |
| C/B: full versus minimal | -1.00% | [-11.63%, +9.62%] | Inconclusive |

| Mode | Mean workload time | Mean native-process wall time |
| --- | ---: | ---: |
| A | 349.97 s | 397.41 s |
| B | 372.67 s | 434.72 s |
| C | 364.43 s | 415.69 s |

Percent changes are averaged within-round ratios, not ratios of the mean times
in the second table. None of the three intervals establishes +/-5% equivalence;
none establishes an overhead above 5% either. Timing variability prevents a
precise instrumentation-cost conclusion. Do not subtract these point estimates
from sweep results or treat the negative C/B mean as a speedup.

For a sweep, using the same observer implementation and settings across cases
provides a consistent measurement method. It does **not** establish equal cost
or cancellation of that cost across configurations: BD count, event/batch rate,
read patterns and scheduling can change the observer's relative contribution.
Report the sweep as observed performance under the stated instrumentation,
with uncertainty, rather than inferred uninstrumented production throughput.
For a PR baseline/candidate comparison, hold both workload parameters and the
observer source/settings fixed; changes in runtime behavior may still interact
with the observer. Component bounds, qualification receipts and clean A/A remain
separate prerequisites for formal PR acceptance. This note waives no gate.

Twelve rounds: **ABC BCA CAB CBA ACB BAC BAC ACB CBA CAB BCA ABC**.
Each order occurs twice; each mode appears four times in each position. Record
36 measured samples, 10k requested events and a separate 200-event warmup each.
A six-sample 1k preflight (ABC, BCA) must pass first in the same allocation.
Neither preflight samples nor prior two-mode samples enter the main statistics.

Report raw loop and native-process wall times for every round/mode. Compute
B/A - 1, C/A - 1 and C/B - 1 within each round, expressed as percentages, with
paired Student-t 95% intervals. The +/-5% screen uses loop time and requires all
three intervals inside the band. Setup and process intervals are descriptive.
Process wall time includes native launch/setup/cache checks/cleanup, but not
separate warmup, prefix/site preparation or scheduler-confirmation latency.
All modes' common boundary work and native counter costs remain outside the
incremental observer-overhead estimate. No overall campaign flag is enabled.

Prior two-mode observations were noisy, including a large late-run timing shift.
The fixed balanced schedule reduces order imbalance; it cannot guarantee that
12 rounds will establish equivalence. Retain every sample and investigate
unresolved variability instead of repeating until a pass.

See [scope](scope.md), [batch script](abc-01.sbatch), and
[maintained driver instructions](../../scripts/performance/regression/OVERHEAD.md).

[Review](review-01.json), [response](response-01.md), [validation](validation-01.json),
[frozen source association](production-01.json). The coordinator also recomputed
the earlier pilot, full and retry sealed captures unchanged (24/72/12 samples).

The job completed within its six-hour limit. Both complete sealed captures and
all samples are retained; no exclusion, replacement or resubmission occurred.

[Submission receipt](submission.json): one `lcls_g` premium allocation, one GPU,
six-hour limit. Preflight output:
`/pscratch/sd/m/monarin/psana2-gpu/pr-regression-abc-20261119/preflight-61610523`.
Main output:
`/pscratch/sd/m/monarin/psana2-gpu/pr-regression-abc-20261119/main-61610523`.
