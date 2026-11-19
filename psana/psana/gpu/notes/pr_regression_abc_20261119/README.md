# Three-mode instrumentation overhead

Status: implementation and offline validation in progress; no job submitted.
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
