# Disposable cleanup qualification — 2026-11-18

Five cases passed on Perlmutter job **61502386**, nid001040. The allocation
COMPLETED/0:0 in **2:30**. Each intended target was CANCELLED/0:9; every separate
sentinel continued after target cleanup, was released gracefully and finished
COMPLETED/0:0. Final direct scontrol query returned “Job step 61502386. not found”.
Independent final review is pending; profile qualification flags remain false.

| Case | Target / sentinel steps | Observed exit timing | Sentinel |
|---|---|---|---|
| step-timeout | 61502386.0 / 61502386.1 | 63.577 s total after 60 s deadline | COMPLETED/0:0 |
| SIGTERM | 61502386.2 / 61502386.3 | 0.917 s from signal to adapter exit | COMPLETED/0:0 |
| SIGINT | 61502386.4 / 61502386.5 | 0.867 s from signal to adapter exit | COMPLETED/0:0 |
| SIGHUP | 61502386.6 / 61502386.7 | 0.766 s from signal to adapter exit | COMPLETED/0:0 |
| controller-timeout | 61502386.8 / 61502386.9 | 62.670 s controller total after 60 s deadline | COMPLETED/0:0 |

These are launcher timings for sleeping CPU ranks, not psana throughput or GPU
teardown measurements. The sentinel deliberately overlaps resources and has a
name extending the target name; exact-name cleanup must preserve it. Three target
ranks bind four cores each, reserve 16 physical CPUs each; sentinel reserves one
CPU. Requested allocation: 64 CPUs, one GPU, 16 GiB. Actual allocation: 128 logical
CPUs (64 physical), four A100s, 16 GiB. No GPU kernels, experiment reads, cache
operations or calibration requests. No unit-test correctness claims are replaced.

The exercise found and fixed two launcher issues:

- squeue omitted live numeric steps in the diagnostic while direct scontrol
  reported them. Matching now uses direct queries, exact names and strict parsing;
  malformed, duplicate, foreign or empty responses fail closed.
- A successful srun can exit before its controller step record disappears. Normal
  launch completion waits up to six seconds without cancelling the finished step.
  Fault cleanup still has one six-second scheduler deadline, including cancellation.

[Failed attempts and diagnostic](failed-attempt.md) retain jobs 61502056 and
61502208, both correctly rejected; diagnostic 61502138 completed normally. All
allocations ended and final step audits are saved. The underlying reason for the
squeue/scontrol discrepancy is not established. Historical placement sacct success
is unaffected, but old squeue absence observations are not cleanup qualification.

[hardware-evidence.json](hardware-evidence.json) pins the production commit,
source/runtime hashes, exact failure receipts, case results and indexed evidence.
Coordinator verified all 109 indexed files, the index hash, runtime manifest hash,
and all five executed sources against production commit `bf4809aec2233a9f174ff752a4382429f108147b`.
Raw artifacts remain at `/pscratch/sd/m/monarin/psana2-gpu/pr-regression-cleanup-20261118`.
Offline final suite: **93 tests PASS** (53.767 s); full log path/hash in evidence.
No new code was edited after this full-suite run or successful hardware execution.

Use [CLEANUP.md](../../scripts/performance/regression/CLEANUP.md) for future agent
instructions. [qualify-03.sbatch](qualify-03.sbatch) is the successful frozen-source
launch; previous scripts remain as historical evidence. Review uses a detached
exact-commit worktree and restricted read tools; scratch verification is performed
by the coordinator, not independently rerun by Claude.

Remaining: six native run51 workload smoke cases, cache/accounting qualification,
instrumentation overhead and component bounds, then timed A/A before PR A/B.
This result does not enable default qualification flags or launch a timed campaign.
SIGKILL, node loss, scheduler outages, filesystem stalls, arbitrary external
interrupts and CUDA/MPI teardown under workload remain outside this disposable
CPU qualification. Each new native allocation must repeat placement preparation.
