# Raw runner instrumentation overhead

Status: **pilot passed; full 10k measurement submitted (job 61542523)**. Overhead
qualification remains pending. Claude round 2 passed code
`a876e8275c42cfd272d52e5e283f0fe99852dde0` with no blockers; all 120 offline tests
passed on that exact commit. Production MPI cleanup remains deferred.

[Scope](scope.md), [driver instructions](../../scripts/performance/regression/OVERHEAD.md),
[review](review-02.json), [response](response-02.md), [validation](validation-03.json),
[frozen source association](production-01.json).

Pilot job **61541892**, nid001400: all 24 measured samples passed, each with 1,000
events and a separate 200-event warmup. It finished in **17m 36s**. All 55 Slurm
accounting rows completed 0:0. The sealed collector was rerun independently and
matched the saved summary. Per-rank checks confirmed four distinct physical
cores each and the declared observer modes with no profiler/periodic stack trace.
See [audited evidence](pilot-evidence.json), [scheduler accounting](job-61541892-final-accounting.txt)
and [pilot script](pilot-01.sbatch). No sample was dropped or retried.

The pilot's two pairs per case are mechanics checks, **not overhead qualification**.
Positive changes mean full accounting was slower than minimal observation.

| Case | Mean change | 95% interval | Pilot interpretation |
| --- | ---: | ---: | --- |
| cpu-raw-events-bd1 | +0.022% | [-1.387%, +1.431%] | Inconclusive |
| gpu-raw-events-bd1 | +0.083% | [-2.309%, +2.475%] | Inconclusive |
| cpu-raw-events-bd2 | -0.107% | [-0.956%, +0.743%] | Inside margin; pilot only |
| gpu-raw-events-bd2 | -0.546% | [-0.961%, -0.132%] | Inside margin; pilot only |
| cpu-raw-steps-bd1 | -0.409% | [-1.507%, +0.688%] | Inconclusive |
| cpu-raw-steps-bd2 | +0.400% | [-2.675%, +3.475%] | Inconclusive |

Full qualification uses all six cases, 10,000 measured events, 200 warmup events
and six balanced adjacent pairs per case (72 samples). Every loop-time interval
must fit within +/-1%. The [full script](full-01.sbatch) has a 12-hour limit and
pins the successful pilot's completion hash; source and input hashes must match.
Expected duration is roughly 8–10 hours from the previous full-prefix accounting
run. Full output: `/pscratch/sd/m/monarin/psana2-gpu/pr-regression-overhead-20261118/full-61542523`.
No acceptance flag is enabled by this stage.

Workload parity was also traced in the installed runtime: CPU EventManager fills
bigdata buffers and constructs Dgrams before yielding an EventEnvelope; Event
stores that list. GPU EventPool.submit calls each declared input preparer's
prepare_batch before dispatch_task; prepare_batch queues dense gather. The
observer toggle therefore retains these reads and gathers. Native psana counters
and baseline count/boundary costs remain common and unmeasured.
