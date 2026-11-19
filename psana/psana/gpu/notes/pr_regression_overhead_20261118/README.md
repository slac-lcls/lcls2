# Raw runner instrumentation overhead

Status: **full job 61542523 completed; revised +/-5% screen passes five cases,
with GPU events / 1 BD still inconclusive after its single fixed retry**. All 72 samples and all 151
Slurm accounting rows passed. The sealed collector reproduced the original
summary. Runtime was **8h 56m** on nid001948. The original +/-1% qualification
remains false in its immutable capture. Production MPI cleanup remains deferred.

On 2026-11-19 Mona changed the tolerance to +/-5% after seeing the full results.
The separate [screen](screen-05percent.json) preserves the original decision;
[full evidence](full-evidence.json) and [scheduler accounting](job-61542523-final-accounting.txt)
record the completed job. Positive changes mean full accounting was slower.

| Case | Mean change | 95% interval | Revised +/-5% screen |
| --- | ---: | ---: | --- |
| CPU events, 1 BD | +0.03% | [-1.46%, +1.53%] | Pass |
| CPU events, 2 BDs | +0.49% | [+0.21%, +0.77%] | Pass |
| CPU steps, 1 BD | +0.00% | [-1.70%, +1.71%] | Pass |
| CPU steps, 2 BDs | +0.45% | [-0.33%, +1.22%] | Pass |
| GPU events, 1 BD | -5.43% | [-22.13%, +11.28%] | Inconclusive; see retry below |
| GPU events, 2 BDs | -0.42% | [-3.76%, +2.91%] | Pass |

The whole interval must fit within +/-5%. The one-BD GPU mean is not evidence of
a speedup: its individual pair differences range from -23.29% to +17.38%.
The [retry script](retry-01.sbatch) ran six fresh pairs for that case, with 10k
events and separate 200-event warmup per sample. **Retry job 61586044 completed
in 1h 11m 32s on nid001745**. All 12 samples and 31 scheduler accounting rows
passed. An independent sealed-collector audit reproduced the saved result;
every measured sample had two zero-residency cache observations and passed NIC
coverage checks. See [retry evidence](retry-evidence.json), [scheduler accounting](job-61586044-final-accounting.txt)
and the [single-submission receipt](retry-submission.json).

| GPU events, 1 BD | Mean change | 95% interval | +/-5% screen |
| --- | ---: | ---: | --- |
| Original | -5.43% | [-22.13%, +11.28%] | Inconclusive |
| Retry | +2.94% | [-11.10%, +16.99%] | Inconclusive |

The first eight retry samples took 180–209 seconds; the last four took 329–413
seconds, affecting both observer modes. The measured full-versus-minimal timing
difference varies substantially. **These runs do not isolate that variability
as a changing instrumentation cost**: workload/I/O/execution variability remains
mixed into the estimate. Five cases pass the revised screen; GPU events / 1 BD
remains unresolved. No further retry was submitted and no qualification flags
were enabled. Investigate variability before further measurement.

Both minimal and full observation modes are implemented by the regression
runner in [native_worker.py](../../scripts/performance/regression/native_worker.py),
not production DataSource options. Full mode installs temporary delivery and
reader-accounting wrappers and records per-event timestamps/steps/CPU bytes and
per-callback sizes. Minimal mode omits that added detail but still counts events
and callbacks. Both retain runner timing, cache conditioning, NIC sampling,
placement and provenance checks. These additions are active when this benchmark
worker runs; ordinary production applications do not acquire them automatically.

The pinned production psana runtime already maintains its own counters/timers,
including KvikioGpuReader read bytes, request counts and I/O timing. Those remain
active in both modes. Thus this experiment measures the incremental cost of
full runner observation over minimal runner observation; it does not measure
all instrumentation cost against an uninstrumented production baseline.

Claude round 3 passed controller `a5769a41ebd41703bc60881d6dd4d9258ce1116e`
with no blockers; 126 offline tests passed. No executable code changed for this
result/interpretation update. Exactly one retry was submitted from full-61542523;
both attempts remain preserved. Retry output:
`/pscratch/sd/m/monarin/psana2-gpu/pr-regression-overhead-20261118/retry-61586044`.

Original code review: Claude round 2 passed
`a876e8275c42cfd272d52e5e283f0fe99852dde0` with no blockers and 120 offline tests.

[Scope](scope.md), [driver instructions](../../scripts/performance/regression/OVERHEAD.md),
[current review](review-03.json), [response](response-03.md),
[current validation](validation-05percent.json), [retry source association](production-02.json).
The original [review](review-02.json), [validation](validation-03.json) and
[source association](production-01.json) remain preserved.

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

The completed original full qualification used all six cases, 10,000 measured
events, 200 warmup events and six balanced adjacent pairs per case (72 samples).
Its predeclared margin was +/-1%, which only CPU events / 2 BDs met. The original
[full script](full-01.sbatch) pins the successful pilot's completion hash.
Full output: `/pscratch/sd/m/monarin/psana2-gpu/pr-regression-overhead-20261118/full-61542523`.
Neither the original run nor the revised screen enables a campaign acceptance flag.

Workload parity was also traced in the installed runtime: CPU EventManager fills
bigdata buffers and constructs Dgrams before yielding an EventEnvelope; Event
stores that list. GPU EventPool.submit calls each declared input preparer's
prepare_batch before dispatch_task; prepare_batch queues dense gather. The
observer toggle therefore retains these reads and gathers. Native psana counters
and baseline count/boundary costs remain common and unmeasured.
