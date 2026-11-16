# Fixed-profile qualification and sizing

Implements the reviewed selection in a new harness directory; historical pilot
code and immutable bundles are unchanged. Four BDs share one GPU, each with one
stream and four KvikIO workers; batch200,bulk512KiB,task1MiB,bounce16MiB,budget6GiB.
Six ranks bind to disjoint physical cores (3,3,7,7,7,7), selected from the actual
six-rank Slurm masks and checked on every rank. The allocation still reserves a
whole node/four GPUs, with only one active; no billed-resource savings claimed.

Qualification: exact pinned31GPU integration tests, full30-file metadata/range
validation, six100-event binding cases,1003-event tail/value witness and one
100k count2 sizing case. Full cases require all four BDs to receive events.
The tail requires five200-event callbacks plus one3-event callback globally;
100k requires500callbacks of200. These expectations follow the independently
validated event set, batch200 and SMD1000 (divisible by200), with one execution
stream and no pressure subdivision allowed. Native assertions must confirm this;
unexpected splitting is a failure, not grounds to relax the expected histogram.

Memory: a coordinator NVML thread samples the selected GPU every100ms from before
process launch until after exit, without creating a CUDA context. Records include
timestamps,free/used/total and exact observed extrema. Acceptance requires samples
bracketing the loop, at least one in-loop sample, no gaps over2s, and at least4GiB
observed free memory. These are sampled peaks, which can miss shorter transients.
Independent hooks observe every successful framework budget reserve/hold under
its existing lock, including setup. Per-BD tracked maxima must fit6GiB; the sum
of individual maxima (not assumed simultaneous) must fit24GiB. Actual totalVRAM
includes contexts and caches beyond those budgets. No payload/stream lifetime
semantics changed; monitoring overhead remains part of the measured configuration.

The qualification launcher requests45min,premium,lcls_g. launch_plan.py requires
a completed one-node phase ledger including all four prior attempts, sums actual
node-seconds, and refuses a requested walltime exceeding the21600s phase cap.
It creates plans but never submits. run.sbatch/campaign validate the actual Slurm
walltime against the saved plan. Later screen planning requires an accepted
qualification audit and uses1.5x whole-case duration,1.5x measured preflight cost
and300s reserve. Screen startup checks remaining time before each six-case block.
The current phase does not submit the screen. All failed attempts remain charged
and must be added to the phase ledger before another submission.

Eleven local audit/affinity/memory/ledger tests pass; Python/shell syntax checks
pass. Exact PR CPU1023-test results are inherited from the unchanged pinned build;
required31native tests run again on the new allocation. Existing input and build
manifests stay pinned. Source file gpu_budget.py is copied under evidence for
review of the observational hooks; it is not a runtime overlay.

Prelaunch review passed at `c3a8cb653fe6374953566c2a9245ee153856272b` with no
blocking findings. [responses-01.md](responses-01.md) records bounded qualification
choices and hardening deferred before the screen. Native results acceptance is
still pending.

Qualification job **61359629** was submitted on lcls_g/premium and started on
nid001264 at2026-11-15T20:16:40UTC (12:16:40PM Pacific). Its hard45min walltime
ends21:01:40UTC (1:01:40PM Pacific). Artifacts are in
`/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/qualify01`;
the immutable `bundle-fixed01` matches the reviewed code commit. The18-case
screen has not been submitted. Consult status.json and scheduler/artifacts for
the latest job state; this submission record is not a completed result.
