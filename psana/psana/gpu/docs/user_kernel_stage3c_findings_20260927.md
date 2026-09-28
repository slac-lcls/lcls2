# Stage 3c: batched scheduling performance acceptance

Stage 3b (`a1c7ae252`) was reviewed with no blocking findings. Stage 3c is
running; performance acceptance remains pending. Stage 4 public task processing
and automatic output D2H remain gated.

## Review and correctness baseline

Reviewed single invocation after input selection/gather, completion recording
after callback work, context expiry, validation before publication registration,
sparse/reordered row mappings, shared backing ownership, terminal consumer
completion, and drain/quarantine behavior. No runtime changes were needed.
The nine implementation/test/harness hashes in the Stage 3b validation record
matched the accepted sources before extending the measurement harness.

The accepted unchanged runtime has 447 local unit passes, 514 full CPU passes
plus four byhand passes, 128 A100 integration passes, and four-rank MPI input
and task-setup acceptance. See [Stage 3b findings](user_kernel_stage3b_findings_20260927.md)
for jobs, the isolated-directory CPU retry, and coverage limits.

## Input-only regression

Job **39317728**, running on **sdfampere023**:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage3c-regression-20260927-r1`.

Frozen Stage 2 `f5b4cfb0e` versus corrected Stage 3 `a1c7ae252`, with no task and
identical benchmark-only dense preparation. Four balanced rounds cover warm
1/2/4 BD and cold 4 BD; the warm 1-BD case includes A/A noise controls.
There are 40 timed samples and eight exact-pixel preflights. The 10,000-event
Jungfrau run uses batch 20, depth 1 and unchanged KvikIO settings. Strict
per-file cache conditioning remains enabled. Initial cache preflight passed.

Expected duration is 90–120 minutes after allocation, subject to conditioning.
A repeatable >5% slowdown requires investigation against the paired and A/A
noise. Input-only success alone cannot accept batched scheduling.

## Callback comparison

Job **39317958**, submitted with a fail-fast correctness smoke before timing:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage3c-callback-comparison-20260927-r1`.

The common maintained harness runs the original per-event Stage 3 runtime
`c64fcb2ba` and corrected batched runtime `a1c7ae252` in separate processes.
Six balanced rounds reverse runtime, profile and mode order. The reference is
internal producer scheduling; a public user-event-loop comparison still depends
on Stage 4.

- Batch sizes 1, 3 and 20; execution depths 1 and 2.
- No task, empty callback, scratch+kernel and publication+kernel.
- Additional matched scratch/publication cases preallocate one output per
  execution slot and its event row views before timing. Slot retirement precedes
  reuse, separating allocation savings from invocation/launch savings.
- The 900-pixel fixture measures scheduling overhead. A synthetic full-frame
  `(32, 512, 1024)` uint16 fixture measures representative Jungfrau-sized task
  data. This fixture performs no disk I/O or calibration.
- Scratch computes raw+1 for every pixel with uint16 wraparound. Publication
  computes the same one uint32 scalar per event in both runtimes.
- Compilation, parsing, raw upload, preallocation, warmup and diagnostic output
  verification are outside timing. Each timed loop drains all in-flight work.
  Micro/full-frame cases use 200/50 submissions per sample.

Separate correctness preflights validate every output through repeated slot
reuse. They count callbacks, user allocations and kernels, framework parser and
gather kernels, completion-event creation, and task metadata uploads. Gather-map
copies are counted as logical prepare/upload calls, not low-level CUDA memcpy
traces. Diagnostic D2H copies are reported separately; automatic D2H is absent.
All diagnostic hooks are restored before timing.

Reports include submission, retirement and total-loop microseconds per subbatch
and per event; paired differences; live user-output byte bounds; preallocated
buffer counts; framework budget and CuPy pool bytes after drain. The latter are
not peak-memory measurements. Fresh and preallocated output policies are labeled.

The smoke runs the entire matrix with one timed submission, twice in reversed
order. Only successful smoke completion starts the six-round campaign. Estimated
allocation duration is 20–30 minutes; queue time is additional. The job has a
45-minute limit. Both runtime sources, inherited native dependencies, scripts
and launch configuration are frozen under a verified 4,972-entry manifest.

Local syntax, command-line interface and whitespace checks passed. A separate
shared-allocation correctness smoke, job **39317997**, is also queued against
the same frozen files so validation need not wait for the exclusive allocation.
Its timings are not performance evidence. Device smoke and performance results
are pending; neither the smoke nor submitted jobs constitute performance acceptance.
