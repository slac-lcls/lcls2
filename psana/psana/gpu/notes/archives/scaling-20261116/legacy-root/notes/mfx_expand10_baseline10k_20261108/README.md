# Clean 10k sample10 baseline: one worker, one GPU

Stage 1 of the [single-node baseline plan](../mfx_expand10_single_node_baseline_20261108.md).
This campaign uses the existing reviewed performance runner and frozen runtime,
without changing the expansion/calibration/cube kernels or framework. It is a
**serial single-worker** baseline; actual MPI BD scaling is a later stage.

Six fresh processes run sequentially in one Slurm allocation, alternating one
and two execution streams: s1-r1, s2-r1, s1-r2, s2-r2, s1-r3, s2-r3. The accepted
sample10 fixture has 10,000 unique original MFX run-10 events, five JF-bearing
files and five ancillary streams, with 37.683 GB total stored XTC/SMD bytes.
Its manifest hash and all file sizes are checked before use. Full file hashes
remain those from generation acceptance; this campaign does not rehash the full
fixture before timing or claim cold-cache storage performance.

Fixed settings: 100 bins (99 active, one empty control), nominal batch 7,
requested warmup 1,000 events, 4 GiB framework budget, 512 KiB read-group target,
KvikIO CPU fallback with four workers, four allocated CPU cores, and one GPU.
Cube state is 6.25 GiB with one execution stream and 12.5 GiB with two. Normal
single-pass calibration is used. No artificial compute inflation, profiler,
per-event publication D2H or full-cube export is included. Existing count-only
admission-pressure instrumentation remains enabled.

Timing is unchanged from the reviewed runner. Warmup rounds up to a callback
boundary and joins existing bank streams once. The steady timer starts inside
the next task callback: that callback's input may already have been read/parsed.
The window ends after EOF and an explicit bank join; rates use the actual
completed-event count. Whole-iteration rate includes all 10k events, initial
warmup/startup and the deliberate boundary join, but excludes setup/preflight
and post-loop validation/cleanup. Both rates are reported; the callback window
is not described as exact end-to-end I/O timing. Compiler cache is node-local
and shared by the six fresh processes. Filesystem cache is uncontrolled and
sequential reads may warm later cases. No shared cache is flushed.

Acceptance uses all 10k original timestamp comparisons in the native runner,
exact accepted/bin/bank counts, and zero admission drains in every case. The
host summarizer additionally reconstructs per-bank bin counts from callback
ordering, checks warmup/event conservation, rederives rates, requires equal
runtime/source/calibration identity and a single host/job, and checks sequential
process order using monotonic memory-sample timestamps. A missing, failed,
partial, wrong-setting or pressured case prevents a passing summary.
This extends event/count acceptance to 10k; it does not newly validate every
cube output pixel. Earlier reviewed 1k numerical tests remain the kernel basis.

GPU and CPU memory samples occur only outside iteration. Reported device maxima
are maxima of those snapshots, not measured continuous peaks. Host VmHWM is
process-lifetime high-water including setup. GPU snapshots before/after the
campaign identify device inventory and idle usage; they do not measure device
utilization during each case or exclude all other CPU/storage activity.

Reproduce with `sbatch --output=EXISTING_LOGDIR/%j.out
--error=EXISTING_LOGDIR/%j.err run.sbatch ABSOLUTE_SOURCE_DIR NEW_RESULT_DIRECTORY`.
All output, job logs and compiler caches go to scratch. Summarize with Python
3.9+ using `summarize.py RESULT_DIRECTORY --output NEW_JSON`. It opens the output
exclusively and requires all six native reports. Raw per-case reports and source
hashes are retained for review; generated data and GPU source are unchanged.

## Measured results

Job **43859846** on **sdfampere011**, A100-SXM4-40GB, completed exit 0 in 4m27s.
All six native cases passed. Each has exactly 1,000 warmup and 9,000 measured
events, 1,500 callbacks (1,000 of size 7 and 500 of size 6), and zero warmup/steady
pressure drains. Two-stream cases split 5,000/5,000 events between banks. The
host summary passed, as did five tests covering native acceptance and rejection
of missing cases, pressure, incorrect bank ownership and false rate arithmetic.

| Streams | Warmed-window rates, r1/r2/r3 (events/s) | Median (range) | Whole-iteration median (range), events/s |
|---|---|---|---|
| 1 | 285.24 / 256.65 / 257.25 | 257.25 (256.65–285.24) | 255.27 (255.01–281.79) |
| 2 | 283.87 / 282.91 / 283.58 | 283.58 (282.91–283.87) | 279.25 (277.37–280.86) |

The ratio of warmed-window medians is **1.1024**, or 10.24% higher with two
streams in this allocation. It is not a controlled causal scaling factor:
one-stream repeat 1 matches/exceeds the two-stream rates, the one-stream spread
is material, and repeated reads/cache/CPU/storage activity are uncontrolled.
All cases are included; no outlier was dropped. Two-stream repeats are locally
consistent; stronger generalization requires the later controlled comparisons.
The shorter prior 1k results are different workload/cache/node measurements and
must not be interpreted as a regression or like-for-like speed comparison.

Warmed windows last 31.55–35.07 seconds. Maximum sampled device usage is 7.91 GiB
(one stream) and 14.84 GiB (two), with 6.25/12.5 GiB persistent cube banks. Host VmHWM
is 1.93–1.96 GiB. These snapshots do not establish device allocation peaks.

The initial GPU inventory snapshot showed 6.2 GiB/100% activity, which was gone
by all six before_setup snapshots (424.125 MiB each). A one-off process query
during s1-r2 showed only our benchmark Python. Its slight perturbation is not
quantified. See [activity observation](gpu-activity-observation.md); do not infer
continuous device exclusivity from the spot checks. Native stderr has the expected
CPU-fallback warning in each case; no native case failed. Job logs here are
normalized copies (NUL/trailing-whitespace removal); byte-exact originals are
bound in artifacts.json.

Native reports, GPU inventory and logs:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/baseline10k_20261108_r1/`

[summary.json](summary.json) contains rates, callback sizes, counts and memory
samples. [artifacts.json](artifacts.json) binds original report/log files by SHA.
[source-verification.json](source-verification.json) records rechecked source and
runtime hashes. The existing runner/kernels/runtime are unchanged. Subsequent
batch tuning, compressed-path attribution and MPI scaling remain separate stages.

The post-campaign GPU snapshot reports 0 MiB used and 97% utilization. This is
one sample immediately after the final process exited; it could reflect recent
benchmark activity, but the available evidence does not attribute the reading.

Independent restricted Claude review **passed** executable commit `525193585`,
with no blocking findings. All three findings were addressed through evidence/
documentation closure; see [responses](review-01/response.md). The coordinator
subsequently supplied the missing review-copy index, rechecked each native copy
byte-for-byte against its scratch original and saved their matching hashes.
The reviewer did not independently execute tests or hash the artifacts. No
executable changes or extra GPU runs followed review.
