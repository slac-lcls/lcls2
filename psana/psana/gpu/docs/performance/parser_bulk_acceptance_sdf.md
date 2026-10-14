# GPU parser and bulk-read performance acceptance — SDF

Status: completed, 2026-09-16. All 96 timing samples and 16 full-window
diagnostics completed their checks, but performance acceptance is **not
passed**. Three-repetition comparisons show material regressions in the tested
SDF CPU-fallback configuration. This does not validate true GDS performance.

Production branch: `codex/psana2-gpu-xtc-parser`, revision `8f94e3c7b`.
Benchmark artifacts, isolated historical sources/builds, and logs are local
under `validation/perf-acceptance-20260916/`; they are not production changes.

Primary outcome: the current bulk-enabled path has median throughput of
**214.3 events/s warm and 104.9 cold**, versus **361.3 warm and 203.0 cold**
for legacy addressing under the same settings. Three repetitions confirm a
material integration-series regression; this is not an isolated XTC-walker
measurement. The bulk toggle's warm effect is variable and its cold effect is
consistently negative here. See [Results](#results) for ranges and paired deltas.

The large-batch mixed-detector comparison also regressed: at batch size 1000
and 8 GiB, bulk loses about 52% cold and 63% warm throughput. Live diagnostics
confirmed 12.8× more execution subbatches despite roughly 2.9× fewer reads.

Diagnostics also found per-segment gather/lookup overhead and a gap between
the memory ledger and sampled live CuPy allocations. Those need investigation
before declaring the implementation performance-accepted or using the ledger
alone to size multiple BDs. No production code was changed by this benchmark.

## Comparison

| Case | Revision | Integrated parser | Bulk reads |
| --- | --- | --- | --- |
| A | `f52e90cc6` | No | No |
| B | `803a70011` | Yes | No |
| C | `8f94e3c7b` | Yes | Disabled |
| D | `8f94e3c7b` | Yes | Enabled |

A to B is an integration-series comparison, not an isolated parser-kernel
measurement. Intervening changes include general detectors, routing, cleanup,
and a master merge. C to D includes residency and scheduling as well as
coalescing. It does not only change KvikIO submission sizes.

## Builds and workload

All installs use the SDF `ps_20241122` Python environment. Historical A's
`build_psana.sh` attempts a removed `setup.py`, so it was not usable. Its
existing Meson build was invoked directly with `debugoptimized`, matching B
and the current install's build type; no historical source was patched.
Imported Python and extension paths are checked and recorded separately for
each prefix. Runtime logs also record module hashes, CUDA/CuPy/KvikIO versions,
GPU visibility, task size, thread count, and actual fallback/GDS mode.

Primary workload: `mfx101210926` run 387, first 10,000 events, Jungfrau streams
5–9. The stage planner reads CPU SMD locators and records exact L1Accept byte
totals and timestamp hashes. It copies prefixes through the
last required event with up to 16 MiB of trailing margin, preserving original
offsets and the matching smalldata files. A local header walk then checks root
extents, byte totals, and timestamp hashes against the SMD plan. It rejects misaligned stream event
sequences, short input, and insufficient scratch capacity.
Useful input bytes mean the requested complete L1Accept dgrams, including XTC
headers, not just array payloads. Rates use decimal GB/s; device budgets use GiB.

Primary settings: one A100, one SMD0 + one EB + one BD, `batch_size=20`,
`n_gpu_streams=1`, 8 GiB per-BD budget, eight KvikIO workers, 1 MiB task size,
compatibility mode, and no D2H in timed loops. The event loop consumes only
timestamps; the pipeline still performs Jungfrau calibration.
Primary/scaling node: sdfampere040, A100-SXM4-40GB,
UUID `GPU-af5b47e0-90dc-4fff-0761-9557f18e66d3`, driver 575.57.08.
The allocation is not node-exclusive; another dashboard allocation was present
on sdfampere040. Paired runs share the same allocation, but external node-load
effects cannot be ruled out. Bulk runs use separate sdfampere003/027 nodes.
MPI uses `--bind-to none`, matching the earlier runner. CPU/NUMA placement is
not fixed between samples; include ranges and avoid interpreting small deltas
as established gains. A tighter affinity-controlled follow-up may be needed.

## Cache and timing protocol

- Stage into a unique job-local `/lscratch/monarin/parser-accept-...` directory.
  Staging and cache preparation are outside timing.
- Validate three events against independent CPU raw and float32-calibration
  hashes before timing each revision; common-mode correction is disabled.
  Signed zero and NaN payloads are canonicalized before hashing; finite
  nonzero values must match exactly.
- The initial historical-A check failed in calibration-name lookup before GPU
  processing. Inspection found a URL-convention mismatch: A uses
  `LCLS_CALIB_HTTP` directly as the full service URL, whereas current code
  appends `/calib_ws/` to the supplied base URL. This is not evidence of a
  missing detector calibration or a parser defect.
  The benchmark therefore freezes the current CPU reference's calibration
  dictionary to a local compressed snapshot and supplies the same constants
  through a benchmark-only override of `Run._setup_run_calibconst` for every
  variant. Normal MPI calibration distribution and GPU algorithms are unchanged.
  This compares processing with matched constants, not calibration-DB clients.
  The primary and scaling snapshots were also compared recursively: all 221
  leaves match, including array dtype/shape/content hashes and scalar metadata.
  Their three independent CPU-reference records match exactly. Different
  serialized pickle byte hashes did not indicate different calibration values.
- Each timed MPI process first finishes a separate 100-event run, warming
  kernel compilation/module loading, then creates the measured DataSource.
- After measured-run setup, cold mode calls file-specific `fsync` and
  `POSIX_FADV_DONTNEED`, requiring at most 1% resident pages via `mincore`.
  Warm mode reads all staged prefixes, requiring at least 99% residency.
  Warm residency is checked again after the timed loop.
- Cold means Linux-page-cache-cold local NVMe, not cold remote storage or
  purged device-controller caches. No system-wide cache purge is performed.
- Three repetitions alternate revision order. A common start barrier precedes
  `run.events()`; completion includes a final BD device synchronization.
  Aggregate timing uses the maximum rank duration. Exact event count, unique
  timestamps, and timestamp hash must match the stage manifest.
- This is completed event-loop throughput, not a sum of kernel timings.
  Any lazy GPU-manager construction inside `run.events()` is included and
  amortized over the 10,000-event window; it is not silently subtracted.
  Subsequent explicit shared-memory teardown, garbage collection, allocator
  cleanup, and MPI process exit are outside this event-loop throughput metric.
  It is not whole-job wall-clock throughput.
- Detailed read/subbatch/residency/budget instrumentation runs separately
  from throughput samples. External 500-ms GPU-memory samples are observational
  high-water estimates, not exact allocation peaks.
  Diagnostic runs also save per-BD Python call profiles for the timed event
  loop. Their cumulative times can include CUDA/I/O waits and are not CUDA
  kernel timings; profiling overhead excludes these runs from rate comparisons.
  Full-window diagnostics additionally compare the first three
  events' raw/calibrated values (and epix raw fields) against the frozen CPU
  reference under the actual batch/depth/budget settings. This exercises
  residency decisions that a three-event preflight alone cannot cover.
  Those diagnostics include selected D2H copies; throughput samples do not.
  The diagnostic loop also retains its last checked GPU facade in a local
  variable until loop teardown. Such references can affect allocation lifetime;
  pool measurements characterize this instrumented access pattern, not a
  reference-free throughput loop. The separate short profile and external
  device samples provide additional evidence without those in-loop checks.
  The earlier completed 1,000-event profiling job did not include these extra
  in-loop pixel copies.
- Read counts are psana-level KvikIO submissions, not measured internal KvikIO
  tasks or OS syscalls. A coalesced request can still be split by KvikIO's
  configured 1 MiB task size.

## Execution and harness validation

Allocation request: one A100, 48 CPUs, 450 GiB host memory, initially three
hours; the final continuations requested four hours.
Primary and scaling run sequentially in the same allocation so they can reuse
the validated node-local stage. Submission:

```bash
sbatch --time=03:00:00 --exclude=sdfampere015 \
  validation/perf-acceptance-20260916/run.sbatch \
  --phase primary --repeats 3 --with-scale
```

| Job | Outcome before accepted timing |
| --- | --- |
| 38405634 | Canceled by UID 0 during staging; cause not established. |
| 38406629 | Harness SMD-constructor misuse; corrected to `Dgram(config=config)`. |
| 38406775 | Full stage validated; MPI launcher needed `--oversubscribe` because Slurm allocated one task with 48 CPUs. |
| 38409088 | Prior stage was absent in the next allocation, even on the same node; reuse across jobs abandoned. |
| 38409316 | Historical calibration URL convention mismatch; frozen inputs introduced as described above. |
| 38409805 | All A/B/C/D CPU/GPU correctness checks passed; diagnostic wrapper incorrectly used `subbatch.header.n_events`, fixed to `subbatch.n_events`. |
| 38410280 | On sdfampere040: correctness/smoke passed; PREEMPTED after two complete primary repetitions (16 accepted samples), before scaling. |

These failed attempts are excluded from throughput results. The benchmark
now checks CPU/GPU correctness and a 100-event diagnostic smoke run before
staging the full dataset. The staging helper's three focused tests passed.
All three Meson installs report `debugoptimized`, optimization `2`, and LTO.
The historical builds use unmodified source. No production files were changed.

The five primary SMD plans matched 335,571,760,000 useful bytes and timestamp
SHA256 `23f9051ac9f98208310171f734251b0ccd1f40fa048d1748553cbd38cce559d6`.
The validated full stage includes 335,656,487,644 bytes with trailing margins.
After primary correctness and smoke checks passed, bulk job `38407865` was
released to run independently, excluding sdfampere040 to avoid local-I/O
contention. An earlier separate scaling job, `38407864`, was canceled in
favor of same-allocation scaling.
Bulk job `38407865` was PREEMPTED after 52 minutes, with 12 accepted samples
(six complete C/D pairs). Its same-node continuation `38416874` preserved these
pairs but failed before correctness processing with
`cudaErrorDevicesUnavailable` on a newly assigned GPU. No timing was accepted
from that retry; the precise CUDA allocation cause is not established.
Retry `38419425` initially targeted sdfampere028, but its long queue estimate
led to allowing another matching A100 node; it runs on sdfampere027. Only
complete C/D pairs are retained, incomplete pairs are repeated, and
the original frozen constants/CPU reference are reused. Scratch is restaged
and GPU correctness/cache checks are repeated in the new allocation. Results
from different nodes are not pooled into a single absolute-rate median;
paired C/D changes remain within each allocation and must be labeled by node.
Seven focused staging/continuation helper tests pass. Allocation boundaries and
GPU identities must remain visible when interpreting any combined results.
Each continuation's `results.json` already includes preserved comparisons.
Summarize only the latest result file per phase; including both it and its
parent would double-count samples. `source_logs` identifies retained records.
Bulk continuation `38419425` completed all remaining comparisons and eight
diagnostics in 2:16:54. Its full staged copy was removed after success; original
data and benchmark logs were retained. Contingency `38422503` detected the
completed matrix and exited without repeating work (five seconds, sdfampere004).
Primary continuation `38419641` completed on sdfampere040 with the original
constants and complete A/B/C/D blocks retained. It finished the third primary
repetition, primary diagnostics, and the entire scaling matrix in 2:14:24.
Scaling continuation `38419679` then detected that all comparisons and
diagnostics were complete and exited without repeating work (four seconds).
The superseded pending scaling job `38418970` was canceled; it produced no
measurements. The full primary/scaling staged copy was removed after success;
original source data, manifests in the logs, and benchmark results were retained.
Separate diagnostic-only job `38417227` completed on sdfampere028 with 100 GiB
host memory and 1,000 run-387 events. It repeated correctness/staging/cache
checks and profiled A/B/C/D; its instrumented timings are not throughput samples.

The 100-event smoke checks accounted for exactly 3,355,717,600 input bytes.
A/B/C made 500 psana read requests; D made 30, with all five streams resident
and five 20-event executions. These instrumented, short-window smoke checks
validate the harness, not throughput acceptance.
Thirty rather than 25 bulk requests is expected: the SMD locators show a
24-byte gap before event index 35, within the second batch. The planner does
not merge across that gap, adding one range per stream. Batch boundaries and
file gaps both limit coalescing; one stream does not imply one request.

The follow-ups compare C/D for run-51 Jungfrau plus epix100 at batch
sizes 10/1000 and budgets 1/8 GiB, then C/D at two/four BDs sharing one GPU.
The run-51 sample is 2,000 events (two full 1,000-event batches), with
69,277,200,000 useful bytes. Run 51 is a same-rate size comparison, not a real
sparse-Jungfrau workload. Its shorter window limits sustained-throughput claims.
Its epix100 dgram is 1,081,424 bytes per event (about 1.031 MiB), not the
hypothetical 100 KiB small detector discussed during admission design. Jungfrau
occupies five streams at approximately 6, 7, 7, 5, and 7 MiB/event. At a
1,000-event batch size, even the epix input alone exceeds a 1 GiB budget, so
this case also tests execution deferral rather than assuming residency.
Follow-ups were launched only after the primary correctness and harness
checks passed.

## Short diagnostic findings

Job `38417227` completed on sdfampere028. All four CPU/GPU checks and smoke
checks passed. Its separate warm, instrumented 1,000-event runs each accounted
for exactly 33,557,176,000 input bytes and fifty 20-event executions.

| Case | Python calls, millions | psana reads | Tracked+held peak, MiB | Sampled device peak, MiB |
| --- | ---: | ---: | ---: | ---: |
| A | 0.404 | 5,000 | 2,560.1 | 3,631 |
| B | 1.548 | 5,000 | 2,561.8 | 3,633 |
| C | 3.264 | 5,000 | 3,201.8 | 3,633 |
| D | 4.001 | 295 | 3,201.8 | 8,761 |

Profiles identify these investigation targets, not a complete causal attribution:

- A/B/C read-future waits were very similar: about 1.34–1.35 seconds. A's
  `_submit_gpu` cumulative time was 0.173 s versus B's 1.686 s and C's 1.870 s.
  The integrated paths made 32,000 `_gather_locator_field_gpu` calls and
  41,600 parser `locate` calls. The gather implementation launches per detector
  segment per event, and eager field location also adds work. These counts
  justify investigating batched gathers/field decoding before blaming the XTC
  walker alone.
- Current `GpuDetectorBinding.stream_ids` reconstructs stream membership from
  field bindings. The property was called 4,000 times in C and 5,000 in D
  (0.400/0.502 s cumulative), versus 1,000 calls in B. Configure-derived
  membership caching is another targeted optimization candidate.
- D made 50 resident-input admissions with all five streams resident. Its
  `_start_resident_input` cumulative time was 2.849 s, including read waits and
  parsing. Parsing moves out of `_submit_gpu` into that phase, so comparing
  `_submit_gpu` alone would misleadingly make D appear cheaper. Repeated buffer
  trimming/admission and reduced overlap require timeline-level investigation.
- D's sampled device peak (8.56 GiB) was much larger than its tracked+held
  peak (3.13 GiB), and exceeded the nominal 8 GiB tracked-allocation quota.
  Device usage includes CUDA/runtime and allocator retention; it is not the
  ledger. A passing ledger bound alone does not establish a device-memory
  high-water bound for multi-BD sizing.

The subsequent **10,000-event full-window diagnostics** in `38419641` added
direct CuPy-pool sampling at reserve/hold calls and after the timed loop. All
four passed their first-three-event CPU-reference pixel checks under the actual
batch/depth/budget settings. All accounted for 335,571,760,000 bytes and 500
twenty-event executions. A/B/C made 50,000 psana reads; D made **2,885**
(577 per stream), with all five streams resident in each batch. Coalescing
therefore reduced request count by about 17.3× without improving cold throughput.

| Case | Ledger peak, MiB | Sampled CuPy used peak, MiB | Pool total at loop end, MiB | Pool used at loop end, MiB |
| --- | ---: | ---: | ---: | ---: |
| A | 2,560.1 | 2,560.1 | 3,200.1 | 1,920.0 |
| B | 2,561.8 | 3,201.9 | 3,201.9 | 2,561.9 |
| C | 3,201.8 | 3,201.9 | 3,201.9 | 2,561.9 |
| D | 3,201.8 | 8,325.5 | 8,325.5 | 5,123.7 |

This narrows the issue: D had a sampled **used** pool peak of 8.13 GiB, not
merely unused cached capacity, despite a 3.13 GiB ledger peak. At loop end,
about 5.00 GiB was still used and 3.13 GiB unused but retained in the pool.
The earlier 100-event smoke run reached the same used peak but ended with
only 2.50 GiB used, so end-of-loop retention is not identical across windows.
Trace the lifetime of
allocations/views across trimming and admission before attributing the exact
cause. The nominal quota bounds ledger charges, not this observed physical
high-water. No OOM occurred in these single-BD tests; that does not validate
multi-BD memory safety. These observations use the same 8 GiB configured budget
and twenty-event executions; they are not the later tight-budget bulk cases.

There is a concrete ownership distinction to examine in source:
`GPUDetector.trim_slot_buffers`, reader `trim_free_buffers`, and parser
`trim_free_buffers` release ledger charges when dropping their stored array
references. That does not prove that all other array/view references are gone.
`InputWindow._try_retire` marks storage released and clears the release
callback, but retains `self.batch`. CUDA-consumer completion makes reuse safe;
it does not itself destroy these Python/CuPy references. This is a plausible
contributor to the observed gap, not a proven attribution of every excess byte.

These short, profiled runs include lazy setup and instrumentation overhead.
Cumulative call times overlap and must not be added together. They are not
CUDA kernel timings or replacements for the uninstrumented 10,000-event rates.
No production optimization or policy change was made.

## Results

Primary: **three accepted samples per cell**, median (minimum–maximum), on
sdfampere040 and the same GPU UUID. The first two repetitions are from
`38410280`; the third is from continuation `38419641` after restaging and
repeating correctness checks with the original frozen constants.

All 24 samples verified 10,000 unique events, the same timestamp SHA256, and
the same 335,571,760,000-byte payload manifest. Across the 12 cold samples,
pre-loop residency was 0–0.0002002%. Across the 12 warm samples, pre-loop
residency was 99.556–100%, and post-loop residency was 100% in every sample.

| Case | Cold events/s | Cold GB/s | Warm events/s | Warm GB/s |
| --- | ---: | ---: | ---: | ---: |
| A: legacy addressing | 203.0 (194.6–221.0) | 6.813 | 361.3 (355.4–368.5) | 12.126 |
| B: integrated parser before bulk | 170.8 (168.8–177.8) | 5.731 | 259.0 (254.8–268.4) | 8.693 |
| C: current, bulk disabled | 157.5 (149.2–162.9) | 5.284 | 229.4 (183.5–231.1) | 7.699 |
| D: current, bulk enabled | 104.9 (97.6–105.4) | 3.520 | 214.3 (210.9–216.7) | 7.191 |

Within each repetition, A→B loses 27.2–28.3% warm throughput and 12.3–19.5%
cold throughput. Current bulk-enabled D is 40.7–41.2% below A warm and
45.8–55.9% below A cold. The bulk toggle alone (C→D) loses 29.7–38.0% cold
throughput. Its warm difference changes sign (+18.1%, −7.3%, −8.1%), so the initial
apparent warm gain is not established. Do not attribute the A/B difference
solely to the XTC walker: that comparison spans the integration series.

All **96 throughput samples** are complete: 24 primary, 48 mixed-detector,
and 24 scaling. All **16 full-window diagnostics** and the separate four-case
short profiling comparison are also complete.

### Mixed-detector throughput

All **48 samples** passed event/timestamp validation: 2,000 unique events,
69,277,200,000 useful bytes, and SHA256
`eb7a5a3e810b806810fc03447c204990dd35615e239f5943c32362538abfa394`.
There are three paired repetitions per condition, with two execution slots.
Initial pairs ran on sdfampere003; continuation pairs ran on sdfampere027,
A100-SXM4-40GB UUID `GPU-d4eeaffa-a368-71d7-b83b-9315de587942`.

Absolute rates below are **bulk off / bulk on**, events/s, kept separate by
node. sdfampere003 has one sample per listed variant/condition. sdfampere027
columns give medians and sample counts per variant. The final column is the
median (range) of three **within-allocation paired** percentage changes; no
absolute-rate median mixes nodes.

| Batch size | Budget, GiB | Cache | sdfampere003 off/on | sdfampere027 off/on | Paired on versus off |
| ---: | ---: | --- | ---: | ---: | ---: |
| 10 | 1 | Cold | 46.8 / 40.9 | 46.0 / 40.6 (n=2) | −11.8% (−12.6 to −11.8) |
| 10 | 1 | Warm | 65.6 / 54.4 | 63.6 / 53.5 (n=2) | −16.8% (−17.1 to −15.1) |
| 10 | 8 | Cold | 101.3 / 75.3 | 102.2 / 76.5 (n=2) | −25.7% (−26.4 to −23.8) |
| 10 | 8 | Warm | 195.3 / 177.5 | 200.9 / 179.0 (n=2) | −10.4% (−11.4 to −9.1) |
| 1000 | 1 | Cold | 47.6 / 45.5 | 46.9 / 44.8 (n=2) | −4.5% (−4.9 to −4.3) |
| 1000 | 1 | Warm | — | 61.2 / 65.8 (n=3) | +7.5% (−0.8 to +14.7) |
| 1000 | 8 | Cold | 102.0 / 54.0 | 111.6 / 53.5 (n=2) | −51.7% (−52.5 to −47.0) |
| 1000 | 8 | Warm | — | 238.6 / 87.4 (n=3) | −63.4% (−63.9 to −62.5) |

The large-batch/8-GiB penalty is repeatable in both cache modes. The apparent
large-batch/1-GiB warm benefit is variable and changes sign in one repetition;
do not treat its median alone as a demonstrated gain.

A **CPU-only planner replay**, not another GPU measurement, uses the exact
run-51 sizes/setup/parser/output costs recorded in
`trace-bulk-sdf-38395705.log`. At batch size 1000, depth 2, and 8 GiB:

| Setting | Resident dense stream IDs | Resident input+parser bytes | Execution plan per 1000-event batch |
| --- | --- | ---: | --- |
| Bulk off | None | 0 | 38 × 26 events + 1 × 12 events |
| Bulk on | 2 (epix), 4 (5-MiB Jungfrau stream) | 6,360,242,000 | 500 × 2 events |

The live diagnostics **confirmed both execution plans**, including resident
dense streams `(2, 4)` for bulk on. That is 12.8 times as many execution
subdivisions despite fewer input reads. This supports a scheduling explanation
for the large-batch penalty, but does not isolate its entire timing cost from
allocation and overlap effects. The planner guarantees minimum progress, not
maximum execution throughput. Reproduce the CPU calculation with
`validation/perf-acceptance-20260916/replay_admission.py` in the activated
current psana environment.

### Mixed-detector read, scheduling, and memory diagnostics

All eight diagnostics passed the selected CPU-reference checks and accounted
for exactly 69,277,200,000 input bytes. Each covers 2,000 events: **200 user
batches** at batch size 10, or **two user batches** at batch size 1000.
Execution subbatches are the smaller units submitted to the two execution slots.
Read counts below cover the whole 2,000-event sample, not one user batch.

| User batch size | Budget, GiB | Resident streams with bulk on | Executions per user batch, off / on | Total psana reads, off / on |
| ---: | ---: | --- | --- | ---: |
| 10 | 1 | epix s006 | 10 × 1 / 10 × 1 | 12,000 / 10,217 |
| 10 | 8 | All six streams | 1 × 10 / 1 × 10 | 12,000 / 1,302 |
| 1000 | 1 | None | 1000 × 1 / 1000 × 1 | 12,000 / 12,000 |
| 1000 | 8 | epix s006, Jungfrau s008 | (38 × 26 + 1 × 12) / 500 × 2 | 12,000 / 4,106 |

With bulk off, every stream makes 2,000 psana reads in every configuration.
With bulk on, measured per-stream counts are:

| User batch size | Budget, GiB | epix s006 | Jungfrau s008 | Each other Jungfrau stream: s003, s005, s007, s009 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 1 | 217 | 2,000 | 2,000 |
| 10 | 8 | 217 | 217 | 217 |
| 1000 | 1 | 2,000 | 2,000 | 2,000 |
| 1000 | 8 | 19 | 19 | 1,017 |

Thus large-batch residency really does reduce small-detector submissions:
epix falls from 2,000 to 19. It also leaves much less room for execution, so
the complete workload becomes slower. At batch size 10 / 8 GiB, executions
do **not** shrink, yet throughput still regresses; subbatch size alone cannot
explain every penalty. File gaps prevent one read per resident stream/batch.
For example, large-batch epix requests range from 3,244,272 to 129,770,880 bytes,
not one approximately 1-GiB request. These are psana requests to KvikIO;
the configured 1-MiB KvikIO task size can split them internally. Internal task
counts and OS syscall counts were not measured.

| User batch size | Budget, GiB | Ledger peak off / on, MiB | Sampled live CuPy peak off / on, MiB |
| ---: | ---: | ---: | ---: |
| 10 | 1 | 898.3 / 906.7 | 898.5 / 1,184.0 |
| 10 | 8 | 3,222.8 / 1,931.4 | 3,222.9 / 4,514.3 |
| 1000 | 1 | 898.3 / 898.3 | 898.5 / 898.5 |
| 1000 | 8 | 7,355.1 / 7,197.9 | 7,355.2 / 13,756.1 |

The largest bulk diagnostic reached **13.43 GiB of live pool allocations**
against a 7.03-GiB ledger peak and an 8-GiB configured budget. About 12.33 GiB
remained used at loop end. Diagnostic reference retention is a caveat, but
uninstrumented large-batch bulk samples on sdfampere027 also reached
**14,201 MiB of sampled device usage**. In the third cold and warm samples,
that peak occurs between cache preparation and post-loop cleanup, not only
during warmup. Device usage includes runtime/context and pool retention; it
is not identical to live pool usage. Both measurements support investigating
physical allocation lifetime before treating the ledger as a memory bound.

Failed, cache-invalid, or incomplete runs are excluded from throughput
comparisons. These completed comparisons do not pass performance acceptance.

### Shared-GPU scaling

All **24 samples** completed on sdfampere040: 10,000 events, one A100, batch
20, depth 1, and 8 GiB budget **per BD**. Every sample verified the same unique
timestamps and useful-byte manifest as the primary comparison. Rates are
median (minimum–maximum), three samples per cell. Percentage changes are
computed within each repetition and then summarized; they need not equal
the ratio of the separately computed rate medians.

| BDs | Cache | Bulk off, events/s | Bulk on, events/s | Paired on versus off |
| ---: | --- | ---: | ---: | ---: |
| 2 | Cold | 250.6 (234.3–257.2) | 167.7 (164.9–169.2) | −32.5% (−34.8 to −29.6) |
| 2 | Warm | 344.2 (340.9–367.7) | 329.6 (320.8–330.0) | −4.1% (−12.8 to −3.3) |
| 4 | Cold | 293.8 (293.0–302.4) | 240.5 (234.9–240.8) | −19.8% (−20.4 to −18.1) |
| 4 | Warm | 378.4 (376.7–399.1) | 381.7 (360.4–393.3) | −4.3% (−4.4 to +3.9) |

More BDs improve aggregate rates for both current paths. Bulk remains
consistently slower cold; the four-BD warm difference changes sign and does
not establish a reliable benefit. Do not transfer the large cold penalty
directly to warm scaling.

First-cold-repetition sampled device peaks, off/on, were **7,259 / 17,515 MiB**
at two BDs and **13,745 / 34,257 MiB** at four BDs. Four-BD completion without
OOM does not establish a general memory bound; per-BD budgets do not include
every allocation observed by the device or CuPy pool.
These are 500-ms `nvidia-smi` samples for the assigned GPU UUID over
the entire benchmark process, including warmup, not exact timed-loop or
per-process allocation peaks. They include runtime/context and retained pool
memory. They support exercising physical memory explicitly rather than
extrapolating the much smaller ledger peak.
This scaling matrix compares current C/D only. It does not establish
historical A/B multi-BD parity, and no rate is compared across different BD
counts as though resource use were identical.

All four full-window scaling diagnostics passed their selected CPU/GPU value
checks and accounted for exactly 335,571,760,000 bytes. Across ranks they each
submitted 500 twenty-event executions. C made 50,000 reads and D made 2,885,
at both two and four BDs. No duplicated input reads are hidden in those totals.

| BDs | Path | Events per BD in diagnostic | Ledger peak per BD, MiB | Sampled live CuPy peak per BD, MiB |
| ---: | --- | --- | --- | --- |
| 2 | C: off | 5,240 / 4,760 | 3,201.8 / 2,817.8 | 3,201.9 / 2,817.9 |
| 2 | D: on | 4,960 / 5,040 | 3,201.8 / 2,817.8 | 8,325.5 / 7,941.5 |
| 4 | C: off | 2,440 / 2,520 / 2,440 / 2,600 | 3,201.8 / 2,817.8 / 2,817.8 / 2,817.8 | 3,201.9 / 2,817.9 / 2,817.9 / 2,817.9 |
| 4 | D: on | 2,460 / 2,440 / 2,640 / 2,460 | 3,201.8 / 2,817.8 / 2,817.8 / 2,817.8 | 8,325.5 / 7,941.5 / 7,941.5 / 7,941.5 |

Entries follow increasing MPI BD rank. These are individual sampled peaks,
not a claim that every rank peaked simultaneously. The diagnostic access and
reference-lifetime caveats above apply. The pool/ledger gap persists at both
BD counts; successful completion on this GPU does not remove that concern.

## Recommended follow-up work (not implemented)

1. Resolve the ledger-versus-live-allocation discrepancy first. Trace when
   trimming releases budget charges versus when arrays, views, pending reads,
   and input/result owners actually relinquish memory. Exercise multiple BDs
   and retained Python event references; do not equate an expired access lease
   with a physically freed allocation.
2. Reduce per-segment launch and lookup overhead while preserving the generic
   detector interface: investigate batched gather/locator work rather than
   reverting to detector-specific raw offsets. Measure CUDA timelines before
   choosing between fused calibration access and separate materialization.
3. Cache run-scoped stream/segment membership where Configure and resolved
   detector ordering make it immutable. Avoid rebuilding it for each event
   during admission and processing.
4. Investigate resident-input scheduling and trimming separately from request
   coalescing. Fewer preads do not prove better overlap, larger execution
   subbatches, or higher throughput. These tests do not compare mean-dgram
   admission against the former total-footprint priority.
5. Repeat targeted comparisons after fixes, with explicit CPU/NUMA affinity
   and the same cache checks. True GDS, production mixed-rate small detectors,
   and D2H-heavy workloads remain separate acceptance work.

Keep `gpu_bulk_read=False` available as a diagnostic/control path while these
issues are investigated. This report does not change its default or any
production implementation.
