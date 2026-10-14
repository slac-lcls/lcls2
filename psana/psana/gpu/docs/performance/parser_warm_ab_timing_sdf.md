# Warm A/B parser timing follow-up (SDF)

This is the measurement follow-up to [parser/bulk acceptance](parser_bulk_acceptance_sdf.md).
Only A (legacy addressing, `f52e90cc6`) and B (integrated parser, `803a70011`)
are compared. Neither bulk-read policy nor production implementation is changed.

## Measurement design

- Real data: `mfx101210926`, run 387, first 10,000 events, Jungfrau, streams 5–9.
- One SMD0, one EB, one BD, one A100; batch 20, pool depth 1, 8 GiB budget.
- Same frozen CPU-verified calibration constants and historical builds as acceptance.
- Node-local bounded XTC prefixes; at least 99% Linux page-cache residency before
  and after each timed run. KvikIO compatibility mode ON: this includes H2D and is
  **not** a GDS measurement. Eight reader threads, 1 MiB KvikIO task size.
- Separate 100-event warmup before every run. Timed loop includes lazy GPU setup
  and ends after the existing final device synchronization.
- Three repetitions each of uninstrumented A/B and CPU/NVTX-instrumented A/B,
  alternating variant/mode order. Separate Nsight traces are diagnostic only.
- Exact raw/calibrated CPU-reference checks for the first three events, both with
  and without hooks; full timed-run event count, uniqueness, and timestamp hash.
- No cProfile, no per-field host timers, no added CUDA synchronization, and no
  changed kernels, reads, memory budget, segment ordering, or lease behavior.

## Instrumentation

`validation/perf-acceptance-20260916/phase_timing.py` installs process-local hooks
only when the benchmark passes `--timing-mode cpu` or `--timing-mode cpu-nvtx`.
The default is `off`. Historical installed sources remain untouched. Selected
function ASTs gain coarse `with recorder.phase(...)` blocks; matching fails if
the expected statement structure changes. Tests strip the added scopes and
verify equality with the original AST. No added scope surrounds a generator yield.

| Host range | Work covered |
| --- | --- |
| `setup.detectors` | Lazy detector/configuration setup, constants/device allocation |
| `upstream.next_batch` | Obtaining the next upstream batch |
| `read.submit`, `read.wait` | KvikIO read submission and completion waiting |
| `submit` | Whole execution-slot submission; inclusive parent of parser/detector ranges |
| `xtc.metadata` | Host dgram records, reusable device metadata allocation/upload (B) |
| `xtc.walk` | Batch construction and XTC-walker submission (B) |
| `xtc.locate_all` | Eager field locators: array initialization, lookup launches, ready events (B) |
| `event.views` | CPU event-to-stream dgram views (B) |
| `detector.sources` | Selecting events/sources for this detector |
| `detector.buffers` | Raw/calibration/presence output buffer management |
| `detector.gather` | All segment/stream lookup, dependency and gather submission for **one event** |
| `detector.zero_target`, `detector.zero_present` | Explicit raw/presence initialization; A zeros raw only when incomplete |
| `detector.calib` | Calibration submission |
| `detector.zero_missing` | B's missing-field output cleanup submission |
| `retire.producer_wait`, `retire.consumer_wait` | Slot completion/consumer retirement |

Host results include call counts, inclusive wall time, exclusive wall time after
subtracting measured children, and maximum call time. Do not add an inclusive
parent to its children. `startup/` means before the first event is delivered;
`steady/` starts afterward, excluding the first submitted subbatch. Startup also
contains useful first-batch processing, not just one-time setup.

NVTX ranges are named `psana.A.<range>` / `psana.B.<range>`. Their durations measure
host scopes, **not** GPU execution. Nsight's CUDA correlation supplies actual
kernel/copy durations. GPU durations may overlap each other and host work; they
are not additive to CPU wall times. Instrumented/traced throughput is not the
clean acceptance throughput, and no measured component is automatically a
recoverable end-to-end speedup.

Nsight wraps only BD rank 2; CUDA/NVTX capture begins after warmup/cache preparation
via the CUDA profiler API. CPU sampling, context-switch tracing and CUDA-event
completion tracing are disabled. The final existing synchronization drains GPU
work before capture ends. Reports also include lazy setup, identifiable by NVTX.

## Reproduction

From the worktree root:

```bash
sbatch --exclude=sdfampere013 --time=04:00:00 \
  validation/perf-acceptance-20260916/run_warm_ab.sbatch
```

The controller writes `job-<id>-warm-ab/` under that validation directory, with
per-case logs, GPU utilization samples, source/runtime provenance, incremental
`results.json`, and separate `.nsys-rep` / `.sqlite` trace artifacts. A case is
accepted only after its MPI process exits successfully. Node-local staged copies
are retained for follow-up within the allocation; original data are never changed.

```bash
python validation/perf-acceptance-20260916/summarize_warm_ab.py \
  validation/perf-acceptance-20260916/job-<id>-warm-ab/results.json
```

## Results

First attempt: SDF job **38477145**, `sdfampere013`. All four pixel-reference
preflights passed, but cache residency was only **84.75%** after warming the
313 GiB stage. The guard aborted before the first timed loop; **no throughput
sample was accepted**. Host-wide available memory alone did not establish that
the target files could remain warm in that allocation. The harness now allows
up to three preparation passes (outside timing), retaining the same 99% before/
after acceptance threshold. The cause of that allocation's eviction is not yet
established. Retry **38477958** requested the prior acceptance node,
`sdfampere040`, but was canceled while pending because its GPUs were allocated
and its estimated wait was several hours. Retry **38478311** requests another
A100, excluding `sdfampere013`; all A/B comparisons remain within one allocation.

### Warm throughput and CPU timings

Job **38478311** ran its twelve regular measurements on `sdfampere003`, A100-SXM4-40GB,
UUID `GPU-7fbdbdf1-d40b-1a98-9c73-9b44fecc1d30`, driver 575.57.08.
All four pixel-reference preflights passed on this node too. The first cache
preparation reached 92.98%, then 98.70%, then 100%. A read-only cgroup check
showed the job at its **450 GiB host-memory limit**, with approximately 442 GiB
charged to file cache, despite ample host-wide available memory. This explains
why host-wide `free` output was not a sufficient warm-cache check; the precise
ownership of all cached pages was not established. No shared source cache was
purged and no system-wide settings were changed.

All twelve regular runs were **100% cache-resident before and after**. The six
clean, uninstrumented measurements give:

| Variant | Median loop time | Median throughput | Throughput range, 3 runs |
| --- | ---: | ---: | ---: |
| A | 21.768 s | 459.4 events/s | 413.8–459.6 |
| B | 33.451 s | 298.9 events/s | 288.6–301.7 |

B has **34.9% lower throughput**, or **11.683 s additional elapsed time per
10,000 events**. The prior acceptance allocation showed about 28% lower warm
throughput and about 10.9 s additional time. Absolute rates differ by allocation;
the roughly 11-second added cost is consistent with a substantial added
per-batch/per-event cost becoming more visible when the baseline read path is
faster. This is a comparison of the two historical revisions, not a current-HEAD
parser toggle or an isolated comparison of only the walker implementation.

Three CPU/NVTX-instrumented runs per variant gave median rates of 441.2 events/s
for A and 300.0 events/s for B. Relative median elapsed times were +4.13% (A)
and -0.34% (B) versus controls. These are **observed differences, not a precise
causal estimate of instrumentation overhead**: the node was shared, CPU/NUMA
placement was not pinned, and startup varied. In particular, one A instrumented
run took 5.887 s in detector setup versus 0.960/0.991 s in the other two, lowering
that run to 361.2 events/s. The instrumentation did not make B intrinsically
faster. Clean throughput comes only from the uninstrumented rows above.

Per-scope medians of **steady host wall time for 9,980 events / 499 subbatches**
(not GPU execution time). The first submitted subbatch is excluded. Medians
are taken independently, so rows need not sum exactly to a median parent time:

| Scope | A | B |
| --- | ---: | ---: |
| Read submission | 1.056 s | 0.901 s |
| Read completion waiting | 16.252 s | 15.596 s |
| Event source selection | 0.234 s | 0.492 s |
| Field access/dependency/gather submission | 0.844 s | 7.009 s |
| Eager field location | — | 5.179 s |
| Dgram metadata preparation/upload | — | 0.145 s |
| XTC batch construction/walker submission | — | 0.067 s |
| Event-to-stream views | — | 0.216 s |
| Calibration submission | 0.242 s | 0.267 s |
| Raw/presence initialization + missing-row cleanup submission | 0 s (complete events) | 0.350 s |
| Producer retirement waiting | 1.088 s | 0.004 s |
| Whole slot submission, **inclusive** | 1.700 s | 14.283 s |

The incremental field-access/gather time is **6.165 s**, and eager field
location adds **5.179 s**. Together they account for about **90% of the extra
host time inside slot submission**. Source selection, metadata/view creation,
and zeroing/cleanup submission are much smaller targets. Median one-time
detector setup is 0.991 s (A) versus 0.945 s (B), so its median is not the
regression source.

Retirement waiting shrinks by about 1.08 s in B: its much longer CPU submission
gives asynchronous GPU work more time to finish before retirement. This is one
reason enqueue time, GPU execution time, and waiting time must not be added as
independent recoverable savings. The `upstream.next_batch` wrapper was not invoked
in this MPI path; no upstream timing conclusion is drawn from its absence.

### CUDA/NVTX trace results

Both separate 10,000-event captures completed and exported successfully using
Nsight Systems 2025.3.1. These are **different samples from the host medians**
above; GPU durations must not be subtracted from those host medians to calculate
an exact CPU-only cost. NVTX-to-CUDA correlation assigns kernels to their
innermost host submission range.

| Device operation | A count | A GPU time | B count | B GPU time |
| --- | ---: | ---: | ---: | ---: |
| Gather | 50,000 | 0.963 s | 320,000 | 1.999 s |
| Calibration | 10,000 | 1.832 s | 10,000 | 1.833 s |
| XTC walk | — | — | 500 | **0.0123 s** |
| Field-location kernel | — | — | 96,000 | 0.594 s |
| Locator-status fill kernel | — | — | 96,000 | 0.149 s |
| Locator-row memset | — | — | 96,000 | 0.162 s |
| Raw-output memset | — | — | 10,000 | 0.197 s |
| Presence memset | — | — | 10,000 | 0.011 s |
| Missing-row cleanup kernel | — | — | 10,000 | 0.511 s |

There are 500 execution subbatches in each capture. A gathers once per event/
stream (5 streams); B gathers once per event/segment (32 segments). B locates
192 configured field handles per subbatch. This is **532,500 kernel launches
in B versus 60,000 in A**, plus 116,000 B memset operations. The locator's GPU
work, including initialization, totals about 0.905 s; the walker itself is
only about 12 ms across the whole sample. Calibration duration is essentially
unchanged. Gather is not merely a host cost: its device time also grows by
about 1.04 s, despite copying the same canonical raw payload.

CUDA API counts further show the submission/dependency multiplication:

| API | A calls | B calls |
| --- | ---: | ---: |
| `cuLaunchKernel` | 60,000 | 532,500 |
| `cudaMemsetAsync` | 0 | 116,000 |
| `cudaStreamWaitEvent` | 0 | 320,500 |
| `cudaEventRecord` | 509 | 97,008 |
| `cudaEventCreateWithFlags` | 509 | 97,008 |
| KvikIO `cuMemcpyHtoDAsync` | 370,000 | 370,000 |
| KvikIO `cuStreamSynchronize` | 370,000 | 370,000 |

The raw reader task counts are unchanged. B adds only 4,060,584 H2D bytes relative
to A over the complete capture (including differences in setup tables and batch
records), not another copy of the raw payload. A has one 512-byte D2H transfer
for its first-dgram legacy layout inference; B has **no D2H transfers** in the
timed capture. Device field parsing did not reintroduce a host payload round trip.

The trace's first-to-last GPU-operation span was 22.56 s (A) versus 39.22 s (B),
while the union of observed kernel/copy/memset intervals covered 19.92 s versus
23.05 s. The much larger uncovered intervals in B are consistent with increased
host dispatch overhead. These are **trace diagnostics, not a measurement of SM
utilization or a promised recoverable speedup**. Profiling amplified B's elapsed
time: traced loop rates were 416.5/246.3 events/s, excluded from clean throughput.
CUDA API totals from KvikIO's eight worker threads also overlap; summing their
waiting times would greatly exceed application wall time.

**Trace-quality caveat:** Nsight's SQLite diagnostics include possible incomplete
CUDA/NVTX collection warnings in both captures. The expected timed-work counts
match exactly: calibration 10,000 in each; gathers 50,000/320,000; B walks 500;
B field locators/status fills 96,000 each; raw read tasks 370,000 each. The
corresponding main NVTX ranges also have their expected counts. Nevertheless,
the coverage/gap figures above should remain qualitative diagnostics, not an
exact idle-time budget. The warnings are preserved in each trace-summary JSON
under `collector_diagnostics`; they are not silently discarded.

## Recommended next work

The data supports **batched field location and batched gathering**, not an
initial rewrite of the XTC walker:

1. Batch locator initialization and field decoding across configured handles.
   This is a contained parser change targeting the 5.18 s host scope and
   96,000-per-operation launch/event pattern. Keep metadata device-resident.
2. Batch canonical gathering across segments/events within the **existing
   execution subbatch**. Its roughly 6.17 s incremental host scope is the
   largest individual measured delta; device gather time also increased.
3. Cache Configure-derived bindings and avoid redundant same-stream waits in
   the calibration submission path, while preserving cross-stream dependencies
   and public field-access correctness. Some of this naturally belongs in the
   batched-gather change rather than a separate micro-optimization.
4. Revisit initialization/missing-row cleanup after the larger changes. Its
   GPU cost is measurable, but it is a smaller target and must preserve missing/
   rejected-field semantics; simply removing zeroing is not safe.

Field location first remains a reasonable contained implementation stage, even
though gathering has the slightly larger measured host delta. Verify each
change independently with correctness checks and clean warm A/B measurements.
Do not add the host and device costs above as a speedup forecast. None of these
optimizations has been implemented by this measurement task, and batch size,
memory admission, slot ownership, leases, and backpressure remain unchanged.

## Completion and artifacts

Job **38478311 completed with exit 0**, elapsed 48m17s. Accepted results comprise
six clean runs, six CPU/NVTX-timed runs, and two separate Nsight captures, plus
the four successful CPU-reference preflights. The focused harness suite has
**21 passing tests**; Python compilation and shell syntax checks pass.
No production psana files or historical A/B sources were edited. The changes
are opt-in validation instrumentation and this report; nothing was committed
or pushed by this measurement task.

Local artifacts are under `validation/perf-acceptance-20260916/job-38478311-warm-ab/`:

- `results.json`, `cpu-summary.json`: accepted samples and per-scope medians.
- `provenance.json`, `stage.json`, per-case logs/GPU CSVs: input, build, runtime,
  cache, and hardware evidence.
- `A-trace-cpu-nvtx-r0.nsys-rep`, `B-trace-cpu-nvtx-r0.nsys-rep`: native timelines.
- Matching `.sqlite` files and `A-trace-summary.json` / `B-trace-summary.json`:
  kernel/API counts, device times, interval coverage, and collector diagnostics.
- `A-stats_*.csv`, `B-stats_*.csv`: NVIDIA's NVTX-correlated kernel, CUDA API,
  and NVTX-range summaries.

The new helpers are `phase_timing.py`, `run_warm_ab.py`, `run_warm_ab.sbatch`,
`trace_rank.py`, `summarize_warm_ab.py`, and `trace_summary.py` in the validation
directory, with four focused test files. Existing `bench.py` gained the opt-in
flags and bounded warm-cache preparation; its default timing mode remains off.
