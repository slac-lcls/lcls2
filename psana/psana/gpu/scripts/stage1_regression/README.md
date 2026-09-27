# Stage 1 and Stage 1b Jungfrau regression comparison

This benchmark extends `../jf_scaling` and reuses its independent SMD, pixel,
read-count, budget, cache-residency and completion gates. It does not modify the
production runtime. The maintained current scaling baseline is
[documented here](../../docs/performance/jungfrau_current_scaling.md).

Two matched comparisons use separate exclusive single-node allocations:

| Comparison | Before | After | Identical workload |
|---|---|---|---|
| Stage 1 extraction | `480f7074c` | `7d0b5941e` | Read, parse, gather and existing calibration; no output D2H |
| Stage 1b removal | `7d0b5941e` | `137e2902f` | Read, parse and requested dense raw gathering; no calibration or output publication |

Each sweeps **one A100 with 1, 2, 3 and 4 BDs**, bulk off/on, cold/warm, two
fresh-process repetitions with matched versions adjacent. Round two reverses
BD/cache/mode/version order. There are 64 timed samples plus 16 separate
200-event pixel/launch preflights per comparison. The timed samples use 10,000
JF-only events from `mfx101210926/r0387`, streams 005–009, batch 20, depth 1,
eight KvikIO workers per BD, 1 MiB tasks/target, automatic budgets and CPU-fallback
I/O. Each sample must read 335,571,760,000 useful bytes in 50,000 requests.

`input_adapter.py` requests the same `DenseInputPreparer.jungfrau_raw` on both
Stage 1b sides, without source calibration dictionaries. It installs only after
legacy MPI setup has seen an empty detector map. The old runtime needs a small
`process_batch` adapter returning no results; slot buffers remain owned until
normal retirement. New runtime uses its input-preparer map. Both versions
perform strict raw-shape validation and the same batched gather. This measures
requested dense input preparation, not the new default parse-only mode or a
public callback (which is not implemented yet).

`diagnostic.py` counts actual kernel calls and inclusive host submission/join
costs only in separate pixel preflights. Those runs include three deliberate
pixel transfers and are not throughput evidence. `profile.py`/`profile_rank.py`
add separate 200-event bulk-on Nsight captures at 1/4 BDs, with no pixel copies.
The NVTX steady marker begins after the first delivered event on each BD;
full traces include setup and must be filtered by that marker for steady counts. Nsight
captures are also excluded from throughput results. Interpret kernel counts
against the recorded workload/event distribution; host durations can overlap
GPU execution and each other.

Primary throughput is 10,000 / maximum rank event-loop seconds, matching the
existing scaling report. DataSource/run setup is recorded separately. Lazy
setup and BeginStep inside `run.events()` remain inside primary timing. An
additional first-delivery-to-end rate is descriptive and is not a synchronized
warmup across ranks. All timestamps, BD distributions, per-rank memory charges,
pinned row-map sizes, physical GPU monitoring, and network interface byte
counters remain in the artifacts. Accepted native traces provide complete CUDA API/copy
counts beyond what Python wrappers can observe.

Frozen roots contain `runtimes/{parent,stage1,stage1b}/python`, `scripts`,
`commits.json`, native dependency identity, constant/reference files and
`hashes.json`. No native source changed between these checkpoints. The runner
verifies the frozen manifest before/after execution and removes only its private
local stage. Cold residency must be below 1%; warm above 99% both before and
after timing. Local prefixes and cache policy match the current JF scaling
report; these results do not establish true-GDS or Weka throughput.

Example, in a prepared Slurm environment with the frozen script directories on
`PYTHONPATH`:

```bash
python scripts/stage1_regression/run.py --root FROZEN_ROOT --source JF_XTC_DIR --comparison stage1b
```

`--smoke` stages only the 200-event prefixes and runs preflights. Full jobs can
use `afterok` dependencies on successful smoke jobs. Do not edit a frozen root
while its jobs are queued/running; use a new attempt directory for corrections.

Use an Nsight installation with its bundled CUPTI libraries. The accepted
September 26 retry uses `/sdf/group/lcls/ds/tools/nsight-2025.3.1/bin/nsys`;
the local 2026.1.1 installation produced reports without CUDA records and was
rejected. Merely creating an `.nsys-rep` is not an acceptance check.

After all eight profile cases finish, extract full and steady native counts:

```bash
python analyze_profiles.py PROFILE_JOB_DIRECTORY --output COUNTS_JSON
```

The extractor requires CUDA kernel records and one completed
`psana.benchmark.steady` range per BD report. Steady counts select activities
whose start timestamp lies inside that BD's range, including worker-thread
CUDA calls. They are window counts, not attribution to individual events.
Durations are inclusive and can overlap; do not sum them into wall time.
The JSON preserves per-rank and aggregate kernel/API/copy counts, copy bytes,
window boundaries, and profile provenance. It rejects incomplete campaigns.

`collect.py` preserves the accepted paired throughput evidence and draws a
four-panel scaling figure. It requires both campaigns' `complete: true` markers
and all 64 timed samples plus 16 preflights per comparison. For example:

```bash
python collect.py --root FROZEN_ROOT --stage1-job 39192035 --stage1b-job 39192036 --output SCRATCH_REPORT.json
```

The JSON, SVG, and PNG share the output basename. Plot ranges show the two
repetitions, not confidence intervals. `summarize.py` can inspect partial jobs;
its provisional output must not be presented as completed acceptance.
