# Stage 1 and Stage 1b Jungfrau regression comparison

This benchmark extends `../jf_scaling` and reuses its independent SMD, pixel,
read-count, budget, cache-residency and completion gates. It does not modify the
production runtime. The maintained current scaling baseline is
[documented here](https://github.com/slac-lcls/lcls2/blob/fa40ec52a/psana/psana/gpu/docs/performance/jungfrau_current_scaling.md).

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

## Stage 2 focused regression

`--comparison stage2` compares frozen `stage1b` and `stage2` runtime directories.
Both use the same benchmark-only dense-input request with `gpu_fn=None`;
callback execution remains unavailable. The stage names in `commits.json` must
identify the exact commits used. For the focused matrix:

```bash
python run.py --root FROZEN_ROOT --source JF_XTC_DIR --comparison stage2 \
  --bds 1 2 4 --modes on --caches warm cold --cold-bds 4 --repetitions 3
```

This runs six 200-event pixel/launch preflights, 18 warm samples, and six cold
samples. `--smoke` runs only the six preflights. Separate the setup/first-delivery
and descriptive post-first-delivery metrics from primary loop throughput.

`constant_cost.py` measures the Stage 2 constant store separately. Run five
repetitions with 1, 2 and 4 MPI processes sharing their one assigned GPU, using
`PS_PARALLEL=none` and the Stage 2 runtime on `PYTHONPATH`. It measures empty setup,
gain-only setup, unchanged-source scans, and changed-source uploads. Loading the
frozen dictionary, CUDA context initialization, and validation copies are outside
timing. Record the first repetition separately from allocator-warmed repeats.
The test has idle/drained consumers and does not measure event-pool draining,
input-cache trimming, database access, or end-to-end DataSource setup. Concurrent
rank wall times, host RSS, upload bytes, and budget peaks are retained in JSON.

For the 1-BD A/A control and balanced A/B follow-up, use
`--comparison stage2-control --bds 1 --caches warm --modes on --repetitions 6`.
The frozen runtime aliases `control_a`, `control_b`, and `stage1b` must resolve
to the same installation and commit; `stage2` keeps the candidate installation.
The runner validates those identities before staging data and requires an even
repetition count. Each odd round runs `control_a, control_b, stage1b, stage2`;
each even round reverses that order. This interleaves six A/A pairs with six A/B
pairs in one allocation, balancing both within-pair order and which pair runs
first. Four pixel preflights precede the 24 timed samples. Runtime and per-sample
measurement code remain identical to the initial Stage 2 campaign.

## Stage 3 controlled regression

`--comparison stage3` compares `stage2` and `stage3`, with `control_a` and
`control_b` aliasing the exact Stage 2 installation and commit. A/A runs only
at warm 1 BD; other points run A/B only. Use an even number of rounds:

```bash
python run.py --root FROZEN_ROOT --source JF_XTC_DIR --comparison stage3 \
  --bds 1 2 4 --modes on --caches warm cold --cold-bds 4 --repetitions 4
```

This gives 40 timed samples: 32 A/B samples across four configurations, plus
eight A/A samples at warm 1 BD. Eight separate pixel/launch preflights run first.
All existing cache, byte-count, identity, budget, manifest and cleanup checks
apply. Both versions use `gpu_fn=None` with identical benchmark dense preparation.

`callback_cost.py` measures Stage 3 internally using the synthetic GPU acceptance
fixture (900 uint16 pixels per event). It compares no task, an empty callback,
registered scratch plus one kernel, and a scalar publication plus one kernel.
The current harness uses one callback per execution subbatch. Scratch covers
the entire event dimension and scalar outputs have shape `(N,)`; each nonempty
scratch/publication callback launches one user kernel. Historical September 27
per-event results use their frozen script and are not measurements of this API.
Batch sizes 1/20 and depths 1/2 run six alternating-order rounds. Compilation,
parsing and initial preparation are outside timing; each submission still queues
one dense gather. Fresh window facades reference immutable fixture storage and
exercise normal leases without accumulating dependencies across repeated uses.

Separate preflights check values, callback selection, publication sizes and one
gather with zero repeated parser launches. Timed measurements separate submit,
retire/drain and total loop time. This synthetic check includes no disk I/O,
automatic publication D2H, or host delivery, and cannot predict full Jungfrau
callback throughput. Its memory counters are post-drain values, not peaks.
