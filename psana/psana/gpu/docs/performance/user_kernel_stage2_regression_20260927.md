# User-kernel Stage 2 performance regression

Status: accepted for proceeding to Stage 3 after the completed interleaved A/A
control and balanced A/B follow-up. No repeatable slowdown above the 5% gate
was observed; run-to-run variability prevents claiming a speedup or equivalence.

The focused comparison uses the accepted Stage 1b checkpoint `600669d15` and
Stage 2 `f5b4cfb0e`. Both run the existing read/parse/dense-raw preparation
benchmark with `gpu_fn=None`. The benchmark adapter requests identical dense
inputs on both versions; this is not callback execution or a default parse-only
measurement. Production runtime code is unchanged by the benchmark work.

## Jobs and artifacts

Frozen root:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage2-regression-20260927-r1`

| Job | Purpose | Result |
|---|---|---|
| 39273214 | Six pixel/launch preflights | Passed |
| 39273215 | Matched throughput comparison | Completed, exit 0; 24 timed samples and six preflights |
| 39273216 | Constant setup/refresh characterization | Completed, exit 0; five repetitions at 1/2/4 workers |

`commits.json`, `native-source.txt`, and `hashes.json` pin 2,441 frozen files,
including runtime/native dependencies, scripts, and reference data. Generated
outputs are `job-JOBID/` and `constants-JOBID/`, with scheduler logs at the root.
The full run repeats the six preflights before timing and verifies its manifest
before/after execution. Native binaries are inherited from the validated Stage
1b installation; neither compared commit changes native sources.

## Matched throughput contract

Reuse the [Stage 1/1b benchmark](../../scripts/stage1_regression/README.md) and
its independent timestamp, pixel, byte/read-count, budget, cache-residency,
GPU-assignment, and retirement checks. The focused matrix is:

- One A100, 1/2/4 BDs, bulk on, warm cache: three alternating-order pairs per
  configuration, 18 timed samples total.
- One A100, four BDs, bulk on, cold cache: three alternating-order pairs,
  six timed samples total.
- Six separate 200-event pixel/launch preflights, excluded from throughput.

Each timed process uses 10,000 JF-only events from `mfx101210926/r0387`,
streams 005–009, batch 20, depth 1, eight KvikIO workers per BD, 1 MiB tasks and
bulk target, automatic per-BD budgets, and CPU-fallback I/O. The expected useful
payload is 335,571,760,000 bytes in 50,000 requests. Inputs are copied into a
private node-local stage. Warm/cold residency and NUMA-interleaved warming follow
the accepted Stage 1b contract. Comparisons share one exclusive node allocation.

Primary throughput is events divided by maximum rank loop time. Preserve
DataSource/run setup, first-delivery latency, the descriptive post-first-delivery
rate, host/device memory information, and per-rank charges separately. A
repeatable slowdown above 5% triggers investigation; three pairs do not prove
statistical equivalence. Expand the matrix or profile only if results warrant it.

## New constant costs

`constant_cost.py` runs five repetitions each with 1/2/4 concurrent MPI workers
sharing one assigned GPU. Only the Stage 2 constant store is measured; Stage 1b
has no corresponding requested-constant feature. Cases are empty setup, gain-only
setup, unchanged refresh, and changed refresh. The saved run-387 gain value has
shape `(3, 32, 512, 1024)`, dtype float32, and size 192 MiB per worker. Other
calibration keys are absent from the measured source dictionary.

CUDA context creation, loading the frozen dictionary, and correctness copies
are outside timing. First allocation and allocator-warmed repetitions remain
separate in JSON. Record rank wall time, current/high-water host RSS, upload
counts/bytes, and peak owned-plus-held budget. Empty and unchanged cases must
upload zero arrays; setup and changed refresh must upload exactly the gain value.
Close must return all constant charges to zero after each repetition.

Consumers are already idle/drained. This measures neither database access,
input-cache trimming, transition-drain latency, nor end-to-end task setup.
Callback throughput remains outside scope until Stages 3–4 are executable.

## Harness checks

33 standalone harness tests passed, including the new focused-matrix order and
coverage checks. They also confirm that the original 64-sample full matrix is
unchanged. The first invocation from the unbuilt source package failed import
collection; the standalone script copy avoids loading unbuilt psana extensions.

## Initial results and the control follow-up

The initial throughput job completed in 56 min 5 s on sdfampere027, with all
24 timed samples, six preflights, final manifest verification, and private-stage
cleanup complete. [Compact results](user_kernel_stage2_results_20260927.json)
preserve per-sample timing, read waits, cache residency, allocation charges,
preflight launch counts, and constant-cost measurements.

| Configuration | Stage 1b median events/s | Stage 2 median events/s | Change |
|---|---:|---:|---:|
| Warm, 1 BD | 388.35 | 356.50 | -8.20% |
| Warm, 2 BDs | 499.03 | 495.32 | -0.74% |
| Warm, 4 BDs | 501.37 | 524.06 | +4.53% |
| Cold, 4 BDs | 176.40 | 176.34 | -0.04% |

Warm 1-BD paired loop-time increases were 7.07, 0.40, and 2.80 seconds;
read-completion wait increases were 7.54, 0.41, and 3.36 seconds. First-delivery
latencies stayed around 0.3 seconds and measured cache residency was 100%.
This locates the observed delay in completion waits without identifying its
cause. Stage 2 changed neither KvikIO's submission/completion implementation nor
its fallback payload-copy path. Requested-constant code is bypassed here because
`gpu_fn=None`. This initial finding motivated the control below.

Constant-store measurements were stable across 1/2/4 concurrent workers: roughly
250 ms for 192-MiB gain staging, 260 ms for an unchanged-source scan without H2D,
and 515 ms for changed-source comparison and upload. Committed storage was
192 MiB per worker; replacement peaked at 384 MiB. These are separate feature
costs and do not explain the input-only loop difference.

Follow-up job **39298015** uses
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage2-controls-20260927-r1`.
It runs one GPU and one BD, warm cache, bulk on, with the same 10,000-event,
batch-20/depth-1 workload, input reference, environment, and runtime snapshots.
Four pixel preflights precede 24 timed samples in a single exclusive allocation:
six A/A pairs and six A/B pairs, with an even number of alternating-order rounds.

- `control_a` and `control_b` both alias the exact Stage 1b installation at
  `600669d15`; the runner verifies resolved path and commit equality.
- `stage1b` uses that same installation; `stage2` uses `f5b4cfb0e`.
- Odd rounds run A/A then A/B; even rounds run B/A then the reversed A/A labels.
  Each side therefore runs first three times, and pair order is balanced too.
- Reuse the same per-sample checks and record loop, first-delivery, setup,
  read-completion waits, and allocation charges. Do not interpret short
  diagnostic preflight rates as throughput.

The original node was occupied at submission, so the control may use another
healthy A100 node. Both comparisons share their new allocation; conclusions use
within-allocation pairs rather than comparing absolute rates across nodes.
The control harness passed 34 standalone tests and its frozen manifest/alias
checks before submission. No production runtime changed.

## Completed control and acceptance

Job **39298015** completed with exit **0:0** in **1:05:18** on sdfampere014.
All 24 timed samples and four pixel preflights passed. Final manifest verification
and private node-local stage removal completed (`complete=true`). The
[compact control evidence](user_kernel_stage2_controls_20260927.json) preserves
all sample timings, paired changes, per-rank counts/charges, launch preflights,
source identities, and log/manifest hashes.

| Comparison | A median loop | B median loop | B rate change |
|---|---:|---:|---:|
| Identical Stage 1b code, A/A | 37.634 s | 37.652 s | -0.05% |
| Stage 1b / Stage 2, A/B | 37.731 s | 36.104 s | +4.51% |

Rates use 10,000 events divided by median loop seconds: Stage 1b 265.03 events/s
and Stage 2 276.98 events/s. The Stage 2 median loop is 1.628 seconds shorter.
This excludes DataSource/run setup and includes first-delivery work in the loop.

Individual A/B paired rate changes were -0.65%, -11.28%, +1.02%, +5.36%,
+36.03%, and +39.75% (median paired change +3.19%). A/A paired changes ranged
from -2.02% to +4.21%. The last round also sped up substantially for both
identical-code control labels; late fast runs show much shorter read-completion
waits. Even the first four balanced rounds give a median-loop rate change of
about -0.97%, below the investigation threshold. These observations do not
establish the cause of timing variability and do not support attributing the
positive aggregate result to Stage 2 code.

Every timed sample delivered the expected 10,000 events, 335,571,760,000 useful
bytes and 50,000 requests, with zero CPU BD payload reads, 100% measured cache
residency before/after, and identical peak owned-plus-held storage of
1,344,113,152 bytes. Each preflight checked three pixel samples and recorded
ten launches each for walk, locator initialization, field location, and gather.

Decision: the initial warm 1-BD slowdown did not reproduce as a consistent
Stage 2 penalty, and the other initial configurations already passed the gate.
Accept Stage 2 for continued development and proceed to producer dispatch and
owner retention. Preserve the timing evidence and rerun callback-path performance
once the integrated publication/delivery path exists; this acceptance does not
cover that future workload.
