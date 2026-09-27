# User-kernel Stage 2 performance regression

Status: submitted September 27, 2026; no throughput conclusion yet.

The focused comparison uses the accepted Stage 1b checkpoint `600669d15` and
Stage 2 `f5b4cfb0e`. Both run the existing read/parse/dense-raw preparation
benchmark with `gpu_fn=None`. The benchmark adapter requests identical dense
inputs on both versions; this is not callback execution or a default parse-only
measurement. Production runtime code is unchanged by the benchmark work.

## Jobs and artifacts

Frozen root:
`/sdf/scratch/users/m/monarin/gpu-validation/jf-stage2-regression-20260927-r1`

| Job | Purpose | Submission state |
|---|---|---|
| 39273214 | Six pixel/launch preflights | Submitted |
| 39273215 | Matched throughput comparison | Depends on successful 39273214 |
| 39273216 | Constant setup/refresh characterization | Submitted independently |

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
