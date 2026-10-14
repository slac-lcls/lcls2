# Stream-read refactor Stage 3: scheduling integration

2026-09-24. Stage 2 reviewed without blocking findings and committed as
`b24eed98c` (`Own stream read groups independently and poll CUDA completion safely`).
The Stage 3 changes described here are based on that commit.

## Runtime changes

- Production bulk-on builds `GroupReadSchedule` from the complete EB packet and
  resolved file/transition epochs. Small adjacent dgrams coalesce within the
  1 MiB target and transition fences; large dgrams stay separate requests.
  Requests are submitted in first-event/stream order.
- Execution ranges honor byte admission and split before a stream's next small
  group. This prevents an execution from waiting for a new small group while
  retaining the planned uses that keep its predecessor alive.
- The run-scoped `InputGroupPool` holds small-stream credits across EB batches.
  A submission reserves the actual rounded raw/parser/detector allocation growth
  before I/O. Pressure recovery drains delivered executions, then unreferenced
  CUDA inputs, and trims only free caches. Retained consumers remain protected.
- EventPool forks already reserved input uses into execution and delivery
  leases. Success, early close, read cleanup and normal shutdown use the same
  group ownership path. Input reclamation polls all groups independently.
- The parser accepts a device table of per-dgram raw pointers, byte bounds and
  group-local row indices. A set of independent reads shares **one walk, one
  locator initialization, and one configured-field lookup launch**. There is no
  raw-payload concatenation or extra raw-device copy.
- Each group exposes its own raw buffer and parser-row slices. The shared
  parser arena remains leased until all its groups finish; each group's raw
  backing can return earlier. Locator views preserve the arena's handle stride,
  which the existing batched gather uses for correct addressing.
- Bulk-off retains its read scheduling and uses a null per-dgram pointer table
  with the same parser kernels. Legacy residency helpers remain for comparison
  fixtures until Stage 4 replaces their failure/transition coverage.

The CPU admission estimate conservatively reserves space for parser arenas
retained by small groups. Exact allocation holds remain authoritative. This
estimate and the Python scheduling overhead are candidates for later tuning.

## Verification

**375 CPU tests passed**, including the new scheduling/controller tests and all
existing GPU unit tests. Run with `PS_PARALLEL=mpi` for the MPI transport test;
`PS_PARALLEL=none` causes that test's module to omit `MPI`.

**26 GPU tests passed**, job **39020519**, `sdfampere010`, A100-SXM4-40GB:

- new production scheduler pixel checks for both small-input fixture sizes;
- exactly three parser launches per multi-group parse, independent of reads;
- existing locator error/status, tail/reuse and lazy-access checks;
- existing multi-owner raw/calibration/passthrough gather checks;
- delayed-consumer input reuse and partial gather-upload failure checks.

Two subsequent failure-path hardenings preserve cleanup across a release error:
shared arena callbacks tolerate retries, and all transferred planned uses are
released even if an earlier release raises. The final CPU suite and real-data
run used these changes. Broader injected multi-group parser failures and
retained-view/transition/tight-budget acceptance remain Stage 4.

## Real JF+feespec validation

Job **39020858**, `sdfampere012`, one A100, 16 CPUs, account `lcls:data`, normal
QoS. Dataset `mfx101210926`, run 387; existing Weka FFB private six-stream data:

`/sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-feespec-bulk-38995226/xtc`

Each mode ran 200 warmup events with full feespec-array checks and three frozen
JF CPU raw/calibrated reference samples, then 1,000 measured events with
timestamp and feespec-sum digests. Batch size 100, execution depth 1, GPU budget
8 GiB, D2H chunk 0, KvikIO CPU fallback, eight workers, 1 MiB task size. The
existing benchmark-only routing override makes feespec's shared s000 exclusive
to the GPU. Its unrelated detector owners are not validated by this test.

| Cold validation | Bulk off | New bulk on |
|---|---:|---:|
| Events | 1,000 | 1,000 |
| Loop seconds | 8.139 | 9.552 |
| Events/s | 122.86 | 104.69 |
| KvikIO requests | 6,000 | 5,019 |
| Requested/useful bytes | 33,566,911,424 | 33,566,911,424 |
| Walk launches | 20 | 29 |
| Locator initialization launches | 20 | 29 |
| Configured-field lookup launches | 20 | 29 |
| JF gather launches | 20 | 29 |
| Prefix cache residency before | 0% | 0% |
| Physical NIC RX, decimal GB | 36.546 | 37.117 |

Bulk-on matches the Stage 1 plan: **19 feespec bulk requests plus 5,000 one-event
JF requests**. Small-group boundaries add nine execution/parser batches over
bulk-off. B++ batching is preserved within each batch; total launches are not
identical to bulk-off. Feespec sum kernels are per-event user work and are not
included in the parser/JF gather counters above.

Both modes pass the reference checks, byte totals and cold network-volume
gate. This single instrumented cold allocation is validation evidence, with
bulk-on taking 17.4% longer here. It does not establish a new performance
baseline or prove five-file worker overlap. Native file concurrency tracing,
controlled cold/warm rounds and longer-run acceptance remain Stage 5. Server
caches were not flushed; file tier placement was not re-audited in this smoke
job (it uses the previously verified Weka FFB staging).

The first smoke attempt, job **39020741**, passed the bulk-off warmup but the
harness rejected bounded-prefix warm preparation before measurement. The
retry used its supported cold mode and verified eviction. That failed attempt
contributed no accepted measured result.

## Artifacts and remaining work

### Review before commit

The review found one cleanup gap in partial multi-group setup: raw leases were
released, but already-constructed child windows could retain parser/raw aliases
through shared-owner cycles until Python garbage collection. Failure cleanup
now explicitly drains and detaches those children. If that cleanup itself
fails, the parser quarantines all remaining ownership for a later close/retry.

Review GPU job **39027254** on `sdfampere010` passed **29 tests**, including
injected second-window construction failure, a failed cleanup followed by
retry, and delayed-consumer buffer reuse with the new shared parser arena.
Artifacts: `/sdf/scratch/users/m/monarin/gpu-validation/stream-read-review-stage3-20260924/`.
No remaining blocking review finding. The broader cleanup inventory remains
in `stream_read_refactor_cleanup.md`.

The user requested that the next phase proceed as **Stage 4 performance
acceptance**. That phase compares previous/current bulk off/on in two cold and
two warm rounds on one allocation, with current cold native traces. Earlier
references to Stage 5 performance describe the original stage numbering.

### Stage 3 evidence

Scratch root:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-schedule-stage3-20260924/`

- `cpu.xml`, `device.xml`, `job-39020519.log`: CPU/device correctness evidence.
- `E-off-smoke-cold.log`, `E-on-smoke-cold.log`, `job-39020858.log`:
  real-data checks, timing, cache evidence and launch counters.
- `run.sbatch`, `smoke-cold.sbatch`, `smoke.py`: test launches and counters.
- `device-source-hashes.json`: overlay hashes for the initial GPU tests.
- `provenance.json`, `results.json`, `source/`, `stage3.patch`: final source and
  accepted real-data result snapshot.

Native dependencies use the frozen Integrated install at
`/sdf/scratch/users/m/monarin/gpu-validation/a-bpp-off-20260923/installs/Integrated`
(E runtime `ac87a93b2`), with Stage 3 GPU Python modules in the scratch overlay.

See `stream_read_refactor_cleanup.md` for legacy residency branches/tests,
duplicate read-plan construction, timing hooks, diagnostic wording and the
existing conservative field-view leases. Keep those cleanup gates explicit
while completing Stage 4 correctness acceptance and Stage 5 performance work.
