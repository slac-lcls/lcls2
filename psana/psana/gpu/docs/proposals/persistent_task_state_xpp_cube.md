# Persistent GPU task state for XPP cubes

**Proposed, not implemented.** Reviewed against checkout `87bce00cb` on
2026-10-28. API names below are a design sketch, not executable examples for the
current release. The implemented interfaces remain in [user kernels](../user_kernels.md)
and [current design](../design.md).

**Discussion update — 2026-11-02:** Keep
[PR #172's Design 1](https://github.com/slac-lcls/lcls2/pull/172): different
steps may overlap across BDs and across execution slots within one BD, with
one step per execution subbatch/occupied slot. This supersedes this sketch's
local BD step-boundary drain. Retaining multiple XPP bins is an open memory
and lifecycle discussion, not a decision to restrict overlap or add a drain
option. Follow #172's payload scope too: GPU task context carries run/step
identity, not scan/step payload values or SlowUpdate values. Those values stay
on the ordered CPU path. The lifecycle API below remains proposed.

## Purpose

Support image sums that survive many GPU task invocations, with explicit
initialization, finalization, memory accounting, and result delivery. The first
application is an XPP cube:

- Step scan: one scan step identifies one bin. Accumulate its images, export
  the completed partial bin, and reuse storage for the next step.
- Fly scan: events belonging to a bin can occur anywhere in the run. Accumulate
  several bins, then export final bins or mergeable partial bins.

The original pseudocode used a global `pixel_array_sum` and proposed
`beginrun`/`endrun` methods. The desired feature is **run-owned, per-stream task
state**. Python module globals and Python destructor timing do not provide the
required ownership or completion rules.

The historical XPP workload was `xpp101605526`, run 216, with `jungfrau1M`
panels of shape `(2, 512, 1024)`. It calibrated with `cmpars=(7, 0, 10)`, applied
a threshold of 5, and accumulated both panel arrays and assembled images,
together with counts. It used ordinary BD distribution, not bin-directed
`set_destination`. This proposal supports that accumulation pattern; it does
not claim scientific equivalence or a measured throughput for a new GPU cube.
The existing [GPU calibration example](../../examples/jungfrau_calibration.py)
does not implement common-mode correction.

## Fit with the current implementation

| Existing mechanism | Extension needed |
| --- | --- |
| Host-only `GpuTask(function, inputs, calibconst)` declaration | Optional factory for a stateful implementation |
| One callback per nonempty selected execution subbatch | Preserve batching; invoke `process(batch, stream)` |
| Stable CUDA execution streams in `EventPool` | One task instance per stream and run instance |
| Input leases and producer completion events | Retain persistent state through submitted work and aggregate export |
| Per-event `batch.publish()` and automatic D2H | Separate keyed step/run aggregate delivery |
| Framework allocation ledger | Charge declared persistent state and reserve execution/export headroom |
| Transition dispatch and pipeline close | Ordered lifecycle hooks and finalization exactly once |

Relevant source: [task declaration](../../gpu_task.py),
[execution pool](../../gpu_stream.py), [producer dispatch](../../gpu_producer.py),
[event manager](../../gpu_events.py), and [allocation ownership](../../gpu_allocation.py).

A callable object can already retain application-owned arrays between calls.
Today the same callable is invoked on different streams, its arrays are outside
framework accounting, and there are no automatic task lifecycle hooks. Such a
prototype is possible, especially with one stream, but it is not this managed
state feature.

## Ownership and concurrency

State identity is:

```text
(run instance, task identity, BD worker, CUDA execution stream)
```

Use a run-instance identity, not just the numerical run number: reopening a
run must not reuse an old accumulator. Here “stream” means a CUDA execution
stream, not an XTC file stream.

```text
Run
  BD 0
    CUDA stream A -> task instance A -> sums and counts
    CUDA stream B -> task instance B -> sums and counts
    combine contributions with matching keys -> BD partial bins
  BD 1
    ...
  writer/application -> combine BD partial bins -> final cube
```

Instances survive execution-slot retirement. Borrowed input arrays, batch
contexts, and raw pointers do not become persistent merely because the task
does. Never save them for a later invocation. Task state owns its allocations
independently of input windows and slot leases.

All instance methods that enqueue GPU work run on the assigned current stream.
Updates to one instance are stream ordered. Different instances have disjoint
mutable arrays; no cross-stream atomics are needed for this ownership scheme.
An accumulation kernel still needs a correct reduction or atomic strategy when
multiple threads in the *same invocation* contribute to the same bin/pixel.

A shared accumulator per BD is an alternative requiring explicit synchronization
or suitable atomic kernels. It is not the initial design. The existing CUDA-IPC
sharing of immutable calibration constants must not be reused as an implicit
sharing mechanism for mutable sums.

## Proposed declaration and lifecycle

Preserve existing `GpuTask(function=...)` behavior. Add a mutually exclusive
`factory=...` form. The factory constructs a fresh instance for each execution
stream, on the assigned GPU worker. Configuration and factory construction are
host-only; device allocation starts in `begin_run`.

Provide a proposed `GpuTaskWorker` base class for user implementations.
`process` is abstract and required; lifecycle hooks default to no-ops so users
override only the methods their analysis needs. These are Python overridable
methods, analogous to virtual methods. The framework owns hook ordering and
completion; inheriting the base class alone does not provide those guarantees.
Keep `GpuTask` as the host-only declaration and `GpuTaskWorker` as the mutable
worker implementation.

```python
from abc import ABC, abstractmethod

class GpuTaskWorker(ABC):
    def begin_run(self, ctx, stream):
        pass

    def begin_step(self, ctx, stream):
        pass

    @abstractmethod
    def process(self, batch, stream):
        ...

    def end_step(self, ctx, stream):
        pass

    def end_run(self, ctx, stream):
        pass

    def close(self):
        pass
```

The name and export location remain proposed. Constructors store host-only
configuration, allocation starts in `begin_run`, and `close` is idempotent
reference cleanup after GPU work and export readers finish. Required science
output belongs in finalization hooks, never in a Python destructor.

The proposed declaration also specifies aggregate names/merge operations and
a persistent-state byte limit per stream. The following fields are all new:

```python
from functools import partial

task = GpuTask(
    factory=partial(XppCube, mode="step", binning=binning, science=science),
    inputs=["jungfrau1M.raw", *binning.input_selectors],
    calibconst=science.constant_selectors,
    state_bytes_per_stream=state_capacity_bytes,
    aggregate_outputs={
        "cube": {"pixel_array_sum": "sum", "count": "sum"},
    },
    aggregate_sink=local_cube_writer,
)
# ds = DataSource(..., gpu_fn=task, n_gpu_streams=2)
```

`binning` and `science` are host-only configurations for user algorithms. Each
instance must create its own mutable algorithm state; only immutable
configuration may be shared. Detector and constant selection retain today's
validation rules. `aggregate_sink` is a host consumer on each BD (or the serial
worker), not a callback automatically run on MPI rank zero.

| Method | Invocation and guarantee |
| --- | --- |
| `__init__(...)` | Host configuration only; no CUDA allocation |
| `begin_run(ctx, stream)` | Once per instance, after placement and input metadata are available |
| `begin_step(ctx, stream)` | Correct ordered step identity is available; no scan/step payload values |
| `process(batch, stream)` | Once per nonempty execution subbatch; borrowed inputs retain existing contracts |
| `end_step(ctx, stream)` | All admitted contributions for this instance's step have completed; export is allowed |
| `end_run(ctx, stream)` | All admitted run contributions have completed; export remaining state exactly once |
| `close()` | Idempotent reference cleanup, after all GPU work/export readers complete; no science output or MPI |

The lifecycle context supplies an immutable run/step identity, termination
status, a budgeted state allocator, and aggregate emission. It does not expose
scan/step payload values, SlowUpdate values, or the live mutable Run/EnvStore.
Use the shared source-step identity from #172, qualified by run identity
and based on the BeginStep timestamp; a BD-local generation is not a cross-BD
bin key. The process batch exposes the same run instance and step identity as
its captured scope, plus its existing aligned inputs and timestamps. A stable
stream ID may be exposed for diagnostics; applications should not key state by
a raw CUDA pointer.

Initial input shape/layout metadata remains available for allocation; it is
not changing scan/step payload state. Requested calibration constants are
initialized once per run and remain unchanged across steps. The CPU application
associates exported step identities with physical scan/bin values separately.

Instances with no events still participate in lifecycle completion. Their
initialized sums/counts are zero; the reducer can omit their numerical payload
while recording completion. Shapes must therefore come from validated input
metadata, not solely from the first event.

## Refined pseudocode

This version implements one accumulator bank per stream. For step scans the
bank contains the current step's label bins. For fly scans it contains all
configured `(bin, label)` combinations and must fit the declared state budget.
The bounded-cache extension for larger cubes is described below.

```python
class XppCube(GpuTaskWorker):
    def __init__(self, mode, binning, science):
        # Host-only configuration; one new instance per run/BD/CUDA stream.
        self.mode = mode
        self.binning_config = binning
        self.science_config = science
        self.sums = self.counts = None
        self.active_step = None

    def begin_run(self, ctx, stream):
        self.binning = self.binning_config.make_worker(ctx)
        self.science = self.science_config.make_worker(ctx)
        event_shape = ctx.input_spec("jungfrau1M.raw").event_shape
        # Includes label combinations; one label gives one bank row per bin.
        self.keys = self.binning.bank_keys(mode=self.mode)
        self.sums = ctx.state.zeros(
            (len(self.keys), *event_shape), dtype="float64")
        self.counts = ctx.state.zeros((len(self.keys),), dtype="uint64")

    def begin_step(self, ctx, stream):
        if self.mode == "step":
            assert self.active_step is None
            self.active_step = ctx.step_id
            # Scope identity only; no scan/step payload values enter the task.
            # Previous end_step exported and cleared the bank.

    def process(self, batch, stream):
        if self.mode == "step":
            assert batch.step_id == self.active_step

        # User science: calibration, common mode if required, threshold,
        # masks, and event selection. Scratch uses admitted workspace and
        # is retained before kernels are launched, through batch completion.
        images, valid_pixels, accepted_events = self.science.prepare(
            batch, stream=stream)
        # Step mode selects label rows within this step's bank using declared
        # event inputs/static configuration; step_id keys the exported bin.
        # Fly mode assigns scientific bins from declared aligned event fields.
        bin_rows = self.binning.assign(batch, stream=stream)

        # Updates persistent state IN PLACE on the supplied stream.
        # Invalid bin rows are excluded. No event image is published/copied.
        accumulate_by_bin(
            images, bin_rows, valid_pixels, accepted_events,
            out_sums=self.sums, out_counts=self.counts, stream=stream)

    def _export_and_clear(self, ctx, stream):
        # emit snapshots each keyed record with bounded host staging.
        # Baseline contract: all reads from these device arrays finish
        # before emit returns. The runtime records source/flush identity.
        for row, key in enumerate(self.keys):
            ctx.emit(
                "cube",
                key=ctx.bin_key(step_id=self.active_step, bin_label=key),
                pixel_array_sum=self.sums[row],
                count=self.counts[row:row + 1],
            )
        # Safe because export no longer reads either allocation.
        self.sums.fill(0)
        self.counts.fill(0)

    def end_step(self, ctx, stream):
        if self.mode == "step":
            assert ctx.step_id == self.active_step
            self._export_and_clear(ctx, stream)
            self.active_step = None

    def end_run(self, ctx, stream):
        if self.mode == "fly":
            self._export_and_clear(ctx, stream)
        # The controller finalizes an open step before calling end_run,
        # including a truncated step marked incomplete. Never export it twice.

    def close(self):
        # Runtime has already joined method work and export consumers.
        # Runtime-owned allocation charges survive until storage is released.
        self.sums = self.counts = None
        self.science = self.binning = None
        self.active_step = None
```

`make_worker`, `prepare`, `assign`, and `accumulate_by_bin` represent application
algorithms, not proposed built-in Jungfrau calibration/binning routines. Their
memory requirements must be included in admission. `ctx.state.zeros`,
`ctx.input_spec`, `ctx.bin_key`, `ctx.emit`, and `batch.step_id` are proposed
framework interfaces. The current `batch.step_generation` is not sufficient
until ordering is corrected.

For event `e`, bank row `b`, and pixel `p`, the kernel contract is:

```text
sum[b,p] += images[e,p] when accepted[e], bin_rows[e] == b, valid_pixels[e,p]
count[b] += 1          when accepted[e], bin_rows[e] == b
```

The event count increments once per event, not once per pixel. Define the
acceptance policy for missing segments and damaged data explicitly. A scalar
event count does not describe pixel-dependent exposure; applications requiring
pixelwise means must also accumulate valid-pixel counts with the same shape as
the sum. Bin edges, out-of-range/NaN exclusion, labels, and segment order must be
identical across BDs. Preserve sums and counts; do not merge partial averages.

For a single bin with all images valid, the operation is conceptually:

```python
# images: (N, segments, rows, columns); sum_images: (segments, rows, columns)
partial = workspace.empty(sum_images.shape, dtype=sum_images.dtype)
cp.sum(images, axis=0, dtype=sum_images.dtype, out=partial)
sum_images += partial
```

Here `workspace` denotes admitted scratch retained through producer completion.
`cp.sum(images, out=sum_images)` would overwrite the accumulator and, without
`axis=0`, reduce pixels as well. `cp.array((x, y, z))` creates three values;
`zeros((x, y, z), dtype=...)` creates the intended image storage. Packed Jungfrau
raw words include gain bits: summing raw uint16 words is not calibrated science.

## Transition ordering and step completion

This feature depends on the ordered transition work tracked by
[#153](https://github.com/slac-lcls/lcls2/issues/153). In the inspected source,
the EB may append BeginStep or EndRun after L1 events in a packet. The manager
currently calls `_handle_steps()` before submitting that packet's events.
Adding lifecycle hooks at that point would finalize too early or assign the
wrong step. EndStep itself currently does not require a drain.

Plan execution and transitions in source order. For the sketch's single-bank
worker, order its own lifecycle as follows; this is not a barrier across the
BD's streams:

```text
events for step A -> complete their contributions -> end_step(A)
                 -> begin_step(B) -> events for step B
last admitted run events -> complete -> end_step if open -> end_run -> close
```

Keep one step per execution subbatch. Discover compact event ranges and scope
boundaries during existing packet traversal, then intersect them with memory
admission ranges. Do not add per-field scheduling, repeated descriptor scans,
or rebuild bulk-read plans for each step. Input read grouping remains distinct
from execution boundaries.

Keep PR #172's overlapping-step baseline. Different execution slots within a
BD may hold different steps, and BDs advance independently. One execution
subbatch/occupied slot contains only one step; slot reuse waits for its
consumers. BeginStep/EndStep alone do not impose a blanket BD drain or a global
barrier. Persistent task state has a separate lifetime from slot storage.

For the single-bank sketch, a worker must finish exporting its old scope before
reusing that bank for the next scope. Other workers may still contribute to
the old step. The BD reducer retains that step's completion identity until all
participating streams finish. The exact hook scheduling and completion protocol
remain to be designed; replayed history and empty steps must not duplicate
output or hooks.

Supporting these hooks does not itself implement public `run.steps()` and
`step.events()`. Existing GPU step-iteration limitations remain until the
ordered controller supplies those adapters. Internal hooks must work while
users consume `run.events()`.

### Open discussion: XPP bin retention with overlapping steps

For a step scan with one bin per step, stream A can still accumulate bin A
while stream B accumulates bin B. Exported partials for A may also wait in the
reducer. Thus a BD can have several distinct bins live at once. Even with
step-boundary draining, one accumulator per stream already means multiple
arrays; they would simply contribute to the same local step/bin.

Overlap need not multiply GPU accumulator storage beyond one bank per stream
if export completes before each bank is reused. However, pending partials,
merge buffers, staging, completion metadata, and sink queues also consume
memory. A fixed stream/slot count alone does not bound retained output across
successive steps. For XPP's `(2, 512, 1024)` panels, each float64 image sum is
8 MiB, before counts, labels, assembled images, scratch, or export storage.

Discuss with the transition-design work:

- When can a worker export/reuse its bank while other workers are still on
  that step, and how is final step completion recorded?
- What byte limits cover live GPU banks, host partials, and sink retention?
- When those limits are reached, how should bounded partial export and
  resource backpressure preserve progress without a blanket step barrier?
- How should delayed streams/BDs and slow sinks be tested to demonstrate
  bounded retention and correct bin sums?

These are follow-up questions for the stateful task feature, not new approval
gates for #153/#172. No local-drain scheduling option is selected by this note.
Fly-scan bins can be interleaved throughout a run regardless of step overlap;
their dense-state or bounded-cache policy remains a separate requirement.

## Aggregate delivery and reduction

Persistent state is mutable working storage; aggregate output is a snapshot of
that storage. Do not call the state declaration `run_scope_outputs`: step and
periodic outputs may be produced from the same run-lifetime allocation.

`ctx.emit` has no event-row axis and does not attach results to an arbitrary
event. Each record contains run identity, logical bin/label, step identity when
applicable, array shape/dtype and segment identity, sums/counts, completion
status, and a runtime-assigned source/flush identity. Different partial flushes
for the same logical bin must be added, not overwritten. Runtime retries must
not redeliver an already accepted record; distributed crash recovery is outside
the initial scope.

The initial export path is deliberately bounded and synchronous with respect
to reading device source arrays. It stages one bin or bounded pixel tile at a
time, then permits reset/reuse only after that read finishes. Lifecycle methods
can enqueue clearing or other GPU work; the runtime records and joins their
completion too, even when there is no subsequent event callback.

Combine stream contributions within the BD before sending final step/run
partials to the writer where practical. A bounded host reducer is sufficient
initially; synchronize each source stream before its copy. It must track stream
completion independently of numerical records, including empty streams.
Charge its resident merge buffers and staging to an explicit host cap. Spill
mergeable partials to the sink when the cap is reached; never retain an
unbounded cube dictionary in a hidden output queue.

A later GPU merge can reduce one bin/tile at a time on a merge stream that waits
for all contributing producer events. That requires budgeted merge workspace
and the same export completion contract. Async snapshots require either an
immutable copy, buffer ownership transfer, or a completion token that gates
reuse; `keepalive` alone is not a reuse notification.

The host sink receives independent arrays plus metadata. It can write per-BD
partials or use an explicit application aggregation protocol. Any retained
arrays count against a bounded sink policy. No hidden `COMM_WORLD` collective
belongs in `end_step`, `end_run`, or `close`: other psana roles do not execute
these hooks, and BD early exits need not occur together. A collective reducer
requires an explicitly established participant communicator and completion
protocol, including workers with no contributions.

## Memory policy and large fly-scan cubes

For dense state, excluding small count/key arrays:

```text
state bytes per GPU = BDs sharing GPU * streams per BD
                    * resident bin/label combinations * pixels * sum itemsize
```

XPP's 1M panel stack needs 8 MiB per float64 sum. With 256 bins and two streams,
that is 4 GiB per BD before counts, additional labels, assembled images,
calibration constants, inputs, scratch, parser storage, and export workspace.
Ordinary event distribution can give every BD contributions to every bin.

Reserve each stream's declared state capacity during setup, before event
admission. Allocation through `ctx.state` consumes that reservation rather than
charging it twice. Actual backing allocations and aliases retain charges until
release. Validate the sum of all stream reservations against the per-BD budget
and preserve enough execution/export headroom to make progress. Fail with a
capacity diagnostic if the minimum working set cannot fit. Do not repeatedly
retry by shrinking batches when persistent state alone consumes the budget.

The baseline sketch uses a dense bank and rejects configurations that do not
fit. Periodically clearing that same dense bank does **not** reduce its memory
requirement. For larger fly-scan cubes, add a bounded resident-bin table:

1. Map selected events to scientific bin/label keys, preserving input ordering.
2. Partition work so each admitted execution's required resident keys fit, or
   process the execution in bounded key groups while retaining its input lease.
3. Before admitting a new key, export an existing key's mergeable partial sum
   and count, wait for its readers, and reuse the entry.
4. Permit an evicted key to reappear later. The sink adds all partials by key.

Track a flush sequence per source, and bound metadata, merge buffers, host
staging, and queued sink work as well as device arrays. The cache policy must
not introduce unbounded event-image retention or an event-by-event host loop.
Benchmark the additional grouping and export traffic separately from dense-bank
performance.

## Binning inputs and routing

For a step scan with one bin per step, use immutable step identity to separate
accumulation and key exported partials. The GPU task does not need the physical
scan value. CPU transition handling retains scan/step payload values; the
application can associate them with exported identities for labeling or
combining bins on the CPU. Scheduling look-ahead may export a partial before
public CPU delivery reaches its step, so this association must use the matching
recorded identity rather than the CPU's mutable current-step value. Any pending
association/output storage must obey the host/sink budget.

Fly binning needs a per-event value available before accumulation. Use declared
aligned device fields (for example a feespec field descriptor). Generic fields
are descriptors, not automatically dense arrays; kernels must validate
presence, locator status, types, and extents. A detector sharing an XTC stream
may require `hybrid_det`, with its existing duplicated-I/O cost. Static bin
edges and label definitions can be supplied through host configuration.

GPU EnvStore and extraction, retention, upload, or task-context exposure of
scan/step and SlowUpdate payload values are outside this proposal, as in #172.
An analysis requiring those values inside a GPU kernel needs a separately
designed input path; persistent state does not provide one. The GPU callback
runs before public event delivery, so a CPU event-loop calculation cannot
retroactively supply binning values to that callback.

Persistent state does not add `smd_callback`/`set_destination` support. Those
remain rejected with GPU routing in this checkout. Bin-directed ownership would
require coherent per-destination CPU/GPU packets and is a separate optimization.

## Termination and failure rules

- Normal EndRun: submit all preceding events, complete the final open step,
  export remaining run state, deliver accepted aggregate records, then close.
- EOF without EndRun, `max_events`, or explicit early close: finalize admitted
  work under a documented partial-result policy and mark affected scopes/run
  incomplete. Hooks must not imply that all recorded events were processed.
- A GPU task runs ahead of public event delivery. On early close its partial
  cube can include admitted events that the user loop has not consumed. Record
  contribution counts and scope status; do not describe it as the user's
  consumed-event prefix.
- Callback or GPU failure: stop admission, mark the result invalid, drain safely
  or retain/quarantine owners if completion cannot be proven. Do not export a
  partially updated sum as a successful result. Preserve the existing MPI fatal
  error policy; a lifecycle hook is not a recovery mechanism.
- Setup failure: close only instances/resources that were constructed, including
  partial allocations. No aggregate output is required from failed setup.
- Repeated cleanup or delivery failure: distinguish finalized, delivered, and
  released state so sums are not added twice. `__del__` performs no required
  export, synchronization, MPI, or lifecycle transition.

## Implementation sequence and acceptance

1. Establish ordered event/transition ranges and step identity. Test real EB
   output through the GPU manager, including trailing transitions and history.
2. Add factory instances, budgeted state allocation, and deterministic lifecycle
   calls without changing existing function tasks or no-task behavior.
3. Add bounded aggregate snapshots, completion records, and a host sink/reducer.
   Start with one stream, then combine multiple streams and multiple BDs.
4. Implement the application cube kernels and compare step/fly results with a
   CPU reference using matched calibration, masking, threshold, and bin policy.
5. Add bounded resident-bin eviction if dense fly-scan state does not fit;
   optimize GPU merging or async snapshots only after the ownership tests pass.

Required coverage includes empty runs/steps, missing detector segments, rejected
events, interleaved and skewed bins, duplicate bin IDs across partial flushes,
multiple subbatches per step, boundaries inside packets, final tails, multiple
runs, slow sink/export consumers, constrained budgets, partial initialization,
callback errors, and early/repeated close. Vary batch size and stream/BD counts:
counts must agree exactly, and floating sums must meet an explicit tolerance
because reduction order changes. Test label and pixel-validity policies too.

For psana core implementation, run the required CPU and longer MPI suites plus
GPU device and MPI lifecycle tests. This document adds no runtime behavior and
does not establish a cube performance result. Future measurements must report
input dataset, science policy, bin/label count, state size, batch/stream/BD/GPU
layout, I/O mode, cold/warm conditions, export traffic, and final drain time.
