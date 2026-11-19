# User kernels and results

`GpuTask` is the implemented interface for external GPU algorithms on the
experiment/run path. Start with the executable example in the [README](../README.md).
Psana calls `function(batch, stream)` once per nonempty selected execution
subbatch. Public event iteration consumes results after that submission.

Prefer this batched path for routine full-detector processing. Parsed per-event
field copy properties support inspection and snapshots, with a host transfer
on first CPU access and a device allocation/copy on each GPU-copy access; see [field copy costs and lifetime](gpu_xtc_parser.md#copy-property-costs-and-lifetime).
Published task outputs have the separate delivery contract described below.

## Declare dependencies

```python
from psana.gpu import GpuTask

task = GpuTask(
    function=my_analysis,
    inputs=["jungfrau.raw", ("jungfrau", "raw", "raw")],
    calibconst=[("jungfrau", "pedestals"), ("jungfrau", "pixel_gain")],
)
```

The string requests a dense input; the tuple requests generic device field
descriptors. Declare only what the algorithm needs. Declaration is host-only;
constructors of callable objects should defer CUDA work until the assigned
worker invokes them. `gpu_fn` accepts a `GpuTask`, not a bare callable. All
referenced detectors must be selected with `gpu_det` or `hybrid_det`.

| Callback access | Meaning |
| --- | --- |
| `batch.size` | Number of selected rows in this execution |
| `batch.timestamps`, `batch.batch_event_indices` | Host tuples preserving selected GPU order and original identity |
| `batch.timestamps_gpu`, `batch.batch_event_indices_gpu` | uint64 device arrays, uploaded lazily during the callback |
| `batch.run`, `batch.batch_id`, `batch.step_generation` | Host run/batch/step identifiers; `step_generation` is the step's index |
| `batch.step_scope` | `StepScope(run, index, begin_ts)` for every event in this execution: the step they were recorded in, identical on every BD (issue #153) |
| `batch.input("jungfrau.raw")` | Borrowed dense input, leading event dimension |
| `batch.present("jungfrau.raw")` | Borrowed `(N, S)` segment-presence array |
| `batch.segment_ids("jungfrau")` | Physical segment IDs corresponding to the dense/descriptor segment axis |
| `batch.field(detector, algorithm, field)` | Borrowed descriptor table; see [parser contract](gpu_xtc_parser.md#task-field-descriptors) |
| `batch.calibconst(detector, key)` | Declared original constant device array |
| `batch.keepalive(*owners)` | Retain user storage until execution consumers finish |
| `batch.publish(name, array, event_indices=None)` | Register named output rows and retain the array |

The context is valid only during the callback. Keep algorithm parameters and
compiled kernels in your callable if useful, but do not cache borrowed input,
constant or context handles for later execution. All submitted GPU work must
use the supplied stream; it is also CuPy's current stream during the call.

## Inputs and constants

The public dense adapter supports Jungfrau raw uint16 panels with shape
`(512, 1024)` or `(1, 512, 1024)`. It supplies `(N, S, 512, 1024)`, with dense
segment rows following the binding reported by `segment_ids`. The current
manager builds this binding from sorted Configure segment IDs. Missing or
rejected segments are zero-filled and marked absent. A valid field with a
non-`Corrupted` damage flag remains usable; presence does not mean damage-free.

A constants array from `batch.calibconst()` stays valid for the whole run,
including when BD ranks share a GPU: the device copy is uploaded once at setup
and released at teardown, never replaced mid-run. Keeping a reference across
events or steps is therefore safe; keeping one past the run is not, so copy
with `.get()` to outlive it. See
[lifetime](device_placement_and_shared_constants.md#lifetime-valid-for-the-whole-run).

Generic fields need no dense adapter or CPU detector class. They preserve
physical segment identity and runtime shape in device descriptors. Configure
payload fields are not event selectors. Pointer-consuming kernels must check
source presence, locator status, type/rank and byte bounds before dereferencing.
The actual `raw_ptr + offset` address may be unaligned: check its alignment
before typed loads, or use byte-safe loads within the validated bounds. Task
field descriptors reference packed input, not public field-view aligned copies.

Constant selectors address the run's calibration dictionary. Psana unwraps
`(array, metadata)` entries and uploads only declared native numeric NumPy
arrays, retaining original shape, dtype and values. Scalars/empty numeric
arrays are allowed; missing keys, unsupported values and undeclared access fail
explicitly. The host calibration XTC transport currently promotes scalar NumPy
arrays from shape `()` to `(1,)` before GPU setup; this inherited limitation is
tracked in [#247](https://github.com/slac-lcls/lcls2/issues/247). GPU upload
preserves the shape it receives. Gains are not implicitly inverted; offsets and
masks are not implicitly prepared. Empty declarations cause no task-constant upload.

Each BD owns a requested snapshot, uploaded once at run setup. psana does not
replace or refresh it during the run: BeginStep applies the host transition
but neither drains prior work (each subbatch carries its own `StepScope`) nor
reuploads constants, because it does not fetch the DB. **Mutating the host-side calibration dictionary after setup does not
update the GPU snapshot.** One constant set per run is the supported model;
dynamic mid-run updates are deferred (#168). Original calibration arrays retain
their physical segment axis. Map dense rows using `segment_ids` rather than
assuming a dense row number is a physical ID.

Inputs and constants are read-only **by contract, not by enforcement**. CuPy
arrays have no NumPy-style WRITEABLE flag and psana cannot intercept native
writes, so nothing stops a kernel writing into an array from
`batch.calibconst()` — it just corrupts the snapshot for every later batch and,
when constants are shared over CUDA IPC, for every peer rank on the GPU. Write
results into separate outputs. To modify values, allocate and register an owned
destination before submitting the copy or kernel:

```python
raw = batch.input("jungfrau.raw")
scratch = cp.empty_like(raw)
batch.keepalive(scratch)
cp.copyto(scratch, raw)
# Submit in-place work on scratch on the supplied stream.
```

## Publish and consume results

`publish(name, array)` maps the leading dimension to all N selected events.
`event_indices=[...]` maps M rows to distinct host integer indices in that
selected subbatch, not original batch indices. Scalars use shape `(M,)`; empty
rows can use `(M, 0, ...)`. Arrays must be native numeric, C-contiguous CuPy
arrays on the producer device. Each `(event, name)` is published at most once.
Detector/input names are reserved; output lookup uses the exact published name.

Register arrays before submitting work that writes them. Publication retains
storage but does not prove readiness; the producer and copy completion events
do that. Do not reuse a published buffer while its terminal copy is outstanding.
`keepalive` is retention, not a notification that a buffer is reusable.

Disjoint groups may share a name with different shapes/dtypes. Later subbatches
may introduce different names. Unpublished names are absent, with no placeholder
output allocation or transfer. Multiple groups cause multiple payload copies;
one callback is not a promise of one kernel or one copy.

`evt.gpu.get(name).on_cpu` returns an independent NumPy row, including a
zero-dimensional ndarray for a scalar. The first access waits if needed and
caches that row. Retained events preserve their output after slot reuse or run
cleanup. Task outputs do not expose `.on_gpu` or `.on_gpu_view()`; those APIs
remain available for parsed input fields during their event lifetime.

Output staging defaults to 64 MiB pinned host memory per BD. Set
`gpu_d2h_pinned_bytes` to change the cap; zero, oversized groups or exhausted
capacity use synchronous ordinary-host copies. Full-image batches can exceed
the cap even if one image fits. The cap excludes user device allocations,
metadata staging and retained NumPy results. See [memory ownership](design.md#memory-ownership-and-backpressure).

## Configuration and cleanup

| DataSource argument | Default | Effect |
| --- | --- | --- |
| `gpu_fn` | `None` | Task declaration; no task means no automatic science/output copies |
| `batch_size` | 20 with GPU routing; 1000 for CPU-only runs | Request event batching; tails/admission may shrink executions |
| `n_gpu_streams` | 2 | Number of execution slots |
| `gpu_bulk_read` | `True` | Group adjacent file ranges independently of callback batching |
| `gpu_bulk_target_bytes` | 1 MiB | Small-input grouping target |
| `gpu_memory_budget_gb` | 0 (automatic) | Per-BD framework device quota, in GiB |
| `gpu_d2h_pinned_bytes` | 64 MiB | Aggregate output pinned staging per BD |

The GPU default applies to both `gpu_det` and `hybrid_det`, with or without a
task. Explicit values override it. This is a DataSource-wide setting, so CPU
detectors in a mixed run share it. The latest JF staging and user-kernel scaling
campaigns used 20 explicitly. The best batch size depends on the kernel work,
scratch/output memory and GPU/BD layout. Good scaling requires investigating
these together; other values still need workload-specific validation. User
scratch/output memory remains outside framework accounting.

Use a positive explicit batch size for reproducible workloads. The legacy
`batch_size=0` fallback still resolves to one event, not the omitted default.
`batch_size=1` still runs the task, but requests one-event batches and changes upstream
batching too; it is not the same control as moving user work into the public
event loop. There is no separate batch-scheduling Boolean. Nonzero
`gpu_d2h_chunk_size` is rejected because automatic image delivery was removed.

Use `with closing(run.events())` for a break or loop-body exception. Closing a
started serial GPU event iterator is terminal for that run. MPI close drains
local work and outstanding EB messages without requesting a collective stop.
Fatal callback/pipeline errors abort the MPI communicator. The
[limitations](limitations.md) describe unsupported input modes and combinations.

## Stateful tasks

A `function` task keeps nothing between invocations. For accumulators that
must survive a run -- a per-step image sum, a histogram -- declare a
`factory` instead and derive from `GpuTaskWorker`:

```python
from functools import partial
from psana.gpu import GpuTask, GpuTaskWorker, state_bytes

class Cube(GpuTaskWorker):
    def __init__(self, detector):            # host-only configuration
        self.detector = detector

    def begin_run(self, ctx, stream):        # once per instance, before any event
        shape = ctx.input_spec(self.detector + '.raw').event_shape
        self.sums = ctx.state.zeros(shape, 'uint64')

    def begin_step(self, ctx, stream):       # ctx.step_scope names the step
        self.sums.fill(0)

    def process(self, batch, stream):        # what function(batch, stream) gets
        self.sums += batch.input(self.detector + '.raw').sum(axis=0, dtype='uint64')

    def end_step(self, ctx, stream):         # this stream's step is complete
        ctx.emit('cube', key='bank', sums=self.sums)

    def close(self):
        self.sums = None

task = GpuTask(factory=partial(Cube, 'jungfrau'), inputs=['jungfrau.raw'],
               state_bytes_per_stream=state_bytes(((8, 512, 1024), 'uint64')),
               aggregate_outputs={'cube': {'sums': 'sum'}}, aggregate_sink=records.append)
```

**Instances.** `factory()` is called once per CUDA execution stream at run
setup, after requested constants are uploaded and before any event is
admitted. Each instance owns its mutable arrays; only immutable configuration
is shared. A new run instance gets new instances. `ctx.run_instance` is
`RunInstance(expt, runnum, begin_run_ts, ordinal)`, identical on every BD.

**State.** `ctx.state.zeros/empty(shape, dtype)` allocates device memory
charged to `state_bytes_per_stream`, reserved at setup; the pool rounds every
array to 512 bytes, so compute the figure with `psana.gpu.state_bytes(...)`,
never by summing `nbytes`. Setup fails with a capacity diagnostic if the
reservation does not leave room for event admission. `ctx.input_spec(name)`
gives a dense input's per-event shape and dtype from configuration, so an
instance can size its state before it sees an event.

**Hooks.** Every method runs with `stream` as the current CUDA stream, so
work issued in a hook is ordered after this instance's earlier `process`
calls -- that stream order, not a drain or a barrier, is what makes
`end_step` safe to export from. Per stream:

| hook | when |
| --- | --- |
| `begin_run` | once, at setup |
| `begin_step` | before the first subbatch of a step on this stream; also for steps this stream received no events for (a `begin_step`/`end_step` pair, so every stream completes every step) |
| `process` | once per nonempty execution subbatch |
| `end_step` | after this stream's last work for the step, before the next step's first; exactly once per stream per step |
| `end_run` | after every admitted execution has retired; `ctx.run_status` is `"complete"` only if EndRun was reached, `"incomplete"` after `max_events` or an early close, `"failed"` (then `end_run` is skipped) after a `process` exception |
| `close` | always, after `end_run`; idempotent, no science output |

Scope 0 -- the interval before the first BeginStep -- gets a hook pair only
if some stream processed events in it, so a step-scan sink sees one record
per step. Hooks never see scan/step payload values, SlowUpdate values, the
EnvStore or a communicator.

**Export.** `ctx.emit(name, key, **arrays)` is allowed in `end_step` and
`end_run`. Each array is copied to the host on the instance's own stream and
that copy is waited for, so the array may be cleared or reused immediately;
nothing waits on the pool. `name` must be declared in `aggregate_outputs`
with an op per field (`"sum"`). Records with the same
`(run_instance, step_scope.begin_ts, key)` are merged by op in a per-BD
reducer, and `aggregate_sink(record)` is called **inline, on this BD** --
keep it cheap -- with one `AggregateRecord` per key once every stream has
finished the step. Completion is the framework's: a stream that emits
nothing still completes the step. `record.complete` is False for a step the
run ended inside, for the run record of a run that did not reach EndRun, and
for a partial delivered early by `aggregate_host_bytes`; partials for one key
carry increasing `seq` and must be added. No collective runs anywhere in
this path; combining BDs is the sink's job.

`examples/step_cube.py` is the complete version of the sketch above, matched
bit-exactly against a CPU reference on run 77 -- see
`validation/gpu-stateful-20261118/`.

## Science examples

[calibrate_jungfrau.py](../examples/calibrate_jungfrau.py) uses the external
[JungfrauCalibration](../examples/jungfrau_calibration.py) callable. Default
pedestal/gain arithmetic is validated against Jungfrau CPU-v3 calibration with
matching policy/constants. Gain codes 0/1/3 are handled; invalid gain code 2 and
missing pixels yield zero. Optional offset/status/status-extra policies request
those constants explicitly; missing requested keys are errors. This example
does not implement common-mode correction. Full calibrated images are float32.

[integrate_jungfrau.py](../examples/integrate_jungfrau.py) uses
[JungfrauAzimuthalIntegration](../examples/jungfrau_azimuthal_integration.py).
One callback launches calibration then integration, retaining the calibrated
image and validity scratch on-device and publishing float64 `(3, nbins)` rows:
mean, sum, contributing-pixel count. Valid zero-intensity pixels count. Bin IDs
are user-provided physical-layout integers; `-1` excludes a pixel.

Copy the driver and its algorithm modules into your own directory. The
integration driver requires an NPZ with `bin_ids` and strictly increasing finite
`edges`. Prepare geometry, beam parameters and bins outside the callback; a
fixed radial bin map is not automatically q-space calibration. Solid-angle,
polarization and pixel splitting are not implemented. At full 32-panel JF size,
image plus validity scratch costs 80 MiB/event, outside the framework quota.
The examples demonstrate the external-user boundary; their scientific policy
is not imposed by psana.
