# User kernels and results

`GpuTask` is the implemented interface for external GPU algorithms on the
experiment/run path. Start with the executable example in the [README](../README.md).
Psana calls `function(batch, stream)` once per nonempty selected execution
subbatch. Public event iteration consumes results after that submission.

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
| `batch.run`, `batch.batch_id`, `batch.step_generation` | Host run/batch/step identifiers |
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

A constants array from `batch.calibconst()` must not be kept past the
callback that obtained it: read it, or copy it with `.get()`, and let the
reference go. When BD ranks share a GPU, one device allocation backs every
peer, so it is rewritten in place or freed at the next transition. See
[lifetime](device_placement_and_shared_constants.md#lifetime-valid-until-the-next-transition).

Generic fields need no dense adapter or CPU detector class. They preserve
physical segment identity and runtime shape in device descriptors. Configure
payload fields are not event selectors. Pointer-consuming kernels must check
source presence, locator status, type/rank and byte bounds before dereferencing.

Constant selectors address the run's calibration dictionary. Psana unwraps
`(array, metadata)` entries and uploads only declared native numeric NumPy
arrays, retaining original shape, dtype and values. Scalars/empty numeric
arrays are allowed; missing keys, unsupported values and undeclared access fail
explicitly. Gains are not implicitly inverted; offsets and masks are not
implicitly prepared. Empty declarations cause no task-constant upload.

Each BD owns a requested snapshot. BeginStep drains prior work and refreshes
changed requested arrays after the host transition; it does not fetch the DB.
Original calibration arrays retain their physical segment axis. Map dense rows
using `segment_ids` rather than assuming a dense row number is a physical ID.

Inputs/constants are read-only by contract. Psana cannot intercept arbitrary
native writes. To modify values, allocate and register an owned destination
before submitting the copy or kernel:

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
