# psana2 GPU pipeline

Psana reads selected detector streams into GPU memory, parses XTC on the device,
and runs user algorithms inside the pipeline. A host-only `GpuTask` declares
inputs and calibration constants. Psana invokes it once per selected execution
subbatch and delivers its named outputs through ordinary `psana.Event` objects.
The algorithm can live in your driver script or an external module.

## Quick start: a batched user kernel

This example reduces each Jungfrau raw frame to a uint64 pixel sum. It sums raw
ADC words, including gain bits; it is a scheduling example, not calibration.

```python
from contextlib import closing
from psana import DataSource
from psana.gpu import GpuTask


def pixel_sum(batch, stream):
    import cupy as cp  # CUDA is initialized only on the worker
    result = cp.empty(batch.size, dtype=cp.uint64)
    batch.publish("pixel_sum", result)  # retain before submitting work
    cp.sum(batch.input("jungfrau.raw"), axis=(1, 2, 3),
           dtype=cp.uint64, out=result)


ds = DataSource(
    exp="mfx100848724", run=51, dir="/sdf/data/lcls/ds/prj/public01/xtc",
    detectors=["jungfrau"], gpu_det="jungfrau",
    gpu_fn=GpuTask(pixel_sum, inputs=["jungfrau.raw"]),
    # GPU default is 20. Tune for your kernel work and memory needs;
    # good scaling requires investigating batch size on your GPU/BD layout.
    batch_size=20,
    max_events=100,
)
for run in ds.runs():
    with closing(run.events()) as events:
        for evt in events:
            print(evt.timestamp, evt.gpu.get("pixel_sum").on_cpu)
```

The callback runs on the supplied CUDA stream before public event delivery.
`batch_size` defaults to **20 with GPU routing** (`gpu_det` or `hybrid_det`),
with or without a `GpuTask`, and **1000 for CPU-only runs**. An explicit value
overrides the default. This setting applies to the entire DataSource, including
CPU detectors in mixed runs; memory admission and tails can produce smaller GPU
execution subbatches. The latest JF staging and user-kernel scaling campaigns
used 20 explicitly. The best batch size depends on the kernel work, scratch/output
memory and GPU/BD layout; good scaling requires investigating these together.
Other values still need workload-specific validation; 20 is a starting point,
not a universal optimum. `gpu_bulk_read` controls file-read grouping
independently. Omitting `gpu_fn` stages/parses inputs without automatic
calibration or output copies.

The same experiment/run interface supports serial and MPI execution. MPI GPU
initialization belongs to BD workers; keep CUDA imports/allocation out of module
initialization. Use the site's built psana/CuPy/KvikIO environment. BD workers
sharing a GPU are discovered by device identity rather than EB-group
arithmetic, so multi-EB accounting is
[resolved](docs/limitations.md#multi-eb-device-accounting); see
[device placement and shared constants](docs/device_placement_and_shared_constants.md).

## Read next

| Document | Purpose |
| --- | --- |
| [Current design](docs/design.md) | Serial/MPI flow, read groups, batching, ownership, budgets and cleanup |
| [User kernels](docs/user_kernels.md) | Inputs, constants, scratch, publication, results and external science examples |
| [GPU XTC parser](docs/gpu_xtc_parser.md) | Configure tables, device locators and segment-preserving field access |
| [Limits and open work](docs/limitations.md) | Supported scope, configuration restrictions and unresolved issues |
| [Read/staging performance](docs/performance/read_staging.md) | Latest full JF and partial JF+feespec cold/warm, bulk off/on matrices |
| [User-kernel performance](docs/performance/user_kernels.md) | Matched scheduling comparison and calibration/integration scaling |
| [CPU/GPU complexity](docs/complexity.md) | Measured code size and responsibility comparison |
| [Device placement and shared constants](docs/device_placement_and_shared_constants.md) | Peer discovery by device identity, per-device budgets and CUDA-IPC constant sharing |

For calibrated images use [calibrate_jungfrau.py](examples/calibrate_jungfrau.py).
For calibration followed by radial integration in one callback use
[integrate_jungfrau.py](examples/integrate_jungfrau.py). Their user-owned algorithm
modules have no psana imports. See the [example requirements](docs/user_kernels.md#science-examples)
before choosing calibration policy or bin geometry.

## Validation and documentation policy

The accepted runtime and science examples are recorded at `6ba5fa586`; acceptance
and documentation were completed through `fa40ec52a`. The suites passed 608 main
CPU tests, 5 longer MPI tests, 173 A100 tests, and 18 public MPI cases, including
four expected callback-error aborts. Details and job IDs are in
[design validation](docs/design.md#validation) and the
[acceptance manifest](docs/performance/evidence/acceptance.json).

Current design and API pages describe the source, not development stages. The
performance pages identify their measured revisions and workloads; those rates
are not universal guarantees. Completed stage plans, handoffs and older reports
live in [Git history](https://github.com/slac-lcls/lcls2/tree/fa40ec52a/psana/psana/gpu/docs).
New measurements replace the relevant current report while preserving compact
provenance. [AMI integration](docs/proposals/ami_integration.md) remains a separate
proposal, not an implemented interface.
