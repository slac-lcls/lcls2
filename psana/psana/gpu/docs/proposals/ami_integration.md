# AMI + PSANA2 GPU Integration Design

**Status:** Proposed; retained for evaluation, not an implementation commitment.

This document describes a possible design for GPU-accelerated detector data
processing in LCLS-II AMI (Analysis Monitoring Interface) via the psana2 GPU
pipeline. All AMI types, nodes, and shmem GPU classes named below are proposed.
The current psana boundary was rechecked while organizing these documents;
line numbers and code sketches are illustrative rather than patch instructions.
Current implementation constraints are maintained in
[Known problems and limitations](../known_issues.md).

The GPU pipeline handles two data modes transparently:

- **Normal mode** (`drp_class='raw'`): bigdata contains raw uint16 ADC pixels;
  `GPUDetector` applies pedestals + gain calibration via `fused_calib_gpu`, yielding
  float32 calib output.
- **Passthrough mode** (`drp_class='fex'`): bigdata already contains float32
  pre-calibrated pixels written by the DRP; `GPUDetector` skips the calibration
  kernel and reshapes the data directly.  No calibration constants are loaded.

For supported file and MPI paths, both modes expose the same API:
`run.events()` yields a plain
`psana.Event`, and GPU results hang off `evt.gpu`, so
`evt.gpu.get("det.calib").on_gpu` returns a float32 CuPy array in either case.
GPU-native consumption should configure `gpu_d2h_chunk_size=0` explicitly.

---

## Background: how the CPU path works today

AMI workers run psana2 **in-process** as a single-process consumer.  The data
source is either a POSIX shared-memory ring (online) or XTC2 files (offline):

```
ONLINE
  DAQ/DRP ──POSIX shmem ring──► psana2 ShmemDataSource
                                    │  det.raw.calib(evt) → numpy
                                    ▼
                                 AMI Worker._process(evt)
                                    │  {"jungfrau:raw:calib": ndarray}
                                    ▼
                                 AMI graph → ZMQ → Collectors → GUI

OFFLINE
  XTC2 files ──► psana2 SerialDataSource / SingleFileDataSource
                     │  det.raw.calib(evt) → numpy
                     ▼  (same from here)
```

There is no IPC between psana2 and AMI; psana2 is called directly as a library
inside the AMI worker process.

---

## The fundamental gap

The psana2 GPU event pipeline (`GpuEventManager`) currently runs in the MPI BD
path and in `RunSerial` for normal experiment/run file input. It does not run
in `RunShmem`, and AMI does not currently publish `evt.gpu` values into its
type system or graph:

```
exp/run file input ─► RunSerial ─► GpuEventManager ─► Event(evt.gpu)
MPI   ──► BD rank   ──► GpuEventManager ──► Event(evt.gpu)
shmem ─► RunShmem  ──► Events (CPU only today)
```

AMI workers do not participate in psana2's MPI fan-out. They call a
single-process file or shmem data source. There are two integration gaps:

| Gap | Description |
|---|---|
| **Gap 1** | AMI source/type/graph code does not discover or carry current `evt.gpu` results; explicit `DataSource(files=...)` also selects `RunSingleFile`, which rejects GPU routing |
| **Gap 2** | `RunShmem` has no GPU staging path; DAQ writes complete XTC dgrams to CPU POSIX shmem |

---

## Three-phase design

### Phase 1 — GPU processing in the serial experiment/run path

`RunSerial` already wires `GpuEventManager` when GPU routing is enabled. This
is the normal `DataSource(exp=..., run=..., dir=...)` serial path. Explicit
`DataSource(files=...)` uses `RunSingleFile` and currently rejects `gpu_det`
and `hybrid_det`, so an AMI file source must either use the experiment/run path
or add equivalent GPU support. Both normal raw-to-calib and pre-calibrated
passthrough modes are implemented in `RunSerial`.

**Data flow:**

```
XTC2 files ──SMDReaderManager──► GpuEventManager
                 │  smdr_man.next_with_gpu() → (cpu_smd_batch, GPUBAT1 bytes)
                 │
                 ├── GpuBatchView (GPUBAT1 parser)
                 │     └── iter_read_descs → (stream_id, bd_offset, bd_size)
                 │
                 ├── KvikioGpuReader.issue_batch()
                 │     └── kvikio.CuFile(xtc_file).pread(offset) → GPU VRAM (async)
                 │
                 ├── EventManager(smd_batch) → cpu_evts (concurrent with reads)
                 │
                  └── EventPool.submit(subbatch, gpu_read, cpu_evts)
                            │  GPUDetector.process_batch() → calib_gpu
                            │    (normal mode: uint16 → fused_calib_gpu → float32)
                            │    (passthrough: float32 reshaped directly, no kernel)
                            ▼
                      Event  ← yielded to AMI worker (a plain psana.Event)
                        evt.gpu.get("jungfrau.calib").on_gpu → cp.ndarray float32
                        evt.gpu.get("jungfrau.calib").on_cpu → np.ndarray float32
```

**Available GPU keys per event:**

| Key | Available? | Notes |
|---|---|---|
| `{det}.calib` | ✓ always | float32, normal and passthrough mode |
| `{det}.image` | ✗ currently | Geometry helpers exist, but `GPUDetector.process_batch()` does not publish this key |
| `{det}.raw` | ✓ normal mode only | uint16 canonical raw; absent in passthrough |

> **Raw data is exposed in normal mode.**
>
> `GPUDetector.process_batch` gathers the canonical uint16 raw array into a
> per-slot buffer and sets `EventContext.raw_gpu`; `gpu_stream.py`
> propagates it as `{det_name}.raw`.  Read it exactly like calib:
>
> ```python
> raw = evt.gpu.get("jungfrau.raw").on_gpu   # cp.ndarray uint16
> ```
>
> `tests/gpu/integration/test_pixel_exact.py` compares this array against
> `det.raw.raw(evt)` from a separate CPU-only reference run, for every event,
> so the canonical segment ordering is covered.
>
> Note that within a GPU run the ordinary `det.raw.raw(evt)` accessor returns
> **None** for a GPU-routed detector rather than raising: splitting removes
> those streams from the CPU batch, so `DetectorImpl._segments()` finds no
> entry in `evt._det_segments` and `AreaDetector.raw()` propagates the None.
> Raw for a GPU-routed detector must therefore be read through `evt.gpu`.
>
> **In passthrough mode** (`drp_class='fex'`) raw data is **structurally
> unavailable**: the DRP applied calibration and never wrote raw ADC values to
> bigdata, so `raw_gpu` is None and the `{det}.raw` key is simply absent.
> This is a DAQ-level constraint, not a psana2 limitation.
>
> **Does AMI need raw data?**  Standard AMI graph nodes (ROI, peak-finding,
> binning, projection) operate on calibrated float32 arrays.  Raw ADU values
> are occasionally useful — noise statistics, gain-mode debugging, pre-pedestal
> hit-finding — but are not a typical online-monitoring use case.

**AMI worker call:**

```python
# ami/data.py  PsanaSource
ds = psana.DataSource(
    exp="mfx100848724",
    run=51,
    dir="/path/to/xtc",
    gpu_det="jungfrau",
    n_gpu_streams=2,
)
for run in ds.runs():
    for evt in run.events():          # yields a plain psana.Event
        calib = evt.gpu.get("jungfrau.calib").on_cpu
        ...
```

---

### Phase 2 — GPU processing in the shmem path (online AMI)

The shmem ring delivers **complete XTC2 datagrams** — bigdata already embedded
in CPU DRAM.  The four components of the files GPU path that assume file-based
I/O are all invalid here:

| Component | Why invalid in shmem mode |
|---|---|
| `smdr_man.next_with_gpu()` | No SMDReaderManager; shmem yields complete dgrams, no GPUBAT1 |
| `GpuBatchView` / GPUBAT1 | No EventBuilder in the shmem path; EventBuilder is an EB-rank concern |
| `KvikioGpuReader.issue_batch()` | Calls `kvikio.CuFile(path).pread(offset)` — no files; data is in CPU DRAM |
| `_split_subbatches(gpu_view)` | Splitting based on GPUBAT1 desc_table; no desc_table in shmem |

The shmem path needs a different input adapter, but it should preserve the
current parser and detector contracts. It can copy each complete selected XTC
dgram to a reusable device slot and build the small descriptor rows expected by
`GpuXtcBatchPool`; it should not parse detector pixels on the CPU:

```
Shmem ring (CPU DRAM, complete XTC2)
    │
    │  selected complete dgram bytes
    ▼
cudaMemcpyAsync(CPU shmem -> reusable GPU input slot)
    ▼
GpuXtcBatchPool.parse()
    │  GPU XTC walk and Configure-derived field locators
    ▼
GPUDetector.process_batch()
    │  normal mode:     applies pedestals + gain → float32
    │  passthrough mode: gathers float32 directly, no calibration kernel
    ▼
Event + evt.gpu                       ← identical API to Phase 1
```

Start with one event per shmem submission to minimize monitoring latency. A
small micro-batch may still improve H2D or kernel efficiency, so its
latency/throughput tradeoff should be measured rather than declared to have no
benefit.

**What remains valid in shmem GPU mode:**

| Component | Valid? | Notes |
|---|---|---|
| `GpuXtcBatchPool` / `GpuEventDgrams` | ✓ reuse | Preserve GPU parsing and general field access |
| `GPUDetector.process_batch()` | ✓ reuse | Existing locator-driven calibration/passthrough entry point |
| `EventPool` | ✓ optional | Overlaps processing of event N with H→D of event N+1 |
| `_D2hPipeline` | ✓ optional | Async D→H hides transfer behind next event |
| `_GpuBudget` | ✓ with known gaps | Reuse slot accounting; fixed allocations still need full accounting |
| `evt.gpu` (`GpuEventState`) / `GPUResult` | ✓ | Unchanged API |
| `on_gpu`, `on_gpu_view`, `on_cpu` | ✓ | Unchanged |
| `KvikioGpuReader` | ✗ | Needs file handles + byte offsets |
| `GpuBatchView` / GPUBAT1 | ✗ | No EventBuilder in shmem path |
| `_split_subbatches` | ✗ as written | Needs a shmem-specific byte admission unit |

---

### Phase 3 — GPU-native AMI graph nodes (CuPy throughout)

In Phases 1 and 2, `evt.gpu.get(...).on_cpu` is called to convert CuPy arrays
to numpy before they enter the AMI graph.  Phase 3 allows AMI graph nodes to
operate on CuPy arrays directly, with D→H deferred to the explicit `GpuToHost`
node.

```
Event (evt.gpu)
    │  evt.gpu.get("jungfrau.calib").on_gpu → cp.ndarray  (stays on GPU)
    ▼
AMI Worker graph (CuPy-aware nodes)
    │
    ├── GpuROI           cp.ndarray[y0:y1, x0:x1]  → cp.ndarray   (GPU)
    ├── GpuPeakFinder    cp.ndarray                 → int (hits)   (GPU scalar)
    ├── GpuToHost        cp.ndarray.get()           → np.ndarray   ← D→H here
    └── Binning          np.histogram(...)          → (bins,counts)
    │
    ▼
ResultStore  (holds np.ndarray — already on CPU after GpuToHost)
    ▼
ZMQ PUSH → NodeCollector → GlobalCollector → Manager → GUI
```

The `GpuToHost` node is the explicit D→H gate.  The AMI type system enforces it:
`GpuArray2d` cannot connect to an `Array2d` input terminal without a
`GpuToHost` node between them.

---

## Changes required in psana2

### 1. `RunShmem` — add GPU path  (`run.py:553`)

`RunShmem.__init__` currently always creates a CPU `Events` iterator.  Mirror
`RunSerial`'s pattern:

```python
# run.py:559  RunShmem.__init__  (add GPU branch)
if self.dsparms.gpu_enabled:
    # Shmem GPU path: data arrives as complete XTC2 dgrams.
    # No KvikioGpuReader / GPUBAT1 — uses ShmemGpuBatchAdapter.
    from psana.gpu.gpu_shmem_events import GpuShmemEvents
    self._evt_iter = GpuShmemEvents(
        configs, dm, self.dsparms, self, smdr_man=smdr_man
    )
else:
    self._evt_iter = Events(configs, dm, ...)
```

### 2. `GpuShmemEvents` — new class  (`psana/gpu/gpu_shmem_events.py`)

A GPU event loop for the shmem path. It replaces file offsets and KvikIO with a
direct H2D input adapter while reusing Configure tables, GPU XTC parsing,
detector bindings, EventPool, and result lifetimes:

```python
class GpuShmemEvents:
    """GPU event processing for the POSIX shared-memory (online) path.

    For each admitted event or small micro-batch:
      1. Copy selected complete XTC dgram bytes from shmem into a budgeted
         reusable device input slot.
      2. Build event/stream/offset/size descriptor metadata.
      3. Run GpuXtcBatchPool.parse() and construct GpuEventDgrams.
      4. Run existing GPUDetector.process_batch() adapters.
      5. Publish EventEnvelope/GpuEventState using EventPool leases.
    """

    def submit(self, event_envelopes):
        slot = self.event_pool.next_slot_id
        gpu_input = self.shmem_input.copy_to_slot(event_envelopes, slot)
        parsed = self.xtc_pool.parse(
            slot, gpu_input.data_gpu, gpu_input.desc_table, gpu_input.stream
        )
        event_dgrams = GpuEventDgrams.from_descriptors(event_envelopes, parsed)
        return self.event_pool.submit_parsed(
            event_envelopes, event_dgrams, self.gpu_detectors, slot
        )
```

The method names in this sketch are intentionally new: current `EventPool`
submission is coupled to `GpuBatchView`/KvikIO records, so extracting a shared
parsed-input submission boundary is part of the work. There is no existing
`GPUDetector.calibrate(det_gpu)` single-event API.

### 3. `GpuEventManager._next_batch()` — improve error message  (`gpu_events.py:886`)

```python
# Current: silent StopIteration
def _next_batch(self):
    if self.smdr_man is None:
        raise StopIteration

# Better: explain why and what to do
def _next_batch(self):
    if self.smdr_man is None:
        raise RuntimeError(
            "GpuEventManager requires an SMDReaderManager (smdr_man). "
            "For DataSource(exp=..., run=..., gpu_det=...) this is set automatically. "
            "For DataSource(shmem=..., gpu_det=...) use GpuShmemEvents instead."
        )
```

### 4. `GpuEventManager.gpu_detinfo` — new property for AMI type discovery

AMI's `_update()` inspects `inspect.signature(det.raw.calib)` to find the
return type annotation.  For GPU detectors this returns `numpy.ndarray` (CPU
path annotation), but AMI needs `GpuArray3d`.  Expose a `gpu_detinfo` dict
that AMI can read instead:

```python
# gpu_events.py  (new property)
@property
def gpu_detinfo(self) -> dict:
    """Return {det_name: {attr: amitypes_type}} for AMI type discovery.

    AMI's PsanaSource._update() reads this to register GPU detector names
    with GpuArray3d types instead of Array3d, enabling GPU-native flowchart
    nodes and correct type-checking between connected terminals.
    """
    try:
        import amitypes as at
        calib_type = at.GpuArray3d
    except ImportError:
        import cupy as cp
        calib_type = type(cp.empty(0))   # cp.ndarray as fallback

    return {
        name: {"calib": calib_type, "raw": calib_type}
        for name in self.gpu_det_names
    }
```

Forward it from `Run`:

```python
# run.py  Run base class (new property)
@property
def gpu_detinfo(self) -> dict:
    """GPU detector type map for AMI integration; empty for CPU-only runs."""
    if hasattr(self._evt_iter, "gpu_detinfo"):
        return self._evt_iter.gpu_detinfo
    return {}
```

### Summary of psana2 changes

| File | Change | Lines |
|---|---|---|
| `psana/psexp/run.py` | `RunShmem.__init__`: add `gpu_det` branch routing to `GpuShmemEvents` | ~10 |
| `psana/psexp/run.py` | `Run.gpu_detinfo`: new property forwarding to `GpuEventManager.gpu_detinfo` | ~8 |
| `psana/gpu/gpu_events.py` | `_next_batch()`: replace silent `StopIteration` with `RuntimeError` | ~6 |
| `psana/gpu/gpu_events.py` | `gpu_detinfo`: new property returning `{det: {calib: GpuArray3d}}` | ~12 |
| `psana/gpu/gpu_shmem_events.py` | **New file**: `GpuShmemEvents` — H→D copy + single-event GPU processing (calibration or passthrough) | ~100 |

---

## Changes required in AMI

### 1. `amitypes/array.py` — GPU array type tokens

The existing `Array2d` metaclass checks `isinstance(x, numpy.ndarray)` — CuPy
arrays fail it.  Three new types needed:

```python
# amitypes/array.py  (new additions)
import cupy as cp   # or lazy import

class GpuArray1dMeta(ArrayMeta):
    @classmethod
    def __instancecheck__(cls, inst) -> bool:
        return isinstance(inst, cp.ndarray) and inst.ndim == 1

class GpuArray2dMeta(ArrayMeta):
    @classmethod
    def __instancecheck__(cls, inst) -> bool:
        return isinstance(inst, cp.ndarray) and inst.ndim == 2

class GpuArray3dMeta(ArrayMeta):
    @classmethod
    def __instancecheck__(cls, inst) -> bool:
        return isinstance(inst, cp.ndarray) and inst.ndim == 3

class GpuArray1d(metaclass=GpuArray1dMeta): pass
class GpuArray2d(metaclass=GpuArray2dMeta): pass
class GpuArray3d(metaclass=GpuArray3dMeta): pass

GpuArray = typing.Union[GpuArray3d, GpuArray2d, GpuArray1d]
```

The `checkType()` machinery in `Terminal.py` runs mypy on stubs — it will
validate GPU→GPU and GPU→CPU connections automatically.

### 2. `ami/data.py:1053` — add GPU DataSource kwargs

```python
# ami/data.py  PsanaSource.ds_keys  (add GPU params)
self.ds_keys = [
    "exp", "dir", "files", "shmem", ...,   # existing
    "gpu_det",           # str | list[str]: detector name(s) to GPU-calibrate
    "n_gpu_streams",     # int: EventPool depth (default 2); ignored in shmem mode
    "gpu_d2h_chunk_size",# int: async D→H chunk size (default 0); optional in shmem
]
```

### 3. `ami/data.py:1222` — `_update()` discovers GPU detector types

```python
# ami/data.py  PsanaSource._update()  (after existing detector discovery)
if hasattr(run, "gpu_detinfo"):
    for det_name, attrs in run.gpu_detinfo.items():
        for attr_name, attr_type in attrs.items():
            # e.g. "jungfrau:gpu:calib" → GpuArray3d
            key = f"{det_name}:gpu:{attr_name}"
            self.data_types[key] = attr_type
            self._gpu_keys.add(key)   # track which keys use GPU path
```

### 4. `ami/data.py:1309` — `_process()` routes GPU keys to CuPy

```python
# ami/data.py  PsanaSource._process()
# Current single-attribute path (line 1353):
event[name] = obj(evt)        # det.raw.calib(evt) → numpy

# Add GPU branch before the existing path:
if name in self._gpu_keys:
    # GPU path: name is "jungfrau:gpu:calib"
    # evt is a plain psana.Event; GPU results live on evt.gpu
    psana_key = name.replace(":", ".", 1).replace(":gpu:", ".")
    # "jungfrau:gpu:calib" → "jungfrau.calib"
    result = evt.gpu.get(psana_key)
    event[name] = result.on_gpu      # Phase 3: cp.ndarray (stays on GPU)
    # event[name] = result.on_cpu   # Phase 1/2: np.ndarray (for immediate AMI compat)
else:
    event[name] = obj(evt)           # existing CPU path unchanged
```

### 5. `ami/comm.py:211` — `Store.get_type()` handles CuPy arrays

```python
# ami/comm.py  Store.get_type()  (add CuPy branch)
@staticmethod
def get_type(data):
    dtype = type(data)
    if isinstance(data, np.ndarray):
        return dtype, data.ndim
    try:
        import cupy as cp
        if isinstance(data, cp.ndarray):
            return dtype, data.ndim    # (cupy.ndarray, 2) for GpuArray2d
    except ImportError:
        pass
    ...
```

### 6. `ami/flowchart/library/` — new GPU-aware nodes

**`GpuToHost`** — the explicit D→H gate; type system enforces its use:

```python
class GpuToHost(Node):
    """Transfer a GPU array to CPU numpy for export or CPU-only nodes."""
    nodeName = "GpuToHost"
    def __init__(self, name):
        super().__init__(name, terminals={
            'In':  {'io': 'in',  'ttype': GpuArray},   # CuPy in
            'Out': {'io': 'out', 'ttype': Array}        # numpy out
        })
    def to_operation(self, **kwargs):
        return gn.Map(name=self.name()+"_op", **kwargs,
                      func=lambda a: a.get())           # cp.ndarray → np.ndarray
```

**Array-module-agnostic operators** — use `cp.get_array_module(a)` so the same
node works for both numpy and CuPy inputs:

```python
# Numpy.py  Sum node — GPU-aware version
def _sum(a):
    xp = cp.get_array_module(a)   # cp if CuPy, np if numpy
    return float(xp.sum(a))

class Sum(Node):
    nodeName = "Sum"
    def __init__(self, name):
        super().__init__(name, terminals={
            'In':  {'io': 'in',  'ttype': Union[Array, GpuArray]},
            'Out': {'io': 'out', 'ttype': float}
        })
    def to_operation(self, **kwargs):
        return gn.Map(name=self.name()+"_op", **kwargs, func=_sum)
```

### 7. `ami/data.py:208` — serialiser safety fallback

A `GpuToHost` node in the flowchart is the preferred D→H gate.  As a safety
net, the serialiser catches any CuPy arrays that slip through:

```python
# ami/data.py  ModuleSerializer  (inside the pickle-5 dumps closure)
def buffer_callback(obj):
    try:
        import cupy as cp
        if isinstance(obj, cp.ndarray):
            buffers.append(pickle.PickleBuffer(obj.get()))  # D→H here
            return
    except ImportError:
        pass
    buffers.append(obj)
```

### Summary of AMI changes

| File | Change | Lines |
|---|---|---|
| `amitypes/array.py` | Add `GpuArray1d/2d/3d` metaclasses + class bodies | ~25 |
| `amitypes/__init__.py` | Export `GpuArray`, `GpuArray1d/2d/3d` | ~5 |
| `ami/data.py` (ds_keys) | Add `gpu_det`, `n_gpu_streams`, `gpu_d2h_chunk_size` | ~5 |
| `ami/data.py` (_update) | Discover GPU detector types from `run.gpu_detinfo` | ~15 |
| `ami/data.py` (_process) | Route `_gpu_keys` to `evt.gpu.get(key).on_gpu` | ~10 |
| `ami/comm.py` | `Store.get_type()`: handle `cp.ndarray` | ~8 |
| `ami/flowchart/library/` | New `GpuToHost` node | ~15 |
| `ami/flowchart/library/Numpy.py` | `np.xxx` → `xp.xxx` using `get_array_module` | ~30 |
| `ami/data.py` (serialiser) | `buffer_callback` intercepts `cp.ndarray` | ~10 |
| `ami/graph_nodes.py` | `SumN`: `np.add` → `xp.add`, CuPy ndim check | ~8 |

---

## Shared-memory equivalent for GPU

For CPU: DAQ writes to POSIX shmem ring → psana2 reads with `mmap`.

For GPU, there is no direct equivalent today.  Three options in order of
feasibility:

```
Option A — CPU shmem + in-process H→D copy  (Phase 2, works now)
─────────────────────────────────────────────────────────────────
DAQ/DRP ──POSIX shmem──► ShmemDataSource ──► XTC2 dgram (CPU DRAM)
                                                    │
                                     GpuShmemEvents (new)
                                     cudaMemcpyAsync(cpu → GPU VRAM)
                                                    │
                                     Event + evt.gpu (CuPy float32)
                                     (calibration kernel or passthrough
                                      depending on drp_class)

Cost: one H→D copy per event
  drp_class='raw'  uint16 ~19 MB → ~0.5 ms
  drp_class='fex'  float32 ~38 MB → ~1.0 ms
DAQ change needed: NONE


Option B — CUDA IPC  (same node, zero-copy peer-to-peer)
─────────────────────────────────────────────────────────
DRP process allocates GPU VRAM, exports cudaIpcMemHandle.
psana2 imports handle → zero-copy read of DRP's GPU buffer.

Cost: near-zero copy overhead
Requirement: DRP and psana2 must run on the same GPU node
DAQ change needed: DRP must write detector pixels to GPU VRAM


Option C — GPUDirect RDMA  (long term, highest performance)
────────────────────────────────────────────────────────────
DAQ node ──InfiniBand RDMA──► GPU VRAM on analysis node
(bypasses CPU DRAM entirely, ~3–4 GB/s per link)

Cost: lowest latency, highest bandwidth
DAQ change needed: significant — DRP must use RDMA PUT to GPU
```

Option A is the practical choice for Phase 2.

---

## End-to-end pipeline diagrams

### Phase 1 + 2 (numpy to AMI, transparent to existing workflows)

```
                    ┌─────────────────────────────────────────────────────┐
  exp/run files     │  AMI Worker process (single-process psana2)         │
  or shmem ring ──► │                                                     │
                    │  DataSource(gpu_det="jungfrau", ...)                │
                    │       │                                             │
                     │  Phase 1 (exp/run): GpuEventManager               │
                     │  Phase 2 (shmem):  GpuShmemEvents                  │
                     │       │  GPU processing (CuPy)                     │
                     │       │  normal:      uint16 → calibration kernel  │
                     │       │  passthrough: float32 reshape, no kernel   │
                    │       │                                             │
                    │  Event (evt.gpu)                                    │
                    │       │  evt.gpu.get("jungfrau.calib").on_cpu       │
                    │       ▼                                             │
                    │  PsanaSource._process(evt)                         │
                    │       │  {"jungfrau:gpu:calib": numpy_array}        │
                    │       ▼                                             │
                    │  AMI graph (unchanged nodes, numpy input)          │
                    │       │                                             │
                    │  ResultStore → ZMQ PUSH                            │
                    └─────────────────────────────────────────────────────┘
                                  │
                         NodeCollector → GlobalCollector → Manager → GUI
```

### Phase 3 (CuPy through AMI graph, D→H at explicit gate)

```
                    ┌─────────────────────────────────────────────────────┐
  shmem / files ──► │  AMI Worker process                                 │
                    │                                                     │
                    │  GpuShmemEvents / GpuEventManager                  │
                    │       │  Event (evt.gpu)                           │
                    │       │                                             │
                    │  PsanaSource._process()                            │
                    │       │  evt.gpu.get("jungfrau.calib").on_gpu      │
                    │       │  → {"jungfrau:gpu:calib": cp.ndarray}      │
                    │       ▼                                             │
                    │  ┌─────────────────────────────────────┐           │
                    │  │  AMI flowchart graph                │           │
                    │  │                                     │           │
                    │  │  [GpuROI]──►[GpuPeakFinder]        │           │
                    │  │     ↓              ↓                │           │
                    │  │  cp.ndarray     int (hit)           │           │
                    │  │     ↓                               │           │
                    │  │  [GpuToHost]  ← D→H gate           │           │
                    │  │     ↓                               │           │
                    │  │  np.ndarray                        │           │
                    │  │     ↓                               │           │
                    │  │  [Binning] [Sum] ...               │           │
                    │  └─────────────────────────────────────┘           │
                    │       │                                             │
                    │  ResultStore (numpy only, GpuToHost ensures this)  │
                    │       │  pickle5 ZMQ PUSH                          │
                    └─────────────────────────────────────────────────────┘
```

---

## Implementation order

| Phase | What | Effort | When |
|---|---|---|---|
| **1** | Reuse the working serial experiment/run path; add `gpu_detinfo` property to `GpuEventManager` and `Run`; add GPU routing to AMI data-source keys; add an `evt.gpu is not None` branch to `_process()` calling `.on_cpu` | Small | First |
| **2** | `GpuShmemEvents` new class; `RunShmem` GPU branch; shmem H2D input adapter with parser reuse | Medium | After Phase 1 validated |
| **3** | `GpuArray1d/2d/3d` in amitypes; `GpuToHost` node; array-module-agnostic operators; `Store.get_type()` CuPy support; `_process()` returns `.on_gpu` instead of `.on_cpu` | Large | After Phase 2 validated |
| **RDMA** | DAQ-side changes (CUDA IPC or GPUDirect RDMA) | Very large | Future |

**Phase 1 has the lowest risk when AMI can identify data as an experiment/run**:
it reuses the existing `GPUDetector` / `EventPool` / `_D2hPipeline` stack.
Supporting AMI's explicit `files=` source is additional psana work because
`RunSingleFile` does not currently construct `GpuEventManager`. Phases 2 and 3
build incrementally on the same result API.
