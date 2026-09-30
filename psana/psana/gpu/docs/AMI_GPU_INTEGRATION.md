# AMI + PSANA2 GPU Integration Design

This document describes the design for GPU-accelerated detector data processing in
LCLS-II AMI (Analysis Monitoring Interface) via the psana2 GPU pipeline.

The GPU pipeline handles two data modes transparently:

- **Normal mode** (`drp_class='raw'`): bigdata contains raw uint16 ADC pixels;
  `GPUDetector` applies pedestals + gain calibration via `fused_calib_gpu`, yielding
  float32 calib output.
- **Passthrough mode** (`drp_class='fex'`): bigdata already contains float32
  pre-calibrated pixels written by the DRP; `GPUDetector` skips the calibration
  kernel and reshapes the data directly.  No calibration constants are loaded.

Both modes expose the identical `GpuEventContext` API — `ctx.get("det.calib").on_gpu`
returns a float32 CuPy array in either case.

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

The psana2 GPU event pipeline (`GpuEvents`) currently lives entirely
inside the MPI fan-out:

```
EB rank ──► BD rank (GpuEvents, GPU processing) ──► SRV rank
                                                       ↑ NullRun.events() = iter([])
                                                       AMI never sees this path
```

AMI workers never participate in psana2's MPI fan-out.  They always call
`DataSource(shmem=..., ...)` or `DataSource(files=..., ...)` in single-process
mode.  There are **two gaps** to close:

| Gap | Description |
|---|---|
| **Gap 1** | GPU event pipeline (`GpuEvents`) does not run in the single-process path used by AMI workers |
| **Gap 2** | No GPU-accessible equivalent of the DAQ shared-memory ring; DAQ writes to CPU POSIX shmem |

---

## Three-phase design

### Phase 1 — GPU processing in the single-process path (files=)

`RunSerial` already wires `GpuEvents` when `gpu_det=` is set
(`run.py:741`).  AMI workers using offline XTC2 files can use GPU event
processing today with almost no changes.  Both normal (raw→calib) and
passthrough (pre-calibrated fex) modes work automatically.

**Data flow:**

```
XTC2 files ──SMDReaderManager──► GpuEvents
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
                      GpuEventContext  ← yielded to AMI worker
                        ctx.get("jungfrau.calib").on_gpu  → cp.ndarray float32
                        ctx.get("jungfrau.calib").on_cpu  → np.ndarray float32
```

**Available GPU keys per event:**

| Key | Available? | Notes |
|---|---|---|
| `{det}.calib` | ✓ always | float32, normal and passthrough mode |
| `{det}.image` | ✓ if geometry loaded | float32 assembled image |
| `{det}.raw` | ✗ not yet | See note below |

> **Raw data via the GPU pipeline is not currently exposed — but the plumbing
> is already in place.**
>
> `gpu_stream.py:121–122` already checks `if ec.raw_gpu is not None` and would
> propagate `{det_name}.raw` into `gpu_results` (and on to `GpuEventContext.get()`)
> automatically.  The wiring is complete.  The only missing step is that
> `process_batch` discards the raw array with `_`:
> ```python
> calib, _ = self._extract_and_calibrate(...)   # raw_u16 thrown away here
> ```
> `EventContext.raw_gpu` is defined in the dataclass but never populated.
> `ctx.raw('det')` accesses CPU detectors only — GPU-routed detectors are
> excluded from the CPU-detector registry so it raises `KeyError`.
>
> **To enable raw GPU data in normal mode** (`drp_class='raw'`): keep the
> `raw_u16` return value alive in a separate per-event slot buffer (similar to
> `_calib_slot_bufs`), set `EventContext.raw_gpu`, and `gpu_stream.py` will
> propagate it automatically.  No changes to `GpuEventContext` or `context.py`
> are required.  There are currently no test cases for this path.
>
> **In passthrough mode** (`drp_class='fex'`) raw data is **structurally
> unavailable**: the DRP applied calibration and never wrote raw ADC values to
> bigdata.  This is a DAQ-level constraint, not a psana2 limitation.
>
> **Does AMI need raw data?**  Standard AMI graph nodes (ROI, peak-finding,
> binning, projection) operate on calibrated float32 arrays.  Raw ADU values
> are occasionally useful — noise statistics, gain-mode debugging, pre-pedestal
> hit-finding — but are not a typical online-monitoring use case.  The feature
> is straightforward to add when a concrete AMI use case requires it.

**AMI worker call:**

```python
# ami/data.py  PsanaSource
ds = psana.DataSource(
    files=["/path/to/run.smd.xtc2"],
    gpu_det="jungfrau",
    n_gpu_streams=2,
)
for run in ds.runs():
    for evt in run.events():    # yields GpuEventContext
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

The shmem GPU path is therefore **simpler** — it replaces four components with
a single H→D copy:

```
Shmem ring (CPU DRAM, complete XTC2)
    │
    │  one datagram per event, bigdata embedded
    ▼
extract detector pixels from dgram    ← replaces GpuBatchView + KvikioGpuReader
    │  drp_class='raw':  dgram.<det>.raw._rawdata  (uint16)
    │  drp_class='fex':  dgram.<det>.fex._rawdata  (float32, pre-calibrated)
    │  already in CPU DRAM, no file I/O
    ▼
cudaMemcpyAsync(cpu → GPU VRAM)       ← single H→D per event
    │  ~0.5 ms for uint16 (19-seg Jungfrau, ~19 MB)
    │  ~1.0 ms for float32 passthrough (same pixels, 2× size)
    ▼
GPUDetector.calibrate(det_gpu)        ← single-event entry point (already exists)
    │  normal mode:     applies pedestals + gain → float32
    │  passthrough mode: reshapes float32 directly, no kernel
    ▼
GpuEventContext                       ← identical API to Phase 1
```

**batch_size in shmem mode is always 1.**

`batch_size > 1` adds queuing latency (20 events × 8 ms = 160 ms at 120 Hz)
with no throughput benefit, because:
- There are no file I/O setup costs to amortize
- Each H→D copy is independent
- Online monitoring requires minimal latency

```
Files path, batch_size=20:
  GDS read setup cost amortized over 20 events:  0.05 ms / event
  50 MB read overlapped with CPU EventManager loop
  → batching gives 3–5× throughput improvement

Shmem path, batch_size=20:
  No GDS reads — no setup cost to amortize
  Added latency: 20 × 8 ms = 160 ms
  → batching gives zero benefit, harmful latency
```

**What remains valid in shmem GPU mode:**

| Component | Valid? | Notes |
|---|---|---|
| `GPUDetector.calibrate(det_gpu)` | ✓ | Single-event entry point; handles both raw uint16 and pre-calibrated float32 |
| `EventPool` | ✓ optional | Overlaps processing of event N with H→D of event N+1 |
| `_D2hPipeline` | ✓ optional | Async D→H hides transfer behind next event |
| `_GpuBudget` | ✓ | OOM prevention still needed |
| `GpuEventContext` / `GPUResult` | ✓ | Unchanged API |
| `on_gpu`, `on_gpu_view`, `on_cpu` | ✓ | Unchanged |
| `KvikioGpuReader` | ✗ | Needs file handles + byte offsets |
| `GpuBatchView` / GPUBAT1 | ✗ | No EventBuilder in shmem path |
| `_split_subbatches` | ✗ | No desc_table to split |
| `batch_size > 1` | ✗ practical | Adds latency, zero throughput benefit |

---

### Phase 3 — GPU-native AMI graph nodes (CuPy throughout)

In Phases 1 and 2, `GpuEventContext.on_cpu` is called to convert CuPy arrays
to numpy before they enter the AMI graph.  Phase 3 allows AMI graph nodes to
operate on CuPy arrays directly, with D→H deferred to the explicit `GpuToHost`
node.

```
GpuEventContext
    │  ctx.get("jungfrau.calib").on_gpu → cp.ndarray  (stays on GPU)
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

### 1. `RunShmem` — add GPU path  (`run.py:551`)

`RunShmem.__init__` currently always creates a CPU `Events` iterator.  Mirror
`RunSerial`'s pattern:

```python
# run.py:551  RunShmem.__init__  (add GPU branch)
if self.dsparms.gpu_det:
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

A simplified GPU event loop for the shmem path.  Replaces
`GpuBatchView + KvikioGpuReader` with a direct H→D copy:

```python
class GpuShmemEvents:
    """GPU event processing for the POSIX shared-memory (online) path.

    The shmem ring provides complete XTC2 datagrams — bigdata already
    in CPU DRAM.  For each L1Accept event:
      1. Extract detector pixels from the XTC2 dgram.
         drp_class='raw': uint16 ADC data  (normal mode)
         drp_class='fex': float32 pre-calibrated data  (passthrough mode)
      2. Copy to GPU VRAM via cudaMemcpyAsync (H→D).
      3. Run GPUDetector.calibrate() — applies calibration kernel (normal)
         or reshapes directly without any kernel (passthrough).
      4. Yield GpuEventContext with float32 calib output in both cases.

    batch_size is always effectively 1 — buffering multiple shmem events
    adds latency without any throughput benefit (no GDS reads to amortize).
    """

    is_gpu_events = True    # Run.events() dispatches to this iterator

    def __init__(self, configs, dm, dsparms, run, smdr_man=None):
        self.configs    = configs
        self.dm         = dm
        self.dsparms    = dsparms
        self.run        = run
        self._setup_detectors()   # creates GPUDetector per det_name;
                                  # sets _passthrough based on run.detinfo drp_class
        self._evt_iter  = smdr_man  # the shmem iterator (yields XTC2 dgrams)
        self._iter      = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = self._events()
        return next(self._iter)

    def _events(self):
        for dgrams in self._evt_iter:
            evt = Event(dgrams=dgrams, run=self.run._run_ctx)
            if not TransitionId.isEvent(evt.service()):
                yield from self._handle_transition(evt)
                continue

            gpu_results = {}
            for det_name, (_, gpu_det) in self.gpu_detectors.items():
                # Extract detector pixels from the XTC2 dgram.
                # drp_class='raw': uint16 ADC values  (gpu_det._passthrough=False)
                # drp_class='fex': float32 pre-calibrated (gpu_det._passthrough=True)
                det_bytes = self._extract_det_data(dgrams, det_name)
                if det_bytes is None:
                    continue
                import cupy as cp
                det_gpu   = cp.asarray(det_bytes)       # H→D copy
                calib_gpu = gpu_det.calibrate(det_gpu)  # normal: uint16→calib kernel
                                                        # passthrough: reshape only
                gpu_results[f"{det_name}.calib"] = calib_gpu
                # NOTE: {det_name}.raw is not populated here (raw_gpu=None).
                # gpu_stream.py already handles ec.raw_gpu when non-None, so no
                # further wiring is needed — just set EventContext.raw_gpu above.
                # In passthrough mode raw data is unavailable (DRP discarded it).

            yield GpuEventContext(
                evt=evt,
                gpu_results=gpu_results,
                cpu_dets=self.cpu_dets,
            )
```

### 3. `GpuEvents._next_batch()` — improve error message  (`gpu_events.py:843`)

```python
# Current: silent StopIteration
def _next_batch(self):
    if self.smdr_man is None:
        raise StopIteration

# Better: explain why and what to do
def _next_batch(self):
    if self.smdr_man is None:
        raise RuntimeError(
            "GpuEvents requires an SMDReaderManager (smdr_man). "
            "For DataSource(files=..., gpu_det=...) this is set automatically. "
            "For DataSource(shmem=..., gpu_det=...) use GpuShmemEvents instead."
        )
```

### 4. `GpuEvents.gpu_detinfo` — new property for AMI type discovery

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
| `psana/psexp/run.py` | `Run.gpu_detinfo`: new property forwarding to `GpuEvents.gpu_detinfo` | ~8 |
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
    # evt is a GpuEventContext from GpuEvents or GpuShmemEvents
    psana_key = name.replace(":", ".", 1).replace(":gpu:", ".")
    # "jungfrau:gpu:calib" → "jungfrau.calib"
    result = evt.get(psana_key)
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
| `ami/data.py` (_process) | Route `_gpu_keys` to `ctx.get(key).on_gpu` | ~10 |
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
                                     GpuEventContext (CuPy float32)
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
  XTC2 files        │  AMI Worker process (single-process psana2)         │
  or shmem ring ──► │                                                     │
                    │  DataSource(gpu_det="jungfrau", ...)                │
                    │       │                                             │
                     │  Phase 1 (files):  GpuEvents                       │
                     │  Phase 2 (shmem):  GpuShmemEvents                  │
                     │       │  GPU processing (CuPy)                     │
                     │       │  normal:      uint16 → calibration kernel  │
                     │       │  passthrough: float32 reshape, no kernel   │
                    │       │                                             │
                    │  GpuEventContext                                    │
                    │       │  ctx.get("jungfrau.calib").on_cpu           │
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
                    │  GpuShmemEvents / GpuEvents                        │
                    │       │  GpuEventContext                           │
                    │       │                                             │
                    │  PsanaSource._process()                            │
                    │       │  ctx.get("jungfrau.calib").on_gpu          │
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
| **1** | Files= path already works; add `gpu_detinfo` property to `GpuEvents` and `Run`; add `gpu_det` to AMI `ds_keys`; add `isinstance(GpuEventContext)` branch to `_process()` calling `.on_cpu` | Small | Now |
| **2** | `GpuShmemEvents` new class; `RunShmem` GPU branch; shmem-path H→D copy | Medium | After Phase 1 validated |
| **3** | `GpuArray1d/2d/3d` in amitypes; `GpuToHost` node; array-module-agnostic operators; `Store.get_type()` CuPy support; `_process()` returns `.on_gpu` instead of `.on_cpu` | Large | After Phase 2 validated |
| **RDMA** | DAQ-side changes (CUDA IPC or GPUDirect RDMA) | Very large | Future |

**Phase 1 has the lowest risk** — it requires four small changes (two in psana2,
two in AMI) and reuses the entire existing `GPUDetector` / `EventPool` /
`_D2hPipeline` stack.  Phases 2 and 3 build incrementally on Phase 1.
