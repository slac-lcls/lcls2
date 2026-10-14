# CPU/GPU path size and complexity comparison

2026-09-26, checkpoint `31d68655a6f2fb711e2489f89d3149172b46dad3`.
Static analysis of current tracked source, before structural simplification.
No implementation or benchmark settings changed.

The GPU-specific runtime occupies **9,455 physical LOC in 24 source files**.
Its largest addition relative to the CPU path is orchestration and explicit
asynchronous ownership, not calibration arithmetic. **5,062 LOC (53.5%)** are
in coordination/read scheduling, field/result lifetimes, and byte accounting.
That grouping also contains setup and diagnostics, so it is not a measurement
of how many lines are strictly necessary for asynchronous execution.

The CPU path is substantially smaller at the event/read layer. But comparing
its 412-line `EventManager` with all GPU code would omit CPU parsing, detector
assembly, calibration, and shared infrastructure. The tables below make those
boundaries explicit. There is no defensible single “GPU is N times as complex”
ratio from these file sizes.

## Scope and measurement

- Workload: experiment-directory, SMD-driven CPU/GPU processing, especially MPI
  BigData ranks and Jungfrau. Direct-file, shared-memory, and DRP entry points
  exist in shared files but are not the main comparison workload.
- LOC means physical newline count, including comments, docstrings, blank
  lines, and embedded CUDA source. Counts are current file sizes, not added
  lines from a branch diff.
- GPU inventory excludes tests, documentation, scripts, and the three benchmark
  drivers. It includes `cuda/fused_calib.cuh` and both package `__init__.py`
  files. Some inventoried routines are references or optional helpers, not
  necessarily executed in every event loop.
- CPU comparison scopes include native code and inherited detector work.
  They are selected responsibility groups, not an exhaustive dependency
  closure. Whole files may implement more functionality than the workload uses.
- Python control-flow counts below come from AST inspection. They omit native
  C++/Cython and CUDA inside strings; they are a review aid, not cyclomatic
  complexity, runtime cost, or a correctness score.

## What each path does

Both paths use the same DataSource/Run machinery, SMD0, EventBuilder,
`BigDataNode`, and MPI batch look-ahead. The MPI CPU path is:

```text
BigDataNode -> Events -> EventManager
  -> SMD offsets / per-stream contiguous reads
  -> dgram.Dgram (C++ parser and NumPy views)
  -> Event -> user calls det.raw.raw / calib / image as needed
```

The MPI GPU path adds a run-scoped manager to that framework:

```text
BigDataNode -> Events -> GpuEventManager
  -> GPUBAT1 descriptors -> file epochs / read groups / byte admission
  -> KvikIO futures -> independently owned input buffers
  -> device XTC parser / field locators -> canonical gather / calibration
  -> EventPool result slots -> evt.gpu fields/results or automatic D2H
```

`GpuEventManager` also invokes the CPU `EventManager` for the CPU side of the
coherent event batch (`gpu_events.py:1379`). Exclusive GPU streams are absent
from CPU bigdata input, but CPU envelopes, transitions, and other CPU detectors
remain. `Run._handle_transition` and public Event materialization are shared.
The GPU path is therefore an extension of the CPU framework, not an independent
replacement whose entire size can be compared with one CPU module.

An important scope difference: CPU calibration is requested by user code;
`EventManager` only supplies datagrams. The GPU `EventPool.submit` invokes
detector processing before delivering results. Comparing their orchestration
files therefore also compares different amounts of scheduled work.

## Responsibility comparison

| Work | CPU source scope / LOC | GPU-specific source scope / LOC | Interpretation |
|---|---|---|---|
| Event delivery, reads, and scheduling | `Events` + `EventManager`: **589** | Manager, reader, execution/group scheduling: **3,192** | Largest visible gap; GPU group also includes setup, D2H, diagnostics, and admission absent from the CPU pair. A 5.4× file-size ratio here is not a matched-functionality overhead estimate. |
| XTC parsing and Configure/field access | `dgram.cc` + `container.cc`: **1,427**, plus XtcData dependencies | `gpudgram/{config,batch,parser}.py`: **2,121** | Both are substantial parser stacks. GPU adds device tables, batched locator kernels, and parser arena lifetimes. CPU creates Python objects/NumPy arrays and supports native format APIs. |
| Detector assembly and calibration | Jungfrau/inherited detector Python **1,811**, calibration wrappers/native files **581**: **2,392** | `gpu_detector.py` + `gpu_calib.py` + CUDA helper: **1,185** | CPU scope is broader: multiple calibration versions, shared derived constants, generic area-detector behavior, and common-mode support. GPU still uses CPU detector/configuration/calibration/geometry infrastructure. |
| Public fields/results and input lifetime | Distributed across `Event`, detector interfaces, native buffer bases, and copies; no equivalent standalone lease stack | `gpu_input.py` + `context.py` + `gpu_input_window.py`: **1,452** | GPU exposes explicit lease-aware device access and completion. This group includes field binding and lookup, not only lease bookkeeping. |
| Aggregate byte accounting | CPU read coalescing uses `PS_BD_CHUNKSIZE`; no equivalent total-owned-memory quota in the inspected CPU loop | `gpu_budget.py` + `gpu_allocation.py`: **418** | CPU chunk size limits a read grouping; it does not bound all retained buffers, calibration outputs, or consumers. |
| Transport metadata | Existing SMD datagrams and shared `PacketFooter` (**60**) | `gpu_batch.py`: **487**, plus integration in shared EB code | CPU reuses the XTC parser; GPU carries and validates a separate descriptor ABI and supports bounded event-range views. |
| Constant sharing/device placement | Shared MPI calibration and cache infrastructure | `gpu_mpi.py`: **547** | GPU additionally handles device selection, GPU-peer grouping, CUDA IPC, and diagnostics. CPU sharing is not zero-cost; see shared inventory below. |

These rows are comparisons of responsibilities, not two disjoint whole-pipeline
totals. For example, CPU `Event` and detector code also participate in GPU runs,
and `gpu_batch.py` does not include the GPU producer changes inside EventBuilder.

The native parser comparison also has a dependency boundary: just five relevant
XtcData files—`DescData.hh` (406), `ShapesData.hh` (383), `NamesIter.hh` (17),
`XtcIterator.hh` (108), and `src/XtcIterator.cc` (63)—add **977 LOC** beyond
the 1,427-line Python-extension front end. This is still not a full transitive
parser inventory. It shows why omitting the native format implementation would
overstate the apparent GPU parser expansion.

## Why CPU lifetime management looks much smaller

The current CPU BigData path uses blocking `os.pread` in
`EventManager._read` (`event_manager.py:258`). Each fill creates a new
`bytearray`, and `_fill_bd_chunk` replaces the current per-stream buffer with
that object. It does not overwrite an old buffer still retained by an event.

`dgram.cc:709` assigns the backing buffer as the NumPy array's base object;
`dgram.cc:1021` documents the reference-counted view mechanism. Retained arrays
keep their backing bytes alive through Python/NumPy references. Consequently,
the CPU loop does not need read-future retirement, cross-stream CUDA events, or
an asynchronous consumer registry. Retention can still increase memory usage.

CPU detector output has a different contract again. For a multi-segment array,
`AreaDetectorRaw.raw` (`areadetector.py:503`) fills a reusable stacking buffer
and copies by default; `copy=False` exposes that reusable buffer. The
single-segment path returns the native array view directly. The fast Jungfrau
calibration path writes cached `DetCache.outa` and returns it
(`UtilsJungfrau.py:467`, `:904–922`). It does not register external asynchronous
consumers before the next calibration call can reuse that output. CPU output
should therefore not be described as universally safe to retain without copies.

GPU input and calibrated output instead use reusable device allocations with
explicit completion obligations. A Python reference alone cannot establish
that a kernel on another stream has stopped reading that allocation. The
implementation tracks several independently progressing lifetimes:

| Lifetime / resource | CPU behavior in the inspected path | GPU behavior |
|---|---|---|
| Input read | Blocking read returns completed bytes | KvikIO submission/future; partial failures must drain safely |
| Raw backing | New buffer object; old views retain it | Reusable slots; input holds and planned uses prevent overwrite |
| Parsed metadata | Native objects/views backed by bytes | Shared device parser arenas, locator aliases, and completion events |
| Detector result | Synchronous computation; copy or reuse depending on API | Execution-slot output plus result leases |
| External consumer | No GPU completion protocol | Independent streams register terminal CUDA events |
| CPU delivery | Data already in host memory | Optional bounded pinned buffers, D2H events, and host-result references |
| Memory pressure | Chunk grouping plus allocator/reference lifetimes | Per-BD quotas, allocation-growth reservations, and admission splits |
| BeginStep / EndRun | Sequential CPU handling and shared transition logic | Drain execution and input consumers before refresh/dispatch |

This explains why some extra code is required. It does not prove every current
class or ownership transition is minimal. Determining that is the subsequent
structural analysis, not a conclusion from LOC alone.

## Where the GPU code is concentrated

This table is a disjoint inventory of all **24** tracked runtime source files
under `psana/psana/gpu/`; paths below are relative to that directory.

| Responsibility | Included files and current LOC | Total | Share |
|---|---|---:|---:|
| Coordination/read scheduling | `gpu_events` 1,532; `gpu_stream` 287; `gpu_kvikio_read` 500; `gpu_stream_read_plan` 125; `gpu_read_plan` 243; `gpu_group_schedule` 75; `gpu_file_epochs` 94; `gpu_admission` 69; `gpu_input_group` 267 | **3,192** | 33.8% |
| Field/result API and lifetime | `gpu_input` 835; `context` 420; `gpu_input_window` 197 | **1,452** | 15.4% |
| Quota/allocation | `gpu_budget` 286; `gpu_allocation` 132 | **418** | 4.4% |
| Parser/configuration | `gpudgram/config` 726; `gpudgram/batch` 453; `gpudgram/parser` 942 | **2,121** | 22.4% |
| Detector/calibration | `gpu_detector` 897; `gpu_calib` 233; `cuda/fused_calib.cuh` 55 | **1,185** | 12.5% |
| Descriptor ABI | `gpu_batch` 487 | **487** | 5.2% |
| GPU MPI/sharing | `gpu_mpi` 547 | **547** | 5.8% |
| Package exports | `__init__.py` 35; `gpudgram/__init__.py` 18 | **53** | 0.6% |
| **Total** | `.py` extensions omitted above except package files | **9,455** | 100% before rounding |

Within the largest file, `gpu_events.py`, setup alone is **313 LOC**. Its three
D2H classes total **238 LOC**. The memory snapshot/format/report functions and
class total **177 LOC** (AST spans, excluding surrounding blank lines and
decorators). Together these account for **728 of 1,532 lines** before the
remaining event/read/transition orchestration. This is why extracting setup and
logging could improve readability without producing a large repository saving.

At the arithmetic leaf, CPU `calib_jungfrau_v3` is only **21 C++ LOC**, plus a
short Python/Cython bridge. The GPU per-pixel calibration header is **55 LOC**
including extensive documentation, with the launch wrapper in `gpu_calib.py`.
Neither arithmetic kernel explains thousands of lines of pipeline machinery.
The CPU supports calibration variants/common-mode processing that the current
GPU detector explicitly rejects when `cmpars` is supplied
(`gpu_detector.py:346`). These are not feature-equivalent calibration packages.

## Control-flow complexity, measured consistently for Python

“Source lines” excludes blank lines, comments, and Python docstrings using
`tokenize` plus AST docstring ranges. Embedded CUDA string contents remain in
source lines. “Control sites” counts `if`, `for`/`async for`, `while`, ternary
expressions, `except` handlers, and comprehension generators/filters. Boolean
operators, `with`, and native/kernel branches are not counted. Definitions
include nested functions and methods.

| Module | Physical LOC | Source lines | Function definitions | Control sites |
|---|---:|---:|---:|---:|
| CPU `psexp/events.py` (shared dispatcher) | 177 | 136 | 5 | 23 |
| CPU `psexp/event_manager.py` | 412 | 281 | 12 | 40 |
| GPU `gpu_events.py` | 1,532 | 1,139 | 54 | 218 |
| GPU `gpu_kvikio_read.py` | 500 | 347 | 20 | 77 |
| GPU `gpu_stream.py` | 287 | 175 | 10 | 34 |
| GPU `gpu_input.py` | 835 | 691 | 67 | 118 |
| GPU `context.py` | 420 | 256 | 22 | 40 |
| CPU `detector/UtilsJungfrau.py` | 924 | 629 | 24 | 102 |
| CPU/shared `detector/areadetector.py` | 578 | 368 | 38 | 60 |
| Shared `psexp/run.py` | 925 | 677 | 53 | 106 |

The greater GPU orchestration size is not just comments: its source-line and
control-site counts are larger too. Counts do not show how those decisions
interact over time, which is important for asynchronous failure recovery.

Representative function spans give a more useful review scale than a global
complexity ratio:

| Function | LOC | Control sites | Main work |
|---|---:|---:|---|
| CPU `EventManager._get_offset_and_size` | 122 | 14 | SMD descriptor extraction, missing rows, read cutoffs, file transitions |
| CPU `EventManager._get_next_dgrams` | 66 | 8 | Choose SMD/bigdata backing, fill chunks, build native dgrams |
| Shared `Events.__next__` | 95 | 18 | MPI/serial/direct dispatch and batch exhaustion |
| GPU `GpuEventManager._setup_gpu_pipeline` | 313 | 43 | Routing, detector setup, constants, parser, budgets, device resources |
| GPU `GpuEventManager._process_batch` | 146 | 33 | CPU/GPU event matching, pre-issued reads, subbatches, delivery, cleanup |
| GPU `EventPool.submit` | 104 | 22 | Input leases, parsing, detector launches, completion/result attachment |

The CPU has its own substantial logic; the GPU's additional difficulty is the
number of owners and completion states that must agree across modules. Splitting
a large method does not reduce that coordination by itself.

## Shared code and CPU-side work that must not disappear from the accounting

The following selected framework inventory is **9,327 LOC**. Both paths rely
on it; it must not be charged only to CPU or only to GPU. Whole files also
contain other modes and GPU-specific integration, so this is a shared-file
inventory, not a count of lines executed by both paths.

| Files (relative to `psana/psana/`) | LOC |
|---|---:|
| `datasource.py`, `dgrammanager.py`, `event.py` | 256 + 716 + 227 = **1,199** |
| `eventbuilder.pyx`, `dgramlite.pyx`, `smdreader.pyx`, `parallelreader.pyx` | 975 + 256 + 1,195 + 236 = **2,662** |
| `psexp/ds_base.py`, `mpi_ds.py`, `run.py`, `node.py` | 951 + 995 + 925 + 1,457 = **4,328** |
| `psexp/smdreader_manager.py`, `eventbuilder_manager.py`, `packet_footer.py`, `step.py`, `run_ctx.py` | 422 + 71 + 60 + 86 + 39 = **678** |
| `psexp/calib_xtc.py`, `mpi_shmem.py` | 319 + 141 = **460** |

Shared detector support adds at least **1,066 LOC** in `calibconstants.py`
(516), `mask_algos.py` (272), `shared_calibc_cache.py` (141), and
`shared_geo_cache.py` (137). Geometry, calibration database access, environment
stores, XtcData dependencies, and external libraries extend this inventory.

The detector comparison's **2,392 LOC** is reproducible from:

- `detector/jungfrau.py` 105, `UtilsJungfrau.py` 924, `areadetector.py` 578,
  `detector_impl.py` 204: **1,811**.
- `pycalgos/utilsdetector.py` 127, `utilsdetector_ext.pyx` 140,
  `UtilsDetector.cc` 242, `UtilsDetector.hh` 72: **581**.

The tracked 177-line `psexp/parallel_pread.pyx` is not imported by the inspected
`EventManager` path. Its presence is not evidence that this CPU loop has an
asynchronous reader or an equivalent completion protocol. Conversely,
`parallelreader.pyx` is used upstream for SMD work and belongs in shared
infrastructure. No library implementation LOC is included for NumPy, CuPy,
KvikIO/cuFile, MPI, or Python's reference counting on either side.

## What this establishes before structural analysis

1. The CPU event/read loop is substantially simpler and smaller. Its blocking
   I/O, reference-counted input bytes, synchronous detector calls, and different
   output-retention contract account for part of the gap.
2. GPU parsing and detector arithmetic are not obvious orders-of-magnitude
   expansions of their CPU equivalents once native code and inherited work
   are visible. Feature sets and shared dependencies prevent a simple ratio.
3. The main added maintenance surface is coordinating descriptors, read groups,
   parser arenas, execution slots, result/field views, byte budgets, and D2H.
   Some of that is required by the promised GPU behavior; LOC does not tell us
   which representations could safely be eliminated.
4. The next analysis can use these responsibility boundaries and lifetime
   differences as its baseline. No deletion target or structural rewrite is
   proposed or validated by this report.

## Verification

Inventoried tracked source at the checkpoint, excluded benchmark/test/script
files from the GPU total, asserted that the eight GPU groups exactly cover the
24-file set without overlap, and recomputed all sums and Python metrics.
Inspected actual callers, native buffer ownership, CPU/GPU calibration entry
points, transition handling, and the unused-on-this-path pread helper.
This is static source analysis; no CPU/GPU timing or acceptance suite was rerun.
Earlier acceptance results establish behavior of the existing implementation,
not the correctness of a future simplification.

Related: [joint simplification plan](proposals/code_size_simplification.md),
[event flow and lifetimes](event_flow_and_lifetimes.md), and
[baseline](simplification_baseline_20260925.md).
