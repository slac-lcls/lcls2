# CPU/GPU path size and complexity comparison

Updated **2026-09-28** through user-kernel checkpoint
`6ba5fa586af4b740b3d3b41c2499e471d7e08dce` (Stages 1–5b), with completed
Stage 5c performance evidence. The September 26 analysis is preserved below as
an explicitly historical baseline. Its old automatic GPU calibration and D2H
descriptions do not describe the current task API.

## Current assessment, including batched user kernels

The GPU runtime now occupies **8,833 physical LOC in 26 files**, compared with
**9,455 LOC in 24 files** at the September 26 checkpoint: **622 fewer lines
(−6.6%)**. This includes **801 LOC** in four new task/input/publication modules.
Removal of automatic calibration and its derived-constant/MPI setup, plus
manager refactoring, more than offset those additions in physical source size.
This is the net checkpoint-to-checkpoint change, including cleanup; it does not
isolate the implementation cost of batching alone.

Some algorithm code moved outside the runtime. The external Jungfrau calibration
and integration implementations add **196 + 156 = 352 LOC**, including their
embedded CUDA kernels. Counting those as well gives **9,185 LOC**, only **270
lines (−2.9%)** below the old runtime total. The old runtime included calibration
but not the new radial integration algorithm. The three example drivers add
another **140 LOC**; runtime plus all five example files is **9,325 LOC**.
These totals expose the scope change rather than treating moved code as deleted
functionality. Tests, benchmark harnesses and documentation are excluded.

CPU still has the simpler synchronous event/read path: **589 LOC** in the shared
`Events` dispatcher and CPU `EventManager`, unchanged across these checkpoints.
The GPU coordination/read group is **2,834 LOC**, before the additional task,
publication and lifetime groups. The file-size gap remains substantial, but the
GPU groups also implement device preparation, asynchronous execution, bounded
memory and host delivery that the CPU pair does not provide. Neither their ratio
nor the total GPU/CPU selected-source ratio measures matched functionality.

Fewer lines do not mean fewer decisions. Across inventoried Python runtime files,
source lines fell **6,732 → 6,525**, while function definitions rose
**395 → 431** and control sites rose **1,076 → 1,122**. The generic task API adds
validation and lifetime cases while separating responsibilities. These AST
counts exclude native code and embedded CUDA control flow and are not cyclomatic
complexity or performance measurements.

## Responsibility changes since September 26

The same disjoint inventory boundaries are used at both revisions; new generic
task/publication files receive their own row. All sizes are physical LOC.

| GPU responsibility | Sept. 26 | User-kernel checkpoint | Change |
|---|---:|---:|---:|
| Coordination/read scheduling | 3,192 | 2,834 | −358 |
| Field/result API and input lifetime | 1,452 | 1,425 | −27 |
| Quota/allocation | 418 | 419 | +1 |
| Parser/configuration | 2,121 | 2,121 | 0 |
| Detector/calibration inside runtime | 1,185 | 392 | −793 |
| Generic task/input context/publication/D2H | 0 | 801 | +801 |
| Descriptor ABI | 487 | 487 | 0 |
| GPU MPI/sharing helpers | 547 | 316 | −231 |
| Package exports | 53 | 38 | −15 |
| **Runtime total** | **9,455** | **8,833** | **−622** |

The 801-line row comprises `gpu_task.py` **143**, `gpu_task_batch.py` **223**,
`gpu_producer.py` **209**, and `gpu_d2h.py` **226**. It includes task declarations,
selective original-constant staging, selected event alignment, callback-scoped
inputs, publication validation, pinned-memory accounting and host delivery.
It is not 801 lines of kernel-launch code. Some responsibilities, especially
D2H, previously lived inside other files; this row is not a pure net addition.

The former detector/calibration group now consists of the **392-line**
`gpu_detector.py`, which prepares dense inputs and gathers physical segments.
`gpu_calib.py` and `cuda/fused_calib.cuh` were removed. The user algorithms own
calibration arithmetic and integration. `gpu_events.py` shrank **1,532 → 1,109**
lines, while `gpu_stream.py` grew **287 → 352** as it gained task execution and
failure-safe owner retention.

Coordination, field/lifetime handling, quotas and the new task/publication group
together occupy **5,479 LOC (62.0%)** of the runtime, versus **5,062 (53.5%)** in
the earlier grouping without task modules. Setup, diagnostics and API validation
are included. Managing asynchronous ownership remains the largest maintenance
surface even though total runtime size fell.

The CPU/shared scopes also remain visible:

| Selected scope | Sept. 26 LOC | Current LOC | Accounting boundary |
|---|---:|---:|---|
| CPU `Events` + `EventManager` | 589 | 589 | Event/read layer only; `Events` also dispatches GPU processing |
| Native parser front end | 1,427 | 1,427 | `dgram.cc` + `container.cc`; XtcData dependencies excluded |
| Jungfrau/inherited detector and native calibration files | 2,392 | 2,418 | Broader CPU functionality, including common-mode/other calibration variants; user integration not included |
| Selected shared framework | 9,327 | 9,249 | Both paths depend on these files; they also contain GPU integration and unrelated modes |
| Selected shared detector/cache support | 1,066 | 1,078 | Both paths; geometry/cache fixes contribute to the change |

These overlapping responsibility scopes must not be summed into exclusive CPU
and GPU pipeline totals. The full per-file inventories and consistently measured
Python metrics are in the [source evidence](cpu_gpu_complexity_comparison_20260928.json).

## User algorithm handling: CPU versus current GPU

The current GPU flow is:

```text
DataSource(gpu_fn=GpuTask(...), batch_size=20)
  -> assigned BD stages only declared original constants
  -> read groups / device parsing / requested dense or field inputs
  -> one callback per nonempty selected execution subbatch
       -> user calibration kernel -> user integration kernel
       -> publish named (N, ...) device results
  -> psana records completion and copies publication groups to host
  -> public event loop consumes each event's .on_cpu result
```

CPU user code normally calls `det.raw.calib(evt)`, then its own integration
function, then consumes the result within the event loop. A CPU user can build
their own batching or parallelism; the inspected CPU event path does not supply
the GPU task execution/publication contract.

| Responsibility | CPU user path | Batched GPU user path |
|---|---|---|
| Declare work | Ordinary Python calls in the event loop | Host-only `GpuTask(function, inputs=..., calibconst=...)` passed as `gpu_fn` |
| Obtain inputs/constants | Detector methods and CPU calibration/cache infrastructure | Declare required inputs and exact constant keys; psana prepares aligned input rows and original arrays on the assigned worker |
| Run calibration/integration | Synchronous CPU calls; integration remains user code | External callable receives `(batch, stream)`; the validated example launches two kernels for all selected rows |
| Scratch and outputs | Python/NumPy allocation; some detector outputs are reused | User owns device scratch/output allocation; register owners with `keepalive` or `publish` before launch |
| Deliver output | CPU result is already host-accessible | `publish(name, array)` enables grouped D2H and per-event `.on_cpu`; accessing the result does not launch the user callback |
| Preserve inputs | Native views retain input bytes; copied/reused detector outputs have distinct contracts | Borrowed inputs/constants are read-only; leases and CUDA completion protect reuse |
| Handle memory pressure | Read chunk sizes and allocator/reference lifetimes | Framework quotas cover owned inputs/parser/requested constants; user device scratch remains outside that quota; output pinned staging has a separate cap |
| Finish or stop early | Normal iterator and CPU object cleanup | Drain queued kernels/copies and retire owners on exhaustion or explicit iterator close; serial and MPI paths require coverage |

The GPU consumer loop is simpler than manually coordinating per-event device
work, but the algorithm author still owns CUDA arithmetic, stream use, allocation
policy and output shape. Psana owns scheduling, input readiness, publication
completion and host delivery. This moves repeated coordination out of the user
event loop; it does not eliminate its implementation or testing cost.

One callback per subbatch is not one kernel total: the example has **one task,
two kernels**. Per-event identity, result-row mapping and public Event delivery
remain. Outputs are copied per publication group, so multiple separately
published arrays may require multiple copies within one execution.

**Default/API detail:** with `gpu_fn`, omitted `batch_size` defaults to **1**.
Use `batch_size=20` to request multi-event batching; memory admission and tails
can produce smaller execution subbatches. `batch_size=1` keeps the task but
executes single-event batches and also changes upstream batching. `gpu_fn=None`
omits user analysis and automatic task output delivery. `gpu_bulk_read` controls
file-read grouping independently. There is no separate kernel-batching Boolean.
See the [implemented task/results guide](user_task_results.md).

## What the performance evidence establishes

Completed Stage 5c job **39380082** passed all **16 diagnostics and 38 matched
pairs**, including identical event-associated histogram hashes. Both variants
use the same calibration/integration kernels, input batching and constants.
The comparison changes per-event public-loop analysis versus pipeline-batched
analysis; its reference uses a documented benchmark adapter to expose leased
dense inputs. It is **not** a comparison against CPU calibration/integration.

| Same-work GPU comparison | Per-event GPU rate | Batched GPU rate |
|---|---:|---:|
| 1 GPU / 1 BD, batch 20, depth 2, warm | 101.30 events/s | 355.28 events/s |
| 4 GPUs / 4 BDs, batch 20, depth 2, warm | 354.39 events/s | 771.92 events/s |

Rates are 10,000 divided by median event-loop seconds, excluding explicit
DataSource/Run initialization but including lazy first-use work, I/O and output
delivery. Bulk reads are ON in this comparison. The independent real-input
kernel check also reproduced the timing benefit with hot buffers, separately
from I/O, allocation and D2H. The completed report, review and timing checks are
retained under
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage5c-followup-20260928-r1/`
in `report.md`, `review.json` and `plausibility.json`. The isolated check repeats
the first real event; it supports timer plausibility without identifying every
cause of the batching benefit.

We can therefore say that the generic GPU API supports the intended batched
user work and substantially improves these measured GPU workloads. We cannot
claim a CPU/GPU throughput advantage from this experiment: that needs a matched
CPU calibration-plus-integration benchmark with the same numerical policy,
input/cache conditions, output contract and startup accounting. CPU has broader
detector/entry-point coverage and calibration options. The external example
does not implement all CPU common-mode or scientific integration corrections.
The final lifecycle acceptance checklist and scaling interpretation also remain
separate from this static complexity inventory.

## Reproducing the updated inventory

The [inventory script](../scripts/complexity_inventory.py) reads committed Git
blobs, not mutable working files. It excludes GPU scripts/tests/docs/examples
and the same three benchmark drivers from both runtime totals, asserts complete
nonoverlapping group coverage, and counts user examples separately. Python
source-line and control-site definitions match the historical method below.
Recalculation reproduces the original **24 files / 9,455 LOC** exactly.

```bash
python psana/psana/gpu/scripts/complexity_inventory.py \
  --before 31d68655a6f2fb711e2489f89d3149172b46dad3 \
  --after 6ba5fa586af4b740b3d3b41c2499e471d7e08dce
```

This documentation update changes no runtime or benchmark settings and requires
no performance rerun. The scheduled scaling completion report remains separate.

## Historical analysis: September 26 checkpoint

The remainder preserves the original analysis at
`31d68655a6f2fb711e2489f89d3149172b46dad3`. References to “current” and source
line numbers in this historical section refer to that revision only.

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
