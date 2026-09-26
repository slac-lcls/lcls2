# Removing calibration from the GPU runtime

2026-09-26. Source audit at `7d0b5941e` on
`codex/psana2-gpu-user-kernels`. This records the user's revised direction:
remove built-in calibration completely from the GPU runtime, including its
no-callback default. Removal is not implemented by this audit.

## Finding

No architectural or external dependency requires built-in calibration.
GPUBAT1 transport, KvikIO reads, the GPU parser, batched gather, input leases,
CUDA completion events, and CPU event delivery can operate without it.
Stage 1 already validated raw preparation without constants on an A100.
The remaining dependencies are implementation work, not a reason to retain
a second calibration path.

The previous design's requirement that `gpu_fn=None` retain automatic
calibration is superseded. Without a task, GPU routing/parsing and explicit
parsed-field access remain available, but no calibration, image assembly,
synthetic output names, or automatic output D2H occurs. Calibration is an
explicit user callback, including in examples and numerical acceptance.

Removing calibration now is feasible. A replacement that executes a user
callback and delivers its outputs still needs Stages 2–4. These are different
milestones: removal must not be described as completed callback support.

## Concrete dependencies to remove

| Area and source | Current coupling | Required change |
|---|---|---|
| [Manager setup](../../gpu_events.py) `_setup_gpu_pipeline`, `_setup_legacy_detector` | Pedestals determine dense shape; setup computes masks/inverse gain/offsets, uploads two fixed arrays and geometry, and chooses batch size from calibration cost | Use Configure bindings and supported input adapters; remove legacy setup and its imports; upload only task-declared source values when Stage 2 is wired; keep a simple default batch size of one and explicit batching |
| [Detector processing](../../gpu_detector.py) `GPUDetector`, `EventContext` | Owns calibrated slots, constant refresh, image scatter metadata, calibration and missing-row cleanup launches | Retain `DenseInputPreparer`; remove the legacy subclass/result container and calibration-specific helpers. User code owns calibration outputs and their missing-data policy |
| [Producer](../../gpu_stream.py) `EventPool.submit` | Calls `process_batch` and manufactures `.calib`, `.raw`, `.image` results | Remove automatic producers; preserve input descriptors, leases, and completion. Stage 3 dispatches the user task and only its publications become outputs |
| [Host delivery](../../gpu_events.py) `_PinnedSlot`, `_D2hPipeline` | Allocates float32 `(events, segments, rows, cols)` storage; fixes layout from the first image and schedules `.calib` copies | Remove image-only automatic D2H. Stage 4 supplies bounded byte-oriented staging for each publication's actual dtype/shape/extent; no output means no output transfer |
| [MPI setup](../../../psexp/mpi_ds.py) `_setup_gpu_geometry`, `_make_gpu_event_manager`; [GPU MPI helpers](../../gpu_mpi.py) | Precomputes GPU scatter indices, elects a calibration leader, exchanges fixed pedestal/gain-mask CUDA IPC handles | Remove GPU geometry preparation and calibration leader/IPC calls, helpers, and exports. Retain device assignment and `bd_ranks_sharing_gpu` for psana-owned input/constant budgeting |
| [MPI CPU caches](../../../psexp/mpi_ds.py) `_setup_jungfrau_shared_calib`, `_setup_jungfrau_shared_caches` | Eagerly builds derived CPU calibration/masks/geometry even for GPU-exclusive detectors | Exclude GPU-exclusive detectors using the same target list on every participating shared-memory rank. Preserve CPU/hybrid detector consumers and their collectives |
| [Transitions and accounting](../../gpu_events.py) `_refresh_legacy_calibration`, memory statistics/admission; [detector estimates](../../gpu_detector.py) | BeginStep recomputes two arrays; accounting expects constants/geometry/calibrated-slot keys | Remove the recipe but retain transition drains. Account for actual framework-owned input buffers and declared uploads; do not reserve or manage user scratch/output allocations |
| [Result access](../../context.py) `GpuEventState.get` | Qualifies bare names with a detector and special-cases `.image` failures | Stage 4 uses exact publication names; preserve terminal-copy readiness and owner retention without detector-output assumptions |
| [Algorithm module](../../gpu_calib.py), `cuda/fused_calib.cuh` | Runtime imports calibration, mask preparation, and image assembly helpers | Move useful numerical implementations to explicit example/test support and remove runtime imports/exports. No hidden wrapper may restore automatic calibration |

## What needs care, rather than another architecture

**MPI collective consistency.** The CPU shared-cache builders are also used
by ordinary CPU and hybrid processing. Deleting them wholesale would change
CPU behavior; skipping them only on GPU BDs could strand other ranks in a
collective. Filter by GPU-exclusive detector selection consistently before
the existing loops. Remove the GPU-only geometry precomputation separately.
Source `calibconst` loading/distribution remains available to CPU detectors and
to explicit task declarations; loading source values is not executing the
GPU calibration algorithm.

**Owners and completion.** Existing `SlotLease` and `InputSlotLease` mechanisms
do not require a calibration result. Keep execution-input protection even
when a callback publishes nothing. For user results, retain registered owners
through all submitted kernels and D2H, including partial failures. CUDA events
establish completion but do not keep an allocation alive by themselves.

**Compatibility.** Existing GPU clients/tests that expect automatic
`evt.gpu.get('calib')`, `.raw`, or `.image` results must migrate to explicit
publication or parsed-input access. There is no compatibility calibration
mode in the revised design. Retire the image-count
`gpu_d2h_chunk_size` option with an actionable error for nonzero requests;
publication controls output delivery. A later host staging byte cap is a
separate setting. Pre-calibrated float32 data in an XTC field stays an input
field; it must not recreate an implicit `.calib` producer.

**Tests and tools.** Several parser/ownership fixtures instantiate
`GPUDetector` for convenience, and allocation tests import
`gpu_calib._upload_fixed_arrays`. Migrate them to raw preparation/generic
`gpu_allocation.upload_owned` instead of retaining runtime calibration just
for fixtures. Adapt pixel-exact acceptance to an explicit user calibration
callback when Stages 2–4 are available. Preserve the old numerical reference
in test/example support in the meantime. Timing scripts that instrument
`GPUDetector.process_batch` or the calibration launcher also need migration.

## Revised next step and acceptance

Add Stage 1b immediately after the completed extraction: remove automatic
calibration, its setup/refresh/IPC, synthetic outputs, and image D2H. This stage
may deliver parsed inputs without a callback; it must reject unfinished task
configuration rather than accept and ignore it. Then continue Stages 2–4 on
one input/task/publication path, with no legacy/default branch.

Stage 1b must demonstrate serial and MPI GPU input processing with absent
pedestals, no legacy calibration/geometry allocations or launches, unchanged
CPU/hybrid calibration, consistent MPI collectives, and safe input retirement
when there are no outputs. Keep parser/gather batching and launch-count
checks. Update affected fixtures and run required core and device tests.
The user calibration/azimuthal-integration example supplies the later
scientific acceptance, with calibrated frames published only when requested.

This audit used source inspection and Stage 1's recorded validation evidence.
No new runtime changes, tests, performance measurements, or MPI removal
acceptance are claimed here. No external kernel download is required to begin:
the selected Amanda kernel commits are already available locally, as recorded
in the [example source note](calib_azint_callback_sources_20260926.md).
