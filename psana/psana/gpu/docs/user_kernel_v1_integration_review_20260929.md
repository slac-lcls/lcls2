# Batched user-kernel v1: integration review

Review checkpoint: `f46d23f95` on `codex/psana2-gpu-user-kernels`.
The complete user-kernel series starts after `480f7074c`; the incremental
changes since the last published checkpoint `21a6628bb` are external science
examples, complexity analysis, benchmark evidence and lifecycle acceptance.
This packet prepares integration review; no destination branch, merge or push
has been selected or performed here.

## Proposed change description

GPU users can declare external algorithms through `GpuTask(function, inputs,
calibconst)`. Psana prepares the selected inputs and exact requested constants,
calls the function once per memory-bounded execution subbatch, retains registered
scratch/output owners through CUDA completion, and delivers published results
through `evt.gpu.get(name).on_cpu`.

For a 20-event subbatch, the included calibration-plus-radial-integration
example invokes the user algorithm once and launches two kernels. The matched
public event-loop reference invokes the same algorithm 20 times and launches
40 kernels. Output transfer is grouped by publication; public event delivery
still materializes individual host rows.

Automatic internal calibration and calibrated-output delivery were removed.
Applications needing calibration now declare the task explicitly. The external
Jungfrau example matches the stated CPU-v3 calibration reference; its integration
example retains intermediate images on device and publishes compact histograms.

## Reviewer focus

- `gpu_task.py`, `gpu_task_batch.py`: host-only declarations, exact selectors,
  physical segment mapping, original constants and selected event identity.
- `gpu_producer.py`, `gpu_stream.py`, `gpu_d2h.py`: completion dependencies,
  owner retention, grouped output copies, quota fallback and failed-drain handling.
- `gpu_events.py`: memory-bounded subbatches, transition drain/refresh ordering,
  read-group integration and event delivery.
- `psexp/ds_base.py`, `psexp/mpi_ds.py`, `psexp/run.py`: configuration,
  MPI setup and deterministic serial cleanup through the public iterator.
- `detector/areadetector.py`, `detector/shared_geo_cache.py`: late geometry-cache
  misses compute locally, preventing event-loop entry into shared collectives.

The [overall review](user_kernel_overall_review_20260928.md) records the original
findings and their fixing commits. The [Stage 6 report](user_kernel_stage6_20260929.md)
closes lifecycle acceptance at the current runtime. No runtime change followed
that validation.

## Accepted evidence

| Area | Result |
| --- | --- |
| CPU compatibility | 608 passed; 173 GPU/other skips; 7 deselections |
| Longer CPU MPI suite | 5 passed |
| A100 integration | 173 passed, no skips |
| Public four-rank MPI | 2 publication cases, 12 lifecycle cases, 4 expected callback aborts |
| Matched user-kernel timing | 16 diagnostics and 38 matched pairs accepted |
| Full JF user-kernel scaling | 22 diagnostics and 88 timed samples accepted |
| JF+feespec user-kernel scaling | 6 diagnostics and 24 timed samples accepted |

The main warm single-BD comparison measured 101.30 events/s for event-loop
scheduling versus 355.28 for the batched task. Full JF scaling reached 918.86
events/s at four GPUs/eight BDs; mixed JF+feespec reached 332.76 at one GPU/four
BDs. These are event-loop rates for the documented workload and cache settings,
excluding explicit initialization. [Stage 5c](user_kernel_stage5c_20260928.md)
preserves the complete matrix, variability definitions and provenance. Earlier
staging-only results measure different work and are not matched regressions.

## Compatibility and limits to retain in integration

`batch_size` remains **1** by default; request batching explicitly, for example
20. `gpu_bulk_read` independently controls file-read grouping. The default
output pinned-memory limit is 64 MiB per BD, with synchronous ordinary-host
fallback. It does not bound arbitrary user GPU scratch or retained host results.
Borrowed inputs/constants are read-only by contract, not hardware enforcement.

Use `contextlib.closing(run.events())` for early exit. Closing one MPI rank's
iterator is not a collective job-stop API. GPU step transitions run through
`events()`; public GPU `steps()` iteration is outside v1. Validation used KvikIO
CPU fallback, so true-GDS acceptance is separate. The fixed-bin radial example
does not implement common-mode, solid-angle/polarization or pixel-splitting
corrections. Performance gains are workload-dependent.

The integration packet is ready for review within these limits. Changing the
batching default, extending detector/science support, and true-GDS validation
are separate follow-ups; none is silently included in this commit series.
