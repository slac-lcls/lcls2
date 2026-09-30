# Stream-read legacy cleanup

2026-09-25. Follow-up item 3, after the ownership acceptance committed as
`4821499c0`. This removes the old policy and its unused production branches;
throughput tuning remains deferred.

## Changes and replacement coverage

| Removed or migrated | Replacement assertions |
|---|---|
| Whole-stream residency candidates, ranking, fit diagnostics, and resident fields in `AdmissionPlan` | `test_gpu_stream_read_plan.py`, `test_gpu_group_schedule.py`, and `test_gpu_input_group.py` cover bounded adjacent groups, deterministic order, stream credits, and independently reclaimable inputs. `test_gpu_admission.py` retains complete-event, parser, detector, allocation-growth, and minimum-progress accounting. |
| `GpuReadSelection`, resident start/close, resident-only reads and mixed resident/transient submission branches | `test_gpu_group_runtime.py` covers request ordering, short tails, missing streams, successive batches, max-events, early close, partial I/O/parser failures, budget rejection, retained views, and transition retry. The hybrid CPU-envelope assertion was moved here before deleting `test_gpu_residency.py`. |
| Old residency device fixture | Shared `gpu_group_fixture.py` constructs the production group path. Pixel/parser-launch checks cover all former padding layouts, including two small streams. `test_gpu_group_ownership_device.py` retains all 18 public-field, tight-budget, D2H, and independent-consumer cases. |
| Residency lifecycle device tests | All nine `test_bulk_lifecycle_device.py` cases now exercise groups: max-events, generator close, read/gather failure, D2H, and calibration refresh after BeginStep followed by EndRun. The early-exit/failure fixture forces several executions; the read failure occurs after earlier parsed windows exist. |
| `KvikioGpuReader._coalesced_plan` and `issue_batch` bulk branch | Bulk-on submits resolved groups through `issue_group`; `issue_batch` explicitly accepts bulk-off only. Direct-group tests compare metadata against the pure CPU reference and bytes against the source. Real CUDA/KvikIO parity still compares bulk-on and bulk-off parsed fields and payloads. Chunked SMD input and transition fences use the stream planner. |
| Discarded generic `ReadPlan` construction in the stream planner | Shared descriptor and physical-span validation preserves duplicates, timestamps, integer bounds, zero rows and overlap rejection. Generic `build_read_plan` remains a CPU reference; the runtime stream planner does not construct it. |
| Multi-range file/failure tests dependent on obsolete planning | Bulk-off tests continue exercising multi-future draining through the shared submit engine. Bulk-on file-reference tests inject physical ranges directly into that engine, including partial submission/open failures and out-of-order completion. Production group fault tests separately cover controller cleanup. |
| Obsolete timing hooks | `bulk_phase_timing.py` replaces resident start/close with group issue/submit, and removes coalescing instrumentation. Current timing installation succeeds with 41 patches. Historical summary parsing and frozen campaigns remain intact. |

Raw-slot generation guards, input holds, parsed windows, completion leases,
shared parser arenas, canonical gathers and the bulk-off path are retained.
The standalone raw/parser alias tests explicitly use bulk-off; production
group ownership has its own device matrix. The allocation-GC test collects
unrelated cyclic owners before taking its memory baseline so suite order does
not affect its assertion.

## Validation

- Full CPU GPU-unit suite plus benchmark-harness tests: **430 passed** (406
  unit tests and 24 harness tests). The count decreased because obsolete
  ranking/residency expectations were replaced, not because tests were skipped.
- Timing instrumentation installed successfully against the frozen current
  runtime, including its checked statement matching and group hooks.
- A100 job **39097057**, `sdfampere002`: **53 passed in 12.89 s**,
  no failures or skips. Runtime and test SHA-256 manifests verified before and
  after execution. This includes all migrated lifecycle cases, the 18-case
  ownership matrix, group scheduling, transition/gather, allocation, isolated
  input-window, and real bulk-on/off byte/field parity coverage.
- `git diff --check` passes.

CPU validation used the frozen `stream-read-cleanup-20260925-r2/python` source
prefix and the existing Integrated native installation. The final device
candidate uses identical runtime Python bytes, with corrected lifecycle and
GC-baseline test setup in `stream-read-cleanup-20260925-r3`.

All generated inputs, frozen Python, source diffs, test hashes, job scripts and
logs live under `/sdf/scratch/users/m/monarin/gpu-validation/`. No native rebuild
was needed. No new throughput comparison was run.

## Deliberately retained work

Untracked historical `compare_admission_priority.py`, `trace_bulk_reads.py`, and
`run_trace_bulk_reads_{sdf,perlmutter}.sbatch` describe the removed residency
policy. They require a historical runtime and are not maintained current group
tracing tools. They remain untouched, along with scratch evidence and prior
benchmark reports. Current group tracing uses the maintained benchmark harness
and timing hooks.

Backend-neutral I/O diagnostic wording, preview bootstrap simplification,
selective public field-owner leases and performance follow-up are separate
items; none is needed to retain the ownership contract during this cleanup.
