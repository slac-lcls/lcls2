# User-kernel integration: preparation handoff

2026-09-26. **Preparation only; no user-task API implemented.**

The detailed design has been promoted to the canonical
[User GPU kernel support proposal](user_gpu_pipeline.md). This file retains
checkpoint, provenance, and validation information; implementation decisions
belong in that proposal. The earlier handoff's requirement to declare managed
scratch/output arenas is superseded by user-owned allocation and publication.

## Starting point

- Branch: `codex/psana2-gpu-bulk-batched-integration`.
- Worktree: `/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-d2h-pipeline`.
- Pre-merge checkpoint: `6041ba64d5275feded6a9073aa19e2d0dc7d0e07`.
- Master brought into this branch: `f743ef5be` from `origin/master`.
- Merge commit: `b307ceb7a`.
- Current proposal review baseline: `d63f45d27`, which also includes the
  remote branch's completed JF+feespec scaling report.
- The merge had no conflicts. `dgrammanager.py` preserves master's loop that
  discards pre-Configure shared-memory datagrams and the GPU branch's
  multi-stream detector/segment routing tables.
- GPU runtime modules and psana/psalg/XtcData native sources are unchanged by
  this merge. Master also brings calibration-service, detector, packaging,
  workflow, and DAQ changes.
- Broad [code-size simplification](code_size_simplification.md) remains
  **deferred**. It is not a prerequisite for this task.

Start implementation on a separate task branch/worktree from the validated
merged branch. Keep old experiments, logs, and validation symlinks out of that
task's changes.

## Read these first

1. [User GPU kernel support](user_gpu_pipeline.md): the canonical detailed
   proposal for `GpuTask(function, inputs, calibconst)`, internal producer
   dispatch, user-owned allocations, `evt.publish(name, array)`, and bounded
   asynchronous host delivery. It does not require advance input/output sizes.
2. [Memory backpressure and results](../memory_backpressure_and_results.md):
   current ownership, byte accounting, result access, and retirement contracts.
3. [Architecture overview](../architecture_overview.md),
   [event flow](../event_flow_and_lifetimes.md), and
   [known limitations](../known_issues.md): supported entry points and boundaries.
4. [Detector materialization ownership](detector_materialization_ownership.md):
   a **deferred** alternative for releasing original XTC storage after gathers.
   Do not treat that model as implemented or as an automatic prerequisite.
5. [Current scaling](../performance/jungfrau_current_scaling.md) and
   [simplification baseline](../simplification_baseline_20260925.md): measured
   baseline configurations. These do not validate a future user-task pipeline.

## Historical design and task context

The matching user-callback design was found as an **uncommitted** September 17
edit in the stale `codex/psana2-gpu-user-callback` worktree, based on `8f94e3c7b`:

```text
/sdf/home/m/monarin/lcls2_worktree/psana2-gpu-user-callback/psana/psana/gpu/docs/proposals/user_gpu_pipeline.md
SHA256: 1bffe2efc33390392f858433f914ac9b55e7ee7e004832b7199bb13d80bfffba
```

That content is now captured and updated in the current proposal. There is no
need to merge or depend on the old worktree to recover the design. Its old
whole-stream residency model and single-consumer result-lease claim were
replaced with the current read-group and multi-consumer ownership contracts.
The old worktree and its uncommitted edits remain untouched.

Commit `192333e26` consolidated and removed earlier design documents. Read them
from Git history when details are useful; do not restore old registries or
pipeline classes solely because the documents mention them.

| Historical source | Where to retrieve it | Useful context |
|---|---|---|
| Kernel registry | `7c41348bb:psana/psana/gpu/docs/gpu_kernel_registry_design.md` | Named CUDA/CuPy registration and launch conveniences; not the chosen minimum integration contract. |
| Kernel scheduling | `64e6fdc36:psana/psana/gpu/docs/gpu_kernel_scheduling_design.md` | Internal dispatch, bindings, multiple outputs, per-slot allocation. |
| Compiled task ABI | `192333e26^:psana/psana/gpu/docs/gpu_task_c_abi_design.md` | Opaque task context, explicit CUDA stream, named buffers, return/error conventions. Original design commit: `831f6c4e7`. |
| Callback capacity notes | `192333e26^:psana/psana/gpu/notes/gpu_callbacks_capacity_model.md` | Conditional work, retention, and bounded output capacity; history includes `acfdf11c4`, `ec30713a2`, and `704b6f15e`. |

For example, `git show 192333e26^:psana/psana/gpu/docs/gpu_task_c_abi_design.md`
reads the last version before consolidation.

The untracked root file `xx` is an illustrative scratch sketch, not executable
code or an accepted API. It describes conditional peak finding, multiple tasks,
an accumulator, and periodic host delivery. Its `kernels=`, `max_output_slots`,
and `call_user_kernel` spellings are not implemented. Preserve these use cases
when discussing requirements without treating the sketch's syntax as binding.

The GPU parser task (`01a06d1e-038a-7151-b61e-e86cd3f27204`, September 4
session) also discussed detector-independent access across streams, segments,
shapes, algorithms, and fields. That history explains why user-task input
binding should build on current Configure/field metadata instead of adding
Jungfrau-specific addressing. The implementation has since evolved; current
code remains authoritative.

## Current implementation boundary

The supported experiment/run path is:

```text
Run / BigDataNode -> Events -> GpuEventManager
  -> stream read groups / input owners / batched GPU parser
  -> EventPool.submit -> GPUDetector.process_batch
  -> result-ready event / result and input leases
  -> optional bounded D2H -> Event(evt.gpu)
```

- `gpu_stream.py:EventPool.submit` is the main task-dispatch boundary to inspect.
  It invokes detector processing before recording result readiness and
  creating result leases. It does not currently invoke registered user tasks.
- `gpu_detector.py` provides detector gather/calibration and output allocation;
  `gpu_events.py` owns setup, admission, event matching, and D2H delivery.
- `context.py` provides `GPUResult`, `SlotLease`, and `on_gpu_view(stream)`.
  An external user can already enqueue GPU work inside a view context with
  automatic D2H disabled. This is different from internal producer scheduling.
- `gpu_input.py`, `gpu_input_group.py`, and `gpudgram/` provide field binding,
  parsed input access, input ownership, and completion tracking.
- `gpu_budget.py` and `gpu_allocation.py` account for managed device storage.
  User scratch/output allocations remain outside that ledger in the proposed
  first implementation. Psana-managed prepared inputs and requested constants
  participate in device admission; pinned output staging has its own aggregate
  byte cap. Publication retains owners without taking over allocation policy.

Both `SlotLease` and `InputSlotLease` already collect multiple consumer events.
Older proposals claiming a single terminal result event are stale. Preserve
open-view protection, completion tracking, failure cleanup, and transition
draining when extending the pipeline.

## Direction for the new task

- `GpuTask` declares a callable, input selectors, and exact calibration keys.
  Psana resolves input metadata and stages only the requested constants.
- `gpu_fn=None` preserves existing calibration. An explicit callback replaces
  automatic processing on the selected GPU path; it can launch user calibration
  and additional kernels on the supplied stream.
- Scheduling occurs inside the GPU subbatch producer, before public delivery.
  The initial callback takes `(evt, stream)` once per selected event; a true
  batch callback can follow. No task launch occurs from the public event loop.
- User code allocates scratch and outputs. `keepalive()` retains temporaries;
  `publish()` registers exact output names, metadata, and owners. No advance
  scratch/output shape declarations or psana-managed device arenas are required.
- Psana records producer completion and schedules every published output for
  host delivery. `.on_cpu` consumes the host token; it does not launch tasks or
  initiate the normal D2H. A bounded synchronous fallback handles pinned pressure.
- Preserve event identity, input-owner dependencies, multi-consumer leases,
  and drain-before-release behavior, including errors and transitions.

Use the proposal's staged implementation and acceptance checks. The first
end-to-end case is Jungfrau threshold/mask/count, followed by conditional peak
output and stateful accumulation. EpixUHR coverage is retained as a follow-up.
The numerical pinned cap, public import spelling, lifecycle conveniences, and
any direct native ABI still need implementation decisions. Do not reopen the
deferred structural simplification merely to start this task.

Acceptance must cover output identity/pixels, missing and conditional results,
partial subbatches, multiple consumer streams, tight budgets, delayed consumers,
BeginStep/EndRun, early exit, and task failure after submission. CPU mocks do
not replace real-device lifetime checks. Run both core psana test groups when
changing DataSource/Run/native integration.

## Merge validation

Validation is recorded on shared scratch under
`/sdf/scratch/users/m/monarin/gpu-validation/master-merge-20260926-r2`.
The snapshot contains current merged Python sources and packaged geometry data,
with unchanged compiled extensions from the verified Integrated runtime.
The manifest records source hashes and both merge parents.

Job **39183159**, host `sdfmilan008`, completed both required groups:

| Command scope | Result |
|---|---|
| `pytest psana/psana/tests/` | **473 passed, 111 skipped, 10 deselected**, 108.12 s |
| `pytest psana/psana/tests/byhand_*` | **4 passed**, 120.97 s |

Both commands used `-q -p no:cacheprovider`, separate scratch `--basetemp`
directories, and a runner that imports the verified snapshot before pytest.
`PS_PARALLEL=mpi`; GPU visibility was disabled on this CPU validation node.
No new GPU acceptance or throughput campaign was run. The snapshot reused
native binaries because their sources are unchanged from the verified runtime;
the updated Python/calibration-service code was exercised by these tests.

Attempt r1 (job 39182940) failed because the preparation script omitted
packaged geometry data and allocated too few MPI slots. Correcting the snapshot
and scheduler request resolved those failures without a production-code change.
Its logs remain separate from the successful r2 evidence.

Changed Python files were also parsed with Python 3.9. The merge preserves
master's existing trailing whitespace and extra EOF blank lines; only the two
formatting hooks were skipped for the merge commit. Other applicable commit
checks passed. Preparation documentation uses the normal full hook set.
