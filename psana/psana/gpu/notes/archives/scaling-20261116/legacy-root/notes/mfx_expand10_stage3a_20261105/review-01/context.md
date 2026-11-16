# Stage 3A independent review

Review the exact commit passed by the harness. Base ff5c3304d. Scope is
notes/mfx_expand10_stage3a_20261105/scope.md. Read cube.py, acceptance.py,
test_cube.py, run_gpu.sbatch, README, validation-summary.json and status.json.
No core code changes and no general managed lifecycle/factory implementation.
The user approved Stage 3A cube correctness on existing Stage 2 sample10 data.

This is a bounded application-owned callable: one fresh object per Run, at most
two distinct CUDA stream objects with disjoint persistent float32 banks. Explicit
finish after full native iteration joins stream work before bounded per-bin D2H
and float64 host merge. close handles aborts. The 5 GiB state cap is additional
to the 4 GiB framework input budget; user arrays are not framework-accounted.
No destructor science output. No retained borrowed descriptors, arrays or batch.

Check repeated-bin accumulation race freedom, exact event counts and malformed
panel rejection, kernel/data ordering and lifetime, state ownership before async
initialization/submission, global timestamp-to-ordinal mapping independent of
callback/BD order, no step-label reset, run isolation, completion/abort semantics,
CPU comparison bound and evidence claims. 30 bins, ordinal % 29, one intentionally
empty bin. CPU uses independently traversed seed expansion + CPU-v3 calibration
and float64 sums. Per-bin bound gamma_n*sum(abs(input))+1e-5, u=2^-24; per-stream
sums float32 and final host merge float64. No full event image D2H.

Stage 2 code is reused unchanged through Capture to keep intermediates alive
without publishing full images; only sim10.errors is published per event. Frozen
calibration is injected through the already-reviewed constructor-scoped test
loader. Relevant existing source: Stage 2 adapter.py and gpu_accept.py, current
GpuTask/ProducerContext, EventPool streams/lifetimes, Jungfrau calibration example.
Current-source contracts are authoritative; persistent_task_state_xpp_cube.md
remains a proposal. No GPU source-step lifecycle hook claims are made here.

Coordinator test results and source/runtime hashes are in validation-summary.json.
Full accepted reports and a copy index are available INSIDE THIS CHECKOUT under
build/stage3a-review-evidence/. Use these copies; absolute scratch paths are
provenance references outside the read sandbox. Copy equality is coordinator-
computed, as are Git blob hashes. Do not claim commands, tests or hashes were
run by the reviewer. Inspect all supplied evidence with bounded Read/Grep.
Do not delegate or start another review loop.

Acceptance: host identity/budget tests; controlled GPU repeated-bin, rejected
missing panel, empty run/bin, cross-step-label persistence on both streams,
completed fresh run, incomplete finish refusal, idempotent finish/close and abort
join tests; native tiny/tail/1000-event tests with exact counts and every final
cube pixel within bound. Source has one native step, so cross-step-label controlled
tests are NOT native multi-step acceptance. Runtime is serial, one A100 with
KvikIO compat ON, no MPI/GDS/35kHz or scientific-cube claim. No new data/DB/public
run changes. Return stable findings and exact reviewed SHA; max three rounds.
