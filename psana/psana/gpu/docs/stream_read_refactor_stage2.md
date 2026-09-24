# Stream-read refactor Stage 2: independent input ownership

Date: 2026-09-24. Base: Stage 1 commit `f3fee4f53` on
`codex/psana2-gpu-bulk-batched-integration`.

## Implemented

- `InputGroupPool` owns a dedicated reader's bounded raw slots. Each live
  `(batch_id, group_id)` has independent I/O, parsed storage and consumer
  ownership. A small stream has at most one outstanding small group, including
  across EB batches.
- Binding a parsed group reserves one planned use for every present event.
  `take_use()` transfers that use to execution/event code; consumers can fork
  it. A temporarily empty active execution set cannot release future inputs.
- Deferred `InputWindow` retirement queries all producer, locator and consumer
  CUDA events. The pool polls every group, allowing a later completed group to
  release before an earlier busy group. No timestamp-based release rule.
- `issue_group()` maps a Stage 1 contiguous request onto the existing KvikIO
  submission path. Generation checks, byte accounting, file/destination
  retention, full future draining and read-error poisoning remain in force.
  Group allocation pressure raises before submission.
- Explicit shutdown drains I/O and CUDA work, cancels untaken planned uses,
  and rejects live uses already transferred to callers. Failed CUDA completion
  retains ownership for retry. Parser pools must also close to drain failed
  parser submissions; their quarantine keeps raw storage protected meanwhile.
- Parsed aliases detach before the release callback exposes reusable backing.
  Successful detachment is not repeated if a release callback needs a retry.

Production still uses the existing residency scheduler. The new owner pool is
opt-in. `InputGroupPool.parse()` is a convenience for ownership validation;
Stage 3 must preserve batched parsing/gather/calibration when binding groups.
One read group must not imply one kernel launch. Existing nondeferred window
behavior remains available, and parser shutdown explicitly drains either mode.

## Validation

**69 CPU tests passed**: 14 group ownership tests plus 55 existing input-window,
reader, allocation and input-view tests. Cases cover:

- later-group reclamation while earlier JF/small consumers remain pending;
- cross-batch small-stream credit and planned future uses;
- forked/retained consumers and stale generation rejection;
- producer plus multiple consumer completion dependencies;
- allocation pressure before submission;
- submission failure, failed future and short reads;
- query/wait failure, callback retry and parser-failure quarantine;
- shutdown with unparsed I/O, untaken uses and still-live consumer references.

**Two real GPU tests passed**, job **39019345**, `sdfampere010`, A100-SXM4-40GB,
driver 575.57.08, CuPy 13.6.0, KvikIO CPU fallback, eight workers, 1 MiB task
size. Pytest reported 4.13 seconds; this is correctness-test duration, not a
throughput measurement.

The new test reads real XTC fixture datagrams through KvikIO and parses them on
the GPU. One coalesced two-event group and an earlier one-event group have a
delayed CUDA consumer. A later one-event group completes and its exact raw
buffer pointer is reused while the delayed event is still incomplete. The
retained bytes and locator rows compare exactly before releasing the earlier
groups. The existing retained-window/execution-slot test also passes.

This uses the small checked-in xppcspad fixture to exercise ownership; it is not
a JF+feespec performance run. The first allocation, **39019306**, on
`sdfampere003` failed before either test could allocate GPU storage
(`cudaErrorDevicesUnavailable`). The retry used an explicit `srun` step on
another node. Neither that failed attempt nor this correctness validation
changes the accepted performance baseline.

## Evidence and reproduction

Artifacts:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-ownership-stage2-20260924/`

- `job-39019345.log`, `device-retry.xml`: accepted device results.
- `job-39019306.log`, `device.xml`: failed CUDA initialization attempt.
- `retry.sbatch`: GPU launch/environment; account `lcls:data`, normal QoS.
- `python/psana`: source overlay on the frozen Integrated native install.
- `provenance.json`: tested source hashes and native install identity.

The native base is
`/sdf/scratch/users/m/monarin/gpu-validation/a-bpp-off-20260923/installs/Integrated`
(E runtime `ac87a93b2`). The overlay replaces the five modified/new GPU Python
modules without rebuilding or changing that baseline. CPU validation uses:

```bash
PS_PARALLEL=none PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  --confcutdir=psana/psana/tests/gpu/unit -q \
  psana/psana/tests/gpu/unit/test_gpu_input_group.py \
  psana/psana/tests/gpu/unit/test_gpu_input_window.py \
  psana/psana/tests/gpu/unit/test_gpu_bulk_read.py \
  psana/psana/tests/gpu/unit/test_gpu_allocation.py \
  psana/psana/tests/gpu/unit/test_gpu_input.py
```

Use the Python, overlay `PYTHONPATH` and native `LD_LIBRARY_PATH` in
`retry.sbatch` when reproducing this installation.

## Stage 3 obligations

1. Submit eligible groups in stream-interleaved order, skipping blocked streams;
   the pool supplies ownership/backpressure, not the runtime scheduler.
2. Transfer planned uses into execution/event consumers and keep batched
   parser/gather/calibration launches. A shared parser allocation may need a
   shared release callback, while raw groups retain independent ownership.
3. Split execution at small-group availability boundaries to prevent a batch
   from waiting for the next small group while retaining its predecessor.
4. Reserve actual raw concurrency, parser/output/scratch/cached bytes and
   allocation growth under the shared GPU budget. Slot bounds alone do not
   establish a complete progress guarantee.
5. Migrate all success/error/early-exit drains before removing residency
   branches. Preserve the cleanup checklist in `stream_read_refactor_cleanup.md`.

Full integration correctness remains Stage 4; cold/warm throughput and native
file-concurrency acceptance remain Stage 5.
