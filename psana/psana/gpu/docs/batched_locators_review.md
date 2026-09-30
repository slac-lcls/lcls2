# Batched locators: B+ implementation review

Reviewed 2026-09-18 against B (`803a70011d18168200927e279cbeaca90568e13f`).
This records the pre-commit review snapshot, not a new
gathering implementation or bulk-read integration.

This historical snapshot describes B+ before canonical gathering and on-demand
Python locator views. Those follow-ups are now covered by the
[gather review](batched_canonical_gather_review.md) and
[final pre-bulk review](batched_pre_bulk_review.md). The
[implementation plan](batched_canonical_gather_plan.md) retains the bulk
integration contract: locate per input window, gather per execution.

## Review result

No new correctness blocker identified in the production diff. Review covered
stream grouping, output-index stability, kernel ordering, error/duplicate
semantics, capacity strides, allocation rollback, cached/lazy access, and
existing consumer retirement. The field-offset decoder body is byte-for-byte
unchanged from B after extracting its work-index calculation into callers.
Source hashes for config.py, batch.py, and parser.py still match the installed
B+ build used in the completed validation and matched A/B/B+ run.

This review did not rerun GPU jobs. It uses the recorded CPU/GPU tests and
benchmark audits described below, with a fresh source-hash comparison and
`git diff --check`.

## Review package and scope

The local generated `validation/batched-locators-20260917/review-current.patch`
(not versioned; Git history is authoritative after commit) includes seven files,
469 insertions and 28 deletions: the three production files, parser documentation,
unit-test additions, the new integration test (otherwise untracked and absent
from a plain `git diff`), and the independent subprocess-test correction.

Additional untracked material consists of the two performance reports, the
validation harnesses/evidence, and the original handoff document. Those are
separate from this implementation patch. The original handoff describes the
pre-implementation plan and has deliberately been preserved as historical input.

| File | Change to review |
| --- | --- |
| `gpudgram/config.py:39` | Deduplicate handles; validate stream/indices; group by input stream; preserve caller order with explicit output indices. |
| `gpudgram/batch.py:88` | Add one `[handle, capacity, 11]` backing allocation per slot; reserve full replacement while old storage is live; roll back failed allocation. |
| `gpudgram/batch.py:179` | Compile scheduling tables once, account for fixed bytes, and upload them in the pool constructor. |
| `gpudgram/batch.py:247` | Replace per-handle `locate()` calls with one `_locate_configured()` call per parsed input. |
| `gpudgram/parser.py:240` | Submit initialization and decoding, record one shared ready event, cache compatible per-handle views. |
| `gpudgram/parser.py:576` | Extract the existing field-offset decoder into a device helper shared by eager and lazy paths. |
| `gpudgram/parser.py:748` | Initialize active dense output rows, including propagation of invalid dgram statuses. |
| `gpudgram/parser.py:766` | One block per dgram, distributing only its input stream's handles and actual ShapesData references. |
| `tests/gpu/unit/test_gpudgram.py:371` | Verify grouping/deduplication/empty ranges and successful/failed backing growth accounting. |
| `tests/gpu/integration/test_batched_locators.py:103` | Verify device equivalence, stream/Names isolation, tails/growth/empty inputs, lazy access, and malformed statuses. |
| `tests/test_extract_subset_xtc2.py:160` | Independent test correction: assert the event count inside the subprocess instead of requiring silent calibration startup; use the active interpreter. |

`gpu_events.py`, `gpu_stream.py`, `gpu_input.py`, `gpu_detector.py`, and the
reader/calibration implementations are unchanged. They appear in the trace
below because they establish ownership and consume the modified parser.

## Call path: setup once per GPU event manager/run

```text
GpuEventManager.__init__                         gpu_events.py:459
  _setup_detectors()                             gpu_events.py:547
    GpuStreamConfigTable.from_configs(configs)   gpu_events.py:603
    collect configured GPU-detector field handles
    construct GpuXtcBatchPool                    gpu_events.py:799
      build_field_location_tables               gpudgram/config.py:39
        deduplicate in caller order
        sort scheduling rows by input stream
        build stream prefix ranges
        store [Names index, field index, output index]
      reserve fixed table bytes                 gpudgram/batch.py:194
      configs.to_device(cp)                     gpudgram/config.py:715
      cp.asarray(stream_handles)                gpudgram/batch.py:208
      cp.asarray(handle_table)                  gpudgram/batch.py:209
      record _config_ready                      gpudgram/batch.py:210
```

The three existing Configure tables and two new scheduling tables live on the
pool. There is no per-subbatch upload of the run-scoped handle table. Event
descriptor metadata is still prepared/uploaded per subbatch. Input stream IDs
here are psana/XTC stream indices, not CUDA stream identities.

## Call path: each execution subbatch

```text
GpuEventManager._events                          gpu_events.py:1290
  retire reusable slot / issue KvikIO read
  _wait_gpu_read                                gpu_events.py:1140
  _submit_gpu                                   gpu_events.py:1069
    EventPool.submit                            gpu_stream.py:132
      require prior slot retirement
      GpuXtcBatchPool.parse                      gpudgram/batch.py:222
        slot CUDA stream waits on _config_ready
        slot.prepare: upload dgram records
        GpuEventBatch.__init__                   gpudgram/parser.py:139
          walk_xtc on slot stream
          record walk_done
        slot.batched_locator_rows               gpudgram/batch.py:88
          reuse capacity, or allocate replacement with rollback
        batch._locate_configured                gpudgram/parser.py:240
          init_locators                         gpudgram/parser.py:748
          locate_fields                         gpudgram/parser.py:766
            blockIdx.x selects dgram
            read owning stream from dgram record
            begin/end = stream_handles[stream / stream+1]
            threads stride through handles x actual references
            locate_field_ref checks matching Configure Names
          record ONE locator-ready event
          cache N DeviceFieldLocators views sharing that event
      GpuEventDgrams.from_batch                  gpu_stream.py:191
      GPUDetector.process_batch                 gpu_detector.py:416
        for each event / canonical segment:
          dgram.locate(handle) -> cached view    gpudgram/parser.py:291
          wait_on(slot stream)
          gather_locator_field_gpu
        calibration and missing-row cleanup
      record result_ready                       gpu_stream.py:210
      create result/input leases; retain xtc_batch and event views
```

The walker, initializer, decoder, and detector kernels are ordered on the
same slot stream. The separate initialization launch prevents initialization
from racing decoding across CUDA blocks. A consuming different stream can wait
on the shared locator-ready event. `_location_tables` and per-handle array
views retain their backing objects through the batch's lifetime.

For a tail, output address calculation is:

```text
((handle_output_index * allocated_capacity) + local_dgram_index) * 11
```

The allocation stride is explicitly passed to both new kernels. Active dgram
count controls the work size, so a smaller tail reuses larger capacity without
overlapping another handle's output rows. Per-handle exposed views are contiguous
and limited to active rows.

## Call path: lazy access and retirement

An unregistered handle still follows `GpuEventBatch.locate()` at parser.py:278:
separate lazy allocation, initialization, single-field decoder, and ready event.
On another CUDA stream it first waits for `walk_done`. Lazy output does not
alias the eager backing and remains separately accounted.

```text
EventPool.begin_retire_next                      gpu_stream.py:82
  synchronize producer stream
  keep slot occupied while caller consumes results
consumer view context exits                      gpu_input.py:417
  register consumer-done event with input lease
EventPool.finish_retire_next                     gpu_stream.py:106
  wait_until_safe_to_reuse for every registered lease
  clear old input views and xtc_batch references
  permit the next read/parse to reuse slot storage
```

Automatic D2H has its existing host-backed retirement path; external GPU
consumers retain their existing registration window. The locator ready event
indicates producer completion, not permission to overwrite a slot. Neither
retirement mechanism is changed by this diff.

## Scope qualifications for approval

1. **N-to-one launch submission is implemented; all O(N) CPU work is not.**
   The Python loop at parser.py:273 still creates per-handle locator views.
   It neither parses payloads nor submits kernels/events per handle.
2. **Decoding is stream-restricted; initialization is dense.** Absent-stream
   rows remain initialized for API compatibility. The GPU still examines each
   relevant handle/reference pair and walks preceding fields for offsets.
3. **No readback was added to the eager parser/calibration path.** The existing
   general field-view accessor at gpu_input.py:488 synchronizes and copies one
   locator row to construct a Python view. A blanket claim of no metadata
   readback for every public field-access API would be incorrect.
4. **Full replacement reservation is intentional.** A tight budget may reject
   growth even when the final new buffer alone would fit, because old and new
   allocations coexist. The new rule applies to combined eager backing; the
   older metadata/lazy allocation policies are unchanged.
5. **The subprocess-test fix is independent.** It addresses a failure reproduced
   on frozen B and can be reviewed/committed separately from the optimization.
6. **Gathering and bulk integration remain deferred.** The gather loop still
   submits per-event/per-segment work and repeated waits. The bulk branch must
   later adapt owner accounting, trimming, window-local row indices, and shared
   dependencies for the combined backing.

## Validation and measured effect

Recorded validation: 187 focused CPU/GPU passes (six slow cases deselected),
including 15 new device cases; main suite 234 passed / 25 skipped / eight
deselected; four longer MPI tests passed. The new tests compare batched and
lazy output/status behavior, explicitly check expected malformed statuses,
and verify stream isolation, tail strides, growth, rollback, and dependencies.
The shared-helper oracle is supplemented by the unchanged-decoder comparison
and real-data CPU pixel-reference checks.

Matched job 38513845 completed with exit 0:0: six correctness preflights and
27 timing samples audited, all with 100% cache residency. Six clean medians:
A 21.850 s, B 33.561 s, B+ 28.743 s. B+ improves B elapsed time by 14.4%.

Separate device traces from job 38504647 verify, per 20-event subbatch:

| Operation | B | B+ |
| --- | ---: | ---: |
| Field decoding kernels | 192 | 1 |
| Locator initialization kernels | 192 | 1 |
| Locator memsets | 192 | 0 |
| Walker kernels | 1 | 1 |
| Gather kernels | 640 | 640 |

Nsight warned of possible incomplete collection; expected principal counts
matched exactly. Traces and CPU-scope medians are separate measurements from
clean throughput and are not additive wall-time components.

Detailed reports: `performance/batched_locators_sdf.md` and
`performance/batched_locators_abo_sdf.md`. No new production edits or benchmark
submissions were made during this review.
