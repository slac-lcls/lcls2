# Stage 1 findings: JF allocation ownership and accounting

Measured 2026-09-23 against frozen bulk HEAD `8f94e3c7b`. Investigation only;
production sources and both integration histories are unchanged.

## Conclusion

The JF discrepancy is reproducible and attributable to live allocations whose
cache charges have been returned. Retired event/input facades continue to own
reader bytes, parser tables, and detector outputs after trimming. Allocator
rounding accounts for the small residual; unused pool cache is separate.

The six short JF cases reproduce a reconstructed allocation peak of
**8,325.501 MiB with bulk on**, versus **3,201.874 MiB with bulk off**. At every
one of 648 checkpoints, allocation/free identities reconcile exactly with
CuPy used-pool bytes. At every checkpoint with a live manager and budget:

```text
ledger committed = requested bytes of current cache/fixed allocations
pool used - ledger committed = detached live requested bytes + allocator rounding
```

There is **zero unexplained used-pool remainder** in these controlled cases.
This does not account for CUDA context or non-pool device memory, establish
multi-BD behavior, or attribute the cold-throughput loss.

## 1. Matched checkpoints identify the gap

Job **38892772**, sdfampere011: 1,000 events per case from JF run 387; six
fresh MPI launches, one SMD0/EB/BD, one A100, batch 20, depth 1, budget 8 GiB.
KvikIO CPU fallback, eight workers, 1 MiB tasks, automatic D2H disabled.
The first three events are checked against the frozen CPU raw/calibration
reference; every case also checks the complete timestamp sequence.

At the largest matched trim checkpoint in the bulk-on/no-user-retention case:

| Quantity | MiB |
|---|---:|
| Ledger committed after trim | 640.058 |
| CuPy used after trim | 5,763.687 |
| Detached reader allocations, rounded | 1,280.105 |
| Detached parser allocations, rounded | 3.522 |
| Detached detector output allocations | 3,840.000 |

The detached allocations comprise **two retired generations**. Each retains
about 640.053 MiB of raw input, 1.761 MiB of parser storage, and 1,920 MiB of
detector output (640 MiB canonical raw plus 1,280 MiB calibrated output).
The associated windows report `released=True` and zero live uses.

In exact bytes:

```text
used - committed = 5,372,514,008
detached requested = 5,372,351,680
all live allocator rounding = 162,328
unexplained remainder = 0
```

Subsequent replacement allocations overlap retained generations, reaching
8,325.501 MiB. That peak is reconstructed from every pool malloc/free event;
the largest periodic snapshot alone is lower. Bulk-off reuses buffers and has
no detached allocation gap at the recorded active checkpoints; its 81,912-byte
used-minus-ledger difference is allocator rounding.

## 2. Retaining user facades explains the end-of-loop memory

All entries below are CuPy used MiB, before forced GC or pool clearing:

| Mode | User references | Loop end | Drop final facade | Drop event-three facade |
|---|---|---:|---:|---:|
| Off | None | 0 | 0 | 0 |
| Off | Final | 2,561.873 | 0 | 0 |
| Off | Event three and final | 2,561.873 | 2,561.873 | 0 |
| On | None | 0 | 0 | 0 |
| On | Final | 2,561.873 | 0 | 0 |
| On | Event three and final | 5,123.687 | 2,561.873 | 0 |

With bulk off, the two facades share reused backing. With bulk on, trimming
and reallocation leave them owning different generations. Dropping references
releases their allocations immediately. This reproduces the earlier
5,123.687 MiB pixel-diagnostic endpoint and 2,561.9 MiB timestamp endpoint.
Even without saved user facades, bulk-on shows the in-flight trim discrepancy.

The manager has been destroyed at these loop-end checkpoints. The raw trace
marks manager/budget liveness and preserves the last observed ledger counter;
that historical value must not be compared as a current ledger after teardown.

## 3. Reference chains and independent causal probes

### Input storage

`InputWindow._try_retire` marks storage released and clears its callback, but
keeps `batch` (`gpu_input_window.py:93–115`). A stream facade also keeps a direct
batch reference (`gpu_input.py:125–156`). The batch retains the bound
`slot.locator_rows` allocator (`gpudgram/batch.py:216`, `parser.py:184`), which
keeps the entire old parser slot reachable.

Job **38892844** isolates these paths with real CuPy allocations and production
reader/parser retirement and trim methods:

1. Retire and trim: **zero ledger bytes**, **33,560,064 live pool bytes**.
2. Clear only `window.batch`: all those bytes remain through `stream.batch`.
3. Detach the bound locator allocator: an otherwise unrelated 3,072-byte
   locator allocation is released; the directly referenced raw/table arrays remain.
4. Drop the stream facade: all remaining variable backing is released, with
   no forced GC or allocator-pool clearing.

These are constructed metadata objects, so the probe isolates ownership rather
than claiming to exercise XTC parsing. The six JF cases exercise the real parser.

### Output storage and stale access

The same causal job holds a 4 MiB output view while growing its slot to 8 MiB.
The ledger charges only 8 MiB although 12 MiB remains live. Trimming returns
the remaining ledger charge while both views still keep 12 MiB alive. Dropping
the views releases 4 MiB and then 8 MiB (`gpu_detector.py:403–441`).

It also retires a `SlotLease`, overwrites the original output, and accesses the
old `GPUResult.on_gpu`: access succeeds and returns the replacement values.
This independently confirms the absent retirement access guard; accounting
alone will not prevent stale results.

### Pipeline-local retention

One-shot active-frame inspection (job **38892980**) identifies a retained
generation through `RunParallel._events_impl.envelope` (`psexp/mpi_ds.py:644`):

```text
MPI iterator's previous envelope -> gpu_state
  -> _event_dgrams -> stream.batch -> raw bytes and parser buffers
  -> _gpu_results -> canonical raw and calibrated output slices
```

The follow-up in job **38893055** finds Python enumerate's cached result tuple
reaching the **same** generation. It does not explain a second distinct generation.

Jobs **38893117** and **38893261** follow referrers of the two distinct GPU
states. The second chain is confirmed as:

```text
Events._batch_source / BigDataNode.start.batch_source
  -> suspended batch-source generator
  -> retained StopIteration exception / traceback
  -> frame -> old EventEnvelope.__dict__.gpu_state
  -> the first generation's reader/parser/output allocations
```

The first-generation GPU state reaches reader allocation 8; the active MPI
envelope reaches reader allocation 207. Their output allocations likewise
have distinct identities. Thus the two chains explain distinct generations,
rather than counting two references to the same bytes.

`Events.__next__` advances `next(self._batch_source)` inside its
`except StopIteration` suite (`psexp/events.py:104–106`). The measured chain
and that control flow support the diagnosis that advancing the generator
under the active exception retains the traceback/frame across its yield.
Moving that advancement outside the exception handler is a candidate Stage 3
cleanup and still needs an isolated regression test; it is not implemented here.

Frame/heap inspection is performed only after the matching memory checkpoint;
it can extend lifetimes, so subsequent memory samples from those follow-ups
are excluded from the quantitative control matrix.

## 4. Code changes and verification

Only new diagnostic scripts, launchers, summaries, and this report were added:

- `validation/ownership-stage1-20260923/allocation_trace.py`: non-owning
  allocation/free observer, weak window inventory, scalar reference paths.
- `jf_probe.py`: six retention patterns across bulk off/on, explicit reference
  deletion, pixel and timestamp checks.
- `retention_probe.py`: isolated real-CuPy input/output and stale-result probes.
- `jf_roots*.py`: one-shot reference inspection in separate processes.
- `run_probe.py`, Slurm launchers: staging and frozen build verification.
- `summarize_stage1.py`: allocation identities, ledger/cache/pool reconciliation,
  completion checks and build/script hash audit.

The frozen baseline install is reused read-only and verified before and after
the six-case job. The tracked production diff remains empty. No ownership fix,
B++ merge, admission change, or performance optimization is implemented.

## 5. Implications for Stage 2

Proceed with allocation-backed charges: the measured ledger currently matches
cache inventories, while live allocations outlast those inventories. Preserve
full replacement reservations and make detached generations keep their charges.

The owner mechanism must cover reader/parser storage and later detector output
storage; outputs account for most of the observed retained bytes. Stage 3 must
also close stale access and detach unnecessary backing from shared event state.
Clearing only the window field, tweaking the budget limit, or returning fewer
cache bytes without tracking the underlying allocation is insufficient.

Accurate charges can reject the existing 8 GiB plan while these generations
remain retained: the observed 8,325.501 MiB live-pool peak already exceeds
8 GiB before admission headroom. Keep intermediate Stage 2 work isolated until
Stage 3 lifetime cleanup and Stage 4 validation establish a usable bounded path.
Do not treat a larger budget as a resolution of the ownership defect.

Stage 1 does not validate delayed consumer races, injected asynchronous failures,
general growth pressure, multi-BD/IPC sharing, or allocator interop. Those remain
explicit tests for the ownership implementation, not evidence that the current
code is safe. The short runs show bounded repeated behavior, not a long-run
memory-leak proof or a throughput result.

## Artifacts

- Matrix and audit: `validation/ownership-stage1-20260923/job-38892772/summary.md`
  and `summary.json`; complete per-case `.jsonl` allocation traces alongside.
- Causal evidence: `validation/ownership-stage1-20260923/causal-38892844.log`.
- Reference paths: `validation/ownership-stage1-20260923/roots-38892980/trace.jsonl`
  through `roots-38893261/trace.jsonl`; the latter contains the complete
  `state_referrers` chain for both retained generations.
- Reproduction instructions: `validation/ownership-stage1-20260923/README.md`.
