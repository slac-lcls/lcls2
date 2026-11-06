## Problem

A persistent SMD callback can consume many input chunks and physical steps
inside one `next(callback_iterator)` call before yielding a selected event.
If it rejects every remaining event, that call runs until end-of-run. Chunk and
step boundaries do not themselves yield control to the batch consumer.

During that interval, BeginStep, EndStep, SlowUpdate and other transitions must
be retained for downstream delivery. Without a suspension/checkpoint mechanism,
delivery waits until the callback yields or finishes, and pending transition
memory can grow with the interval. Transitions are delayed, not inherently lost;
rejected L1 events need not all be retained. The issue affects long gaps between
yields as well as the extreme reject-entire-run case.

## Example

```python
def callback(run):
    for istep, step in enumerate(run.steps()):
        for evt in step.events():
            if istep == 3:
                yield evt
```

With 100 events per step and 10 events per SMD chunk, the first call requesting
a selected event consumes steps 0, 1 and 2 (300 rejected events across roughly
30 chunks) before yielding the first event in step 3. Transitions encountered
along the way wait for the consumer to regain control. Larger gaps increase
latency and pending transition storage; an ongoing live run may not reach a
yield or run end promptly.

## Scope and design decision

This is separate from the callback-counter reset bug: `ievt` must continue
across chunks within a physical step and reset at a new physical step. Retain
that behavior while deciding how to support prompt, bounded transition delivery.

The working branch `tests/smd-callback-chunk-state` previously introduced
greenlet-based same-thread stack suspension in commit
`7e2d5d3714b9ca1ef87cc9be8a1dc5879cbeed8b`. That implementation and its runtime
dependency are being revoked at the user's request. Do not reintroduce greenlet
or another execution-model dependency as an assumed solution. Agree on the
callback contract and design before implementation.

Options to evaluate include explicit callback checkpoints/API changes or a
bounded producer/consumer design with a documented threading/MPI contract.
Ordinary synchronous generators cannot transparently suspend an arbitrary
nested callback stack. Preserve existing callback compatibility where feasible;
document any required migration rather than silently changing the contract.

## Acceptance criteria

- [ ] Agree on a design and dependency/API policy before implementation.
- [ ] Deliver transitions during a long no-yield interval without waiting for
      the next selected event or end-of-run, with measurable pending-memory bounds.
- [ ] Exercise long rejected stretches within a step, across steps/chunks, and
      reject-entire-run behavior, including a controlled live-input case.
- [ ] Preserve exact selected timestamps, physical step boundaries, scan/EPICS
      metadata, destination routing, and counter continuity within a step.
- [ ] Define early callback return, cancellation, exceptions and MPI shutdown;
      verify no deadlocks, duplicated transitions, lost transitions or callback
      reinvocation at chunk boundaries.
- [ ] Distinguish callback output/transition bounds from existing input chunks,
      EnvStore/StepHistory caches and data retained by user code.
- [ ] Validate serial and single-EB MPI paths, document multi-EB scope, and run
      both psana core test groups with independent implementation review.

## Evidence

The triggering behavior follows from the synchronous generator call path; no
production out-of-memory incident or transition-loss incident is claimed here.
The counter-continuity regression suite and finite reject-all/routing tests live
under `psana/psana/tests/test_callback_chunk_state.py` and
`psana/psana/tests/test_callback_stream.py` on the working branch. Finite tests
that pass after run end do not establish bounded memory or prompt delivery
during the no-yield interval.
