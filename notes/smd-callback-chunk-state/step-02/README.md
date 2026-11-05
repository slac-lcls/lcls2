# Persistent callback input iterator

`psana.psexp.callback_iterator.CallbackIterator` keeps one callback generator and
one callback-side `RunSmallData` alive across incoming EventBuilder chunks. A
physical EndStep ends `step.events()`; exhausting an input chunk fetches another
builder without ending the callback's step/event loops.

This is step 2's input/lifetime component. Production `CallbackBatchBuilder`,
serial delivery and MPI routing are unchanged. The step-1 production regressions
remain expected failures until step 3 integrates this component.

The constructor takes an iterable of EventBuilders, Configure dgrams, DataSource
parameters, and a callback. It is an iterator of selected Events and a context
manager. `chunks_read` counts source builders independently of callback calls.
Input may be fetched lazily and may reuse old chunk buffers.

Native proxy records point into their input chunk. The new iterator replaces
them with records backed by the owned dgrams already copied by `EB.events()`.
This avoids another payload copy while keeping event and transition data valid
after refill. Routing destinations remain attached to each selected event.
Records serialize with `as_bytearray()`; they must not be passed into the native
proxy-recycling method, which requires native ProxyEvent objects.

The callback is never automatically restarted after return or exhaustion.
Natural completion closes the input iterator but leaves accumulated transition
records available. `close()` also clears pending transitions, runs callback
cleanup, and closes upstream input exactly once. Closing before iteration does
not start the callback or read a chunk. A fresh run uses a fresh iterator.

Step 3 still must supply batch delivery and backpressure. In particular, a
callback can reject events indefinitely inside a single `next()` call. Merely
replacing the production chunk builder with this iterator would allow the
transition queue to grow. Production integration must drain transitions during
input consumption, enforce bounded buffering, serialize the new owned records,
and handle upstream MPI shutdown after early return. Local callback state also
does not provide global event counts across multiple EB ranks.

Tests use real generated XTC and SMD files and native EventBuilders. A direct
file read supplies exact timestamps and scan/EPICS metadata. The source poisons
each old chunk buffer before refill and on close; retained event and transition
packets must still match the native serialization captured before poisoning.
Cases cover chunk sizes 100/4/7, batch sizes 1/3, all steps/step 1, event-only
iteration, reject-all, early return, explicit close (including before start),
exceptions, and a fresh iterator for a second run in the same process.

```bash
python -m pytest psana/psana/tests/test_callback_iterator.py -v
```

The implementation/review loop remains active for subsequent work in this
session. See `scope.md`, `status.json`, and the round records for its state.
