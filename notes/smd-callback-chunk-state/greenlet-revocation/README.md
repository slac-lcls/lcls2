# Persistent callbacks without greenlet

The production callback remains alive across SMD chunks in one run. Its local
event counter continues across chunks within a physical step and resets when
the callback starts the next physical step. A new run starts a new callback.
The original 24 counter-continuity regressions remain mandatory ordinary passes.

StreamingCallbackBatches now uses an ordinary synchronous Python generator.
The greenlet dependency and stack-switching implementation have been removed.
Callback execution shares the caller's thread and context. Existing serial/MPI
integration, destination routing, owned input records and cleanup remain.

Transitions are serialized into a pending queue while the callback executes.
The batch consumer regains control only when that callback yields or returns;
it cannot interrupt a running callback by closing the stream from the same
thread. After control returns, queued records are emitted in order, respecting
the existing per-batch record, byte and selected-event limits. One oversized
record may occupy its own batch. These output limits do not bound the pending
transition queue during a long interval without a callback yield.

The separate issue draft in [issue.md](issue.md), with title and labels in
[issue.json](issue.json), describes that deferred delivery/memory problem. It
covers long rejected stretches as well as an entirely rejected run; transitions
are delayed, not inherently lost. No replacement execution dependency is assumed.

Validation uses the source overlay and existing native build without the
temporary greenlet install. The standalone delivery/cleanup driver also rejects
greenlet imports explicitly. Finite reject-all tests verify eventual delivery;
they do not claim prompt delivery or bounded pending memory.

Both this revocation and the exact issue payload undergo independent Claude
review before publication. See scope.md and the round records for the contract
and findings. Historical step-3 documents retain their original evidence and
are marked superseded.
