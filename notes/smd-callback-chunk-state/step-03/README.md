# Production callback lifetime

Serial experiment-directory reading and MPI event-builder ranks now construct
one CallbackIterator per run over a sequence of input EventBuilders. Physical
EndStep ends step.events(); a chunk boundary no longer restarts either local
enumerate counter. State remains local to the executing EB rank, not global
across multiple EBs. The no-callback native batching path is retained.

StreamingCallbackBatches uses greenlet to suspend the callback stack when an
output batch is ready. This also works inside transition accumulation, so a
callback that filters all events cannot prevent output delivery. Callback,
reader and MPI execution remain on the caller's thread. Context variables are
copied into the callback execution context. greenlet >=3.1,<4 is a new runtime
dependency declared in pyproject.toml, imported only when using a callback.
Source-checkout/shared environments must install it before using this path;
validation installs 3.2.5 only under /tmp/smd-callback-step03/deps.

Pending output is serialized immediately, releasing parsed transition Dgrams.
Batches are bounded by 1000 records, 8 MiB of serialized record payload and the
requested selected-event batch_size. One oversized record is allowed by itself.
Packet footers and one in-progress record add bounded overhead. Output stays
paused until the consumer asks for another batch; there is no background queue.
These limits concern callback output. User-retained events, existing input
chunks, EnvStore history and the existing MPI StepHistory buffers are separate
and are not made globally memory-bounded by this change.

A batch contains selected events for one destination, with default and explicit
destinations separated at batch boundaries. Transition-only batches use default
routing; existing StepHistory delivers missing transitions to other BDs.
The MPI broadcast path ignores destinations, as serial delivery does.

Normal callback return stops selecting events for that run; the iterator drains
remaining input and forwards transitions. The MPI input source also receives
the terminal empty message before shutdown. This is a drain, not immediate
cancellation of an ongoing live acquisition. Explicit close injects cancellation
into the suspended stack and releases its input. Serial run iteration and the
termination flag close suspended callbacks. A new run starts a fresh callback.
Callback errors remain primary when cleanup also raises.

Tests compare exact event timestamps and scan/EPICS metadata against direct-file
reading. The original 24-case regression matrix now requires ordinary passes,
one callback invocation, and independently measured input-chunk counts.
Additional coverage exercises reject-all, early return, mixed default/explicit
destinations, deliberately tiny output limits, repeated runs, max_events,
10,000-transition backpressure, oversized records, cancellation and cleanup
failure. Both psana core groups must pass at the reviewed code SHA.

Native psana extensions are unchanged. Numerical Jungfrau calibration and
global multi-EB scheduling remain outside this step.
