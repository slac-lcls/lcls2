# psana.event

`Event` is the object yielded by `run.events()` and `step.events()`. It holds
the dgrams of one event (one per stream; missing ones are `None`) and gives
access to the event's `timestamp` (64-bit: seconds in the upper 32 bits,
nanoseconds in the lower 32 bits), `datetime()`, `service()` (transition id)
and `run()`, which returns a small `RunCtx` (`psana/psana/psexp/run_ctx.py`),
not the full `Run`. Detector data are read through detector objects
(`det.raw.calib(evt)`), not from the event directly.

`service()` returns `(env() >> 24) & 0xF` of the first dgram that has
`env()`, through `psana.utils.first_service`. That function raises
`RuntimeError` unless the value is between 1 and 13. Transition id 0 is
ClearReadout in both [`TransitionId`](transitionid.md) and the C++ enum (see
[XTC2 data format](../../features/xtc2.md)), so a dgram with that id makes
`service()` raise instead of returning 0.

See [DataSource, Run and the event loop](../../features/datasource-run-events.md).

::: psana.event
