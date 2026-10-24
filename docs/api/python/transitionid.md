# psana.psexp.transitionid

`TransitionId` is the Python class that holds the transition codes as class attributes; `psana.psexp` re-exports
it, so `from psana.psexp import TransitionId` works. The values are
ClearReadout=0, Reset=1, Configure=2, Unconfigure=3, BeginRun=4, EndRun=5,
BeginStep=6, EndStep=7, Enable=8, Disable=9, SlowUpdate=10,
L1Accept_EndOfBatch=11, L1Accept=12 and NumberOf=13. The class methods
`name(id)`, `value(name)`, `isEvent(id)` (true for 11 and 12) and `all_ids()`
look them up.

The C++ enum `XtcData::TransitionId` uses the same numbers 0 to 12 but calls
value 11 `Unused_11` (see [XTC2 data format](../../features/xtc2.md)).
`evt.service()` returns one of these codes, but it raises `RuntimeError` for 0
(ClearReadout); see [psana.event](event.md).

::: psana.psexp.transitionid
