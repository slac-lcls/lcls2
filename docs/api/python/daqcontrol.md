# psdaq.control.DaqControl

`DaqControl(*, host, platform, timeout)` is the client API for the control
process: `getState()`, `getStatus()`, `setState(state, phase1Info={})`,
`setTransition(...)`, `setConfig(...)`, `setRecord(...)`, `monitorStatus()`
and others. It is used by `daqstate`, `timed_run` and the scan helpers
(`TimedRun`, `ConfigScan`, `BlueskyScan`).

See [DAQ control](../../features/daq-control.md).

::: psdaq.control.DaqControl
