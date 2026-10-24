# psdaq.control.TimedRun

`TimedRun(control, *, daqState, args)` moves a DAQ between states for a run of
fixed length; `psdaq/psdaq/control/lab3_timed_run.py` is the example script.
`control` is a [`DaqControl`](daqcontrol.md) object (the code only calls its
`setState` and `monitorStatus`), `daqState` is the state returned by
`control.getState()`, and `args` is the parsed command line (the constructor
reads `args.v`). The constructor starts two daemon threads: one requests the
states with `control.setState`, the other reads `control.monitorStatus()` and
updates `daqState`. `stage()` and `unstage()` request `connected`,
`set_running_state()` requests `running`; each call waits until the
requesting thread reports that the state was reached (it never reports this
if `setState` returns an error). Send `'shutdown'` on `push_socket` to stop
the requesting thread.

See [DAQ control](../../features/daq-control.md).

::: psdaq.control.TimedRun
