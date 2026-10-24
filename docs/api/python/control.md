# psdaq.control.control

The DAQ control process (`control` command). `CollectionManager` runs the
state machine (states and transitions from `ControlDef`), talks to the DRP,
TEB and MEB processes over ZMQ, answers client requests (`DaqControl`,
`daqstate`, the GUI) and drives the XPM through `DaqXPM`. `RunParams` reads
the run-parameter PVs recorded at BeginRun; `main()` parses the command line.

See [DAQ control](../../features/daq-control.md).

::: psdaq.control.control
