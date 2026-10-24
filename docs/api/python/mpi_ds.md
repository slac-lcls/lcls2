# psana.psexp.mpi_ds

`MPIDataSource` is returned by `DataSource(exp=..., run=...)` on the smd0,
event-builder and big-data ranks of an MPI job with more than one rank (the
roles are assigned by `Communicators` in `psana/psana/psexp/node.py`). Its
`runs()` yields `RunParallel` objects, whose `events()` and `steps()` deliver
events on the big-data ranks. `RunParallel` also implements random access
(`build_table()`, `event(ts)`) and shares calibration constants between ranks.

See [MPI parallel analysis](../../features/mpi-parallel.md).

::: psana.psexp.mpi_ds
