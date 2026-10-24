# psana.psexp.serial_ds

`SerialDataSource` is returned by `DataSource(exp=..., run=...)` when
`PS_PARALLEL=none` or when the job has only one MPI rank (see
[psana.datasource](datasource.md)). It reads the runs on a single process.
The constructor builds the run-number list (every run found in the xtc
directory when `run` is not given), finds the small-data and big-data files of
the first run, reads the Configure dgrams from the small-data files, and opens
the big-data files with a `DgramManager`. `runs()` yields one `RunSerial` (see
[psana.psexp.run](run.md)) for each BeginRun. Keyword arguments are read by
[`DataSourceBase`](ds_base.md).

See [DataSource, Run and the event loop](../../features/datasource-run-events.md).

::: psana.psexp.serial_ds
