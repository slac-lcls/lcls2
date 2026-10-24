# psana.psexp.ds_base

`DataSourceBase` is the abstract base class of every data source returned by
[`DataSource`](datasource.md) (`SerialDataSource`, `SingleFileDataSource`,
`MPIDataSource`, `ShmemDataSource`, `DrpDataSource`, `NullDataSource`). Its
constructor reads all the keyword arguments that `DataSource` accepts (`exp`,
`run`, `dir`, `files`, `max_events`, `batch_size`, `detectors`, `xdetectors`,
`small_xtc`, `timestamps`, `live`, `skip_calib_load`, ...); unknown keywords
are logged as "Unrecognized kwarg" and ignored. It also finds the xtc2 and
small-data files of a run and provides `smalldata()`, `unique_user_rank()`,
`is_bd()` and `is_srv()`. Subclasses implement `runs()` and `is_mpi()`.

`DsParms` is the dataclass of parameters that is passed from the data source
to its runs.

The keyword arguments, with their defaults, are listed under "Other
Parameters" in the `DataSourceBase` docstring below. A shorter table of the
commonly used keywords is in
[DataSource, Run and the event loop](../../features/datasource-run-events.md#datasourcekwargs).

See [DataSource, Run and the event loop](../../features/datasource-run-events.md).

::: psana.psexp.ds_base
