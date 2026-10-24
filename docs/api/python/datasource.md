# psana.datasource

`DataSource(*args, **kwargs)` is the entry point of psana2. It is a function,
not a class: it looks at the keyword arguments (checked in the order `shmem`,
`exp`, `files`, `drp`) and returns an instance of one of the data source
classes, all derived from
[`DataSourceBase`](ds_base.md):

| Keyword | Returned class | Module |
|---|---|---|
| `shmem=` | `ShmemDataSource` (`NullDataSource` on the first `PS_SRV_NODES` MPI ranks) | [`psana.psexp.shmem_ds`](shmem_ds.md), [`psana.psexp.null_ds`](null_ds.md) |
| `exp=` (and `run=`) | `SerialDataSource` when `PS_PARALLEL=none` or only one MPI rank; otherwise `MPIDataSource` on smd0/eb/bd ranks and `NullDataSource` on the other ranks | [`psana.psexp.serial_ds`](serial_ds.md), [`psana.psexp.mpi_ds`](mpi_ds.md), [`psana.psexp.null_ds`](null_ds.md) |
| `files=` | `SingleFileDataSource` | [`psana.psexp.singlefile_ds`](singlefile_ds.md) |
| `drp=` | `DrpDataSource` (its constructor currently raises AttributeError; see drp_ds) | [`psana.psexp.drp_ds`](drp_ds.md) |

`PS_PARALLEL` is read in `psana/psana/psexp/tools.py` (default `"mpi"`). If
none of these keywords is given, `InvalidDataSource` is raised. With `exp=` in
MPI mode with one rank, `DataSource` creates the `SerialDataSource` inside a
`try`: only if that raises `FileNotFoundError` does it log the error and call
`sys.exit(1)`. With `PS_PARALLEL=none` there is no such `try`, and the
exception reaches the caller. The docstring below says the same thing in a
shorter form. Importing this
module also sets the environment variable `PS_PROMETHEUS_JOBID`.

See [DataSource, Run and the event loop](../../features/datasource-run-events.md)
for how the returned object is used.

::: psana.datasource
