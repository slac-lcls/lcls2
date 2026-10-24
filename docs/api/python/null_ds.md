# psana.psexp.null_ds

`NullDataSource` is returned by `DataSource` to MPI ranks that read no data:
with `shmem=`, the first `PS_SRV_NODES` ranks; with `exp=` in MPI mode with
more than one rank, every rank that is not an smd0, eb or bd rank (see
[psana.datasource](datasource.md)). Its `runs()` yields one `NullRun`, which
has no events and no steps. `is_srv()` returns True, and `unique_user_rank()`
and `is_bd()` return False.

See [MPI parallel analysis](../../features/mpi-parallel.md).

::: psana.psexp.null_ds
