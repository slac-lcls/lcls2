# psana.psexp.drp_ds

`DataSource(drp=...)` calls `DrpDataSource` when none of `shmem`, `exp` or
`files` is given (see [psana.datasource](datasource.md)). As written, the first
line after `DataSourceBase.__init__` reads `self.drp`, which `DataSourceBase`
does not set, so the call raises AttributeError (see Notes below). The rest of
the class reads dgrams through `DgramManager(["drp"])`, and its `runs()` yields
one `RunDrp` (see [psana.psexp.run](run.md)) for each BeginRun. The constructor
also reads attributes of the `drp` keyword argument; the other keyword arguments
are read by [`DataSourceBase`](ds_base.md).

See [DRP and event building](../../features/drp-eventbuilding.md).

::: psana.psexp.drp_ds
