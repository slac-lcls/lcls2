# psana.psexp.singlefile_ds

`SingleFileDataSource` is returned by `DataSource(files=...)`. `files` is one
xtc2 file name or a list of names (a single string is turned into a list by
[`DataSourceBase`](ds_base.md)). The files are read one after another, and
small-data files are not used. A file that does not exist raises
`FileNotFoundError` when it is opened: the first file in the constructor, the
others when `runs()` reaches them. `runs()` yields one `RunSingleFile` (see
[psana.psexp.run](run.md)) for each BeginRun.

See [DataSource, Run and the event loop](../../features/datasource-run-events.md).

::: psana.psexp.singlefile_ds
