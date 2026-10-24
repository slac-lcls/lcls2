# psana.smalldata

`SmallData` is the object returned by `ds.smalldata(filename=..., batch_size=..., callbacks=[...])`.
On big-data ranks it collects per-event values (`event()`) and summary data
(`save_summary()`, reductions such as `sum()`), and sends them in batches to
`Server` objects on the `PS_SRV_NODES` server ranks, which write HDF5 files
with h5py. In parallel mode each server writes `<base>_part<N>.h5` and
`done()` joins them into one file with HDF5 virtual datasets; in serial mode
the file is written directly.

See [MPI parallel analysis](../../features/mpi-parallel.md).

::: psana.smalldata
