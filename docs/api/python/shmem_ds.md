# psana.psexp.shmem_ds

`ShmemDataSource` is returned by `DataSource(shmem=tag)`. It reads built
events from a shared-memory server (the DAQ's monitoring event builder
`monReqServer`, or the `shmemServer` test tool) through
`DgramManager(["shmem"], tag=...)`, and its `runs()` yields `RunShmem`
objects. The optional `supervisor` / `supervisor_ip_addr` arguments let one
process fetch calibration constants and broadcast them to the others over ZMQ.

See [Shared memory and live analysis](../../features/shmem-live.md).

::: psana.psexp.shmem_ds
