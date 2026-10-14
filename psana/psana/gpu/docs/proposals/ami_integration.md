# AMI integration: open design questions

**Proposed, not implemented.** This is a future integration topic; the current
psana API is described in [design](../design.md) and [user kernels](../user_kernels.md).
The earlier detailed proposal is available in
[Git history](https://github.com/slac-lcls/lcls2/blob/fa40ec52a/psana/psana/gpu/docs/proposals/ami_integration.md).
Its automatic calibration and device-output sketches are superseded.

The implemented GPU path handles experiment/run serial and MPI input. It does
not provide a GPU shmem/DRP path or a task-output device handoff for AMI.
A practical design must resolve:

- Whether the first integration consumes existing named host results offline
  or requires GPU-resident values across AMI graph operations.
- How device arrays and completion dependencies are represented and retained
  across graph consumers, including fan-out, cancellation and cleanup.
- How online shared-memory source ownership is connected to GPU input lifetime
  before the DAQ ring can reuse its storage.
- Which process owns CUDA, quotas and device assignment, and how backpressure
  propagates without assuming one worker has exclusive use of the GPU.

Any prototype should reuse the current parser/input/task contracts and validate
source lifetime, failures and bounded memory. No AMI type/node API or new
DataSource mode is committed by this note.
