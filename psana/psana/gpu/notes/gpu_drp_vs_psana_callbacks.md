# GPU DRP Versus Psana GPU Callbacks

## Summary

The LCLS GPU DRP has a host-independent CUDA launch path for a constrained
real-time data plane. It does not make the whole DAQ CPU-independent.

The important distinction is:

- A psana Python callback normally performs Python dispatch and host CUDA
  submission for each event.
- The GPU DRP prebuilds CUDA graphs, launches them during Configure, and lets
  device code tail-launch the graphs again as new work arrives.

This can remove per-event Python and host CUDA-launch latency from FPGA input,
calibration, trigger-primitive processing, and graph-capable reduction. The
tradeoff is a much narrower programming model: compiled CUDA algorithms,
preallocated stable buffers, fixed graph topology, explicit queues, and strict
completion and buffer-lifetime rules.

This investigation used `origin/master` commit
`5759d821dd7ff22f9f62945bc8fe33b3904fb8f8`. The related
`origin/features/gpu` branch differed mainly by a reducer-instance refactor;
the scheduling architecture was already on master.

## GPU DRP Call Path

```text
FPGA
  | GPUDirect RDMA
  v
GPU DMA buffers
  |
  +-> Reader graph: wait for DMA -> calibrate -> rearm FPGA
  |                                      |
  |                                      v
  |                             calibrated buffer
  |                                      |
  +-> Trigger graph: trigger primitive / peak finding
  |                                      |
  |                            pinned host result queue
  |                                      v
  |                           CPU builds TEB datagram
  |                                      |
  |                                TEB decision
  |                                      |
  +-> Reducer graph <- CPU publishes selected buffer index
             |
             v
       reduction/compression
             |
       CPU recorder/file writer
```

### Initialization And FPGA Input

`MemPoolGpu`:

1. Creates and selects a CUDA context.
2. Requires GPUDirect RDMA and host-memory mapping support.
3. Opens `/dev/datadev_*` and maps the FPGA control registers.
4. Allocates FPGA DMA destinations with `cudaMalloc`.
5. Registers those GPU allocations with the FPGA driver through
   `gpuAddNvidiaMemory()`.
6. Maps FPGA control registers into CUDA address space so the GPU can rearm DMA
   buffers directly.

The FPGA therefore writes detector data directly into device memory rather
than sending the payload through CPU memory. It also writes a size/doorbell
word that makes DMA completion visible to the GPU.

Source:
[MemPool.cc](https://github.com/slac-lcls/lcls2/blob/5759d821dd7ff22f9f62945bc8fe33b3904fb8f8/psdaq/drpGpu/MemPool.cc#L44-L245).

### GPU-Resident Scheduling

During Configure, psdaq captures, instantiates for device launch, and uploads
three graph loops:

1. **Reader graph**
   - Polls the FPGA-written doorbell.
   - Calibrates raw data into an intermediate GPU buffer.
   - Publishes the buffer index, clears the doorbell, and rearms the FPGA.
   - Tail-launches itself.

2. **Trigger input generator graph**
   - Polls the Reader's device queue.
   - Runs the compiled trigger primitive.
   - Publishes an index to a host-visible queue.
   - Tail-launches itself.

3. **Graph-capable reducer**
   - Polls a host-to-device queue for a TEB-selected event.
   - Runs reduction or compression.
   - Publishes completion to a device-to-host queue.
   - Tail-launches itself.

The host launches each graph once during Configure. A final graph kernel calls:

```cpp
cudaGraphLaunch(cudaGetCurrentGraphExec(),
                cudaStreamGraphTailLaunch);
```

Sources:
[Reader.cu](https://github.com/slac-lcls/lcls2/blob/5759d821dd7ff22f9f62945bc8fe33b3904fb8f8/psdaq/drpGpu/Reader.cu#L290-L657),
[TrgInpGen.cu](https://github.com/slac-lcls/lcls2/blob/5759d821dd7ff22f9f62945bc8fe33b3904fb8f8/psdaq/drpGpu/TrgInpGen.cu#L222-L380), and
[Reducer.cu](https://github.com/slac-lcls/lcls2/blob/5759d821dd7ff22f9f62945bc8fe33b3904fb8f8/psdaq/drpGpu/Reducer.cu#L336-L534).

This is a real GPU-resident scheduler, but it is not an arbitrary list of
Python functions. Device-launchable CUDA graphs have fixed structure and
restrictions on graph nodes, memory, and updates.

## CPU Work That Remains

The GPU DRP still performs substantial CPU work:

- A receiver thread consumes GPU-produced indices and validates DMA status,
  pulse IDs, event counters, transitions, and readout groups.
- A collector constructs `EbDgram` objects and sends trigger input to the TEB.
- The TEB receiver evaluates the trigger result and publishes selected buffer
  indices to resident reducer graphs.
- A recorder waits for reducer completion, releases buffers, handles monitoring,
  and writes data.
- Configuration, transitions, plugin loading, networking, file handling, and
  shutdown are CPU-managed.

Current master also contains explicit `cudaStreamSynchronize()` calls for:

- D2H monitoring data before posting it to CPU consumers.
- Asynchronous file-writer buffer reuse and completion checks.
- Reducers implemented as raw kernels rather than device-resident graphs.

The recorder spin-polls reducer completion queues. The GPU also polls its DMA
and work queues with short sleep/backoff kernels. Thus neither side is simply
idle while the other independently processes an unconstrained workload.

Source:
[PGPDetector.cc](https://github.com/slac-lcls/lcls2/blob/5759d821dd7ff22f9f62945bc8fe33b3904fb8f8/psdaq/drpGpu/PGPDetector.cc#L141-L270).

## Comparison With Psana Callbacks

| Property | Psana Python callback | GPU DRP graph |
| --- | --- | --- |
| Python dispatch per event | Yes | No |
| Host CUDA submission per event | Usually yes | No for resident graph stages |
| Arbitrary Python control flow | Yes | No |
| Stable preallocated buffers | Not required | Required |
| Fixed graph topology | Not required | Required |
| Direct FPGA-to-GPU input | No | Yes |
| GPU polls for work | No | Yes |
| CPU event bookkeeping | User/psana event loop | Still present in the DRP |

CUDA operations launched by a psana callback may be asynchronous, but the CPU
still invokes Python and submits the operation or graph replay. Asynchronous
execution avoids waiting for completion; it does not eliminate host scheduling.

The GPU DRP removes that host submission by combining:

- Direct FPGA DMA into registered GPU memory.
- GPU access to FPGA MMIO for buffer rearming.
- Device-tail-launched CUDA graphs.
- Preallocated buffer pools and fixed pointer lifetimes.
- Host/device coherent pinned queues with atomic completion protocols.
- Compiled trigger and reducer interfaces that can be captured into graphs.

Supporting the same model in psana would therefore require a batch-oriented,
graph-compatible stage contract. It would not be an automatic optimization of
ordinary `gpu_callbacks=[python_function, ...]`.

For ordinary callback semantics, see
[gpu_callbacks_capacity_model.md](gpu_callbacks_capacity_model.md).

## Nsight Systems Evidence

Three existing captures were found on `drp-srcf-gpu001`:

- `/sdf/home/c/claus/tmp/report15.nsys-rep`, October 2025.
- `/sdf/home/c/claus/tmp/report18.nsys-rep`, December 2025.
- `/sdf/home/c/claus/tmp/report19.nsys-rep`, December 2025.

`report18` captured a real `/dev/datadev_1` run using `epixuhremu`, two reducer
workers, and `PfplReducer`. Its extracted timeline showed two initial host graph
launches followed by long-lived ingress kernels:

```text
GPU:
  t=0 ms       _handleDMA   --------------------- 1148.6 ms
  t=0.05 ms    _collector   --------------------- 1148.5 ms

CPU ReducerWkr0/1, later:
  cudaMemcpyAsync
  cudaStreamSynchronize
  cudaGraphLaunch
  GPU: _receive -> d_reset -> d_encode -> _graphLoop
  cudaStreamSynchronize
```

The capture contained:

- Two initial graph launches for the long-lived ingress kernels.
- Ten later reducer graph launches from CPU reducer threads.
- Twenty `cudaStreamSynchronize()` calls.
- Approximately 2--9 microseconds per stream synchronization in this light,
  roughly 1 Hz sample.
- Long condition-variable waits in reducer and recorder threads while waiting
  for work.

`report15` contained 632 reducer executions, 641 host `cudaGraphLaunch()` calls,
and 1,264 stream synchronizations. The older reducer implementation was
therefore host-scheduled per event.

These reports predate the April 2026 separate-kernel, self-tail-launching
refactor. They validate autonomous FPGA ingress in the older persistent-kernel
implementation, but they do not measure the current resident reducer graph.
Current-master performance remains unverified by a matching timeline.

## Recommended Current-Master Capture

Nsight Systems 2026.1 on the GPU DRP host supports tracing graphs launched from
both host and device. Start with graph-level tracing to minimize overhead:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cuda-graph-trace=graph:host-and-device \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o /tmp/drp_gpu_current \
  <normal drp_gpu command>
```

Current master defines `NVTX_DISABLE`. A profiling build should re-enable NVTX
if named CPU ranges are needed. A new capture must be coordinated with a test
DAQ partition and FPGA ownership rather than attached to an uncoordinated
production process.

## Conclusion

The strongest defensible statement is:

> The GPU DRP has a host-independent CUDA launch path for a constrained,
> preconfigured data plane.

It is too strong to say that the GPU runs independently of the CPU. The CPU
still performs event validation, TEB communication, reducer admission, output
handling, and buffer retirement. The model may reduce launch latency and jitter
at high rate, but its performance advantage over psana callbacks must be
measured against current code and includes the cost of GPU polling, reserved
resources, queue traffic, and remaining CPU work.
