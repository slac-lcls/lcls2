## psana2-gpu Performance Notes

This note summarizes the current `cupy + 3stage` results on staged Jungfrau run
data and the main conclusions from profiling.

### Measured Baselines

Using staged local data for `mfx101344525:r125`:

- CPU event loop only (`ds_count_events.py`, no detector call): about `70 Hz`
- CPU `det.raw.raw(evt)`: about `50 Hz`
- CPU calibration (`ds_count_events.py --calib`): about `25 Hz`
- GPU calibration (`validate_jungfrau_gpu.py`, device-backed output): about
  `49 Hz`

Important note on the `70 Hz` event-loop number:

- this is not a "no psana work" baseline
- event creation already includes `Event._complete()`, which calls
  `_assign_det_segments()`
- so the `70 Hz` baseline already includes event assembly and detector-segment
  dictionary construction
- it does **not** include the later `det.raw.raw(evt)` call

These numbers imply:

- storage + psana event delivery plus detector-segment construction can already
  feed the pipeline faster than the current GPU path
- adding `det.raw.raw(evt)` reduces the CPU baseline from about `70 Hz` to
  about `50 Hz`
- the GPU path is substantially better than CPU calibration
- the GPU path is still below the raw-read ceiling

### Current Model

The current implementation is:

- runtime: `cupy`
- pipeline: `3stage`
- submission unit: one event at a time
- async behavior: later `stage1` work can overlap with earlier in-flight GPU
  work, but the pipeline is still event-oriented rather than batched

Important characteristics of the current design:

- psana reads bulk data on the CPU
- detector segment data is packed into a reusable host slot buffer
- each event is copied to device separately
- calibration is implemented as a sequence of CuPy kernels
- each event currently incurs a device-to-device output copy to protect buffer
  lifetime

### Queue Depth Findings

Increasing `gpu_queue_depth` did not materially improve throughput.

Observed runs on the same staged dataset:

- `gpu_queue_depth=3`: about `37.3 evt/s`
- `gpu_queue_depth=6`: about `35.6 evt/s`

Conclusion:

- queue depth is not the main limiter in the current model
- the pipeline is not obviously starving for more in-flight slots

### Current Bottleneck Assessment

The main bottlenecks in the current `cupy + async 3stage` path are most likely:

1. CPU-side per-event preparation
   - psana read path
   - dgram / detector segment handling
   - host staging into the slot buffer

2. Per-event host-to-device transfer
   - one event copied at a time
   - no batching of multiple events into one transfer

3. CuPy small-kernel calibration path
   - the current calibration path is a chain of many CuPy kernels rather than a
     tighter fused/custom kernel

4. Per-event device-to-device output copy
   - used to isolate event result lifetime from slot-buffer reuse

5. Limited effective overlap
   - the async control flow is in place
   - measured throughput does not yet show a strong overlap win

### What Is Probably Not The Main Bottleneck

- Weka/network read bandwidth for the staged-data experiments
- queue depth
- raw PCIe bandwidth saturation at the current rates

At the current measured rates, the GPU path appears more limited by per-event
software/data-motion overhead than by hardware ceilings.

### Nsight Observations

Nsight Systems runs showed:

- the same basic memory traffic pattern before and after the async refactor
- one host-to-device copy path per event plus cache uploads
- one device-to-device copy per event
- multiple small CuPy kernels per event

The async refactor changed control flow, but did not yet show a clear reduction
in copy count or a dramatic throughput improvement.

### Bulk-Data Redesign Focus

The most interesting areas for redesign from the current bottleneck picture are:

- psana read path
- dgram / detector-segment handling
- host staging into the GPU slot buffer
- batching or otherwise reducing per-event transfer overhead

The project note suggested a "GPU bigdata core" idea: after the CPU-side event
builder provides offsets, a GPU-side worker reads bulk data, then provides the
equivalent of `det.raw.raw(evt)`.

That is directionally a good idea, but only if it is implemented with the right
division of responsibility.

#### The Wrong Model

The wrong model is:

- GPU reads detector payload
- GPU creates full `Dgram`-like objects
- GPU recreates detector segment dictionaries
- normal Python detector code consumes those objects unchanged

This fights the current psana design because:

- `Dgram(view=...)` is built around host-accessible buffers
- `Event._assign_det_segments()` builds Python dictionaries of detector objects
- `DetectorImpl._segments(evt)` expects those CPU/Python structures

Trying to mirror that full object model on the GPU would add complexity without
removing the real hot-path costs.

#### The Right Model

The recommended model is:

- CPU remains the control plane
- GPU handles the bulk detector payload path
- do not require full `Dgram` / `_det_segments` semantics for the GPU hot path

In practice:

- CPU event metadata path still handles:
  - timestamps
  - transitions
  - config/cache identity
  - stream routing
  - payload offsets and sizes
- GPU-side detector path handles:
  - bulk detector payload fetch
  - raw unpack / stack
  - calibration and later GPU stages

Conceptually:

```text
CPU EB / metadata path
    ->
GPU-aware payload reader
    ->
GPU raw buffer
    ->
GPU raw/calib path
    ->
small result to CPU or keep on device
```

This means the real abstraction should be closer to a lightweight event/payload
descriptor than a GPU reimplementation of a full `Dgram`.

#### Practical Implementation Path

The most practical path is phased:

1. Keep the current CPU control plane.
   - EB/SMD/event metadata remains CPU-managed.

2. Add a GPU-aware detector payload reader.
   - first version: CPU `pread` into pinned host memory, then async H2D
   - later version: direct storage-to-GPU if the system stack supports it

3. Let the GPU detector backend consume payload descriptors directly.
   - avoid depending on `det_raw._segments(evt)` for the GPU fast path
   - avoid host-side detector segment packing for hot detector data

4. Only then consider batching multiple events into one transfer.
   - batching can help, but it should follow the payload-path redesign
   - batching alone will not solve the object-model mismatch

#### Summary

The "GPU bigdata core" idea is a good direction if interpreted as:

- CPU control plane
- GPU detector payload plane

It is not a good idea if interpreted as:

- moving the current full `Dgram -> _det_segments -> DetectorImpl._segments()`
  model onto the GPU unchanged

The first major win is likely to come from bypassing CPU-side payload handling
and host staging for the GPU detector path, not from trying to reproduce the
existing Python object graph on the device.

### Short Conclusion

The current `cupy + async 3stage` model is a meaningful improvement over CPU
calibration, but it is still below the raw-read ceiling because of per-event
staging, transfer, kernel-launch, and lifetime-isolation overheads.

The next likely performance gains will come from:

- reducing CPU-side staging work
- reducing or removing the per-event device-to-device output copy
- using a more fused/custom compute path
- achieving real overlap, not just async-capable control flow
- eventually exploring direct bulk-data read into GPU memory
