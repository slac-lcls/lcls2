## psana2-gpu Performance Notes

This note summarizes the current `cupy + 3stage` results on staged Jungfrau run
data and the main conclusions from profiling.

### Measured Baselines

Using staged local data for `mfx101344525:r125`:

- CPU raw/event loop (`ds_count_events.py`, `det.raw.raw`): about `70 Hz`
- CPU calibration (`ds_count_events.py --calib`): about `25 Hz`
- GPU calibration (`validate_jungfrau_gpu.py`, device-backed output): about
  `49 Hz`

These numbers imply:

- storage + psana event delivery can already feed the pipeline faster than the
  current GPU path
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
