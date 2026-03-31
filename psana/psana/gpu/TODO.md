## psana2-gpu TODO

### Architecture

- Add first-class GPU context control to the runtime/execution layer.
  - Current async support only covers psana2-gpu's built-in internal stages.
  - User GPU work after `det.raw.calib(evt)` is not modeled as part of the
    managed pipeline.
  - If future stages such as common mode, assembly, integration, or user
    kernels should stay device-resident and be scheduled with the same event
    pipeline, the runtime needs an explicit shared-context / shared-stream /
    shared-buffer contract.
- Spin GPU-side XTC parsing into a separate `GPUDgram` workstream.
  - Design note:
    [`GPU_DGRAM_DESIGN.md`](./GPU_DGRAM_DESIGN.md)
  - Near-term deliverables:
    minimal Jungfrau `NamesLookup` schema,
    CPU reference walker,
    and device-side structural parser views
