# Stage 2 scope

Implement a native psana GpuTask adapter for sim10.seed, materialize all 32
uint16 panels, and reuse the unchanged Jungfrau calibration example. Compare
all raw pixels and calibrated float32 bits against CPU v3 on expanded input.

Gate on 10 events, then 1,000 events and a three-event tail sample with original
native closing transitions. Source run 10 has only one step; no internal native
step boundary exists. Add no artificial steps. Exercise batched callbacks,
both CUDA streams, and partial final batches. Record file sizes and memory.

Close Stage 1 F-002 (Shapes-before-Data guard) and F-004 (writer/helper/manifest
provenance checks) in a hardened Stage 2 writer, preserving Stage 1 artifacts.
Use frozen MFX pedestal/gain/offset bytes. No psana core edits, DB deployment,
public run assignment, cube work or throughput claims. Use private scratch,
one A100, native GPU input/parser/task path and KvikIO compatibility mode.

Independent restricted Claude review of the exact implementation commit,
maximum three completed rounds. Branch work/synthetic-mfx-scan, base 1b7fd499e.
Commit/push scoped code and compact evidence; keep full reports in scratch.
