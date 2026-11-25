# Notes for AI coding assistants working in psdaq

Before changing GPU DRP code -- `drpGpu/`, the GPU trigger primitives
(`psdaq/trigger/src/*_gpu.cu`), or the shared DRP code they call in `drp/` -- read
`docs/gpu-drp-rules.md`.  It is written for AI consumption: terse rules, each keyed to the
files it governs, each learned from a real failure.

When debugging or measuring the DAQ -- rates, deadtime, Prometheus, Slurm, libfabric --
read `docs/reading-evidence.md`: traps that lead to wrong conclusions from correct data.
