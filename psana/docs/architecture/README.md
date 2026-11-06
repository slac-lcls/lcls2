# Psana architecture

Status: **implemented-reference**. Reviewed source:
`410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8`, 2026-10-15,
on `features/psana2-gpu`. Source paths: the complete list is in
[`event-flow/workflow-model.json`](event-flow/workflow-model.json),
under `baseline.source_paths`.

Offline experiment/run processing follows small-data reading, timestamp-aligned
event building, bigdata processing and public Event delivery. In MPI mode these
responsibilities span SMD0, EB and BD processes. Serial mode performs the relevant
work locally.

A GPU BD is a CPU process that orchestrates reads into GPU storage, parsing,
user-task submission and result delivery. The diagrams distinguish the host
calls, data transfers, CUDA dependencies and buffer ownership involved.

## Explore one event

[Open or download the interactive event flow](event-flow/index.html).
On GitHub, download the raw HTML and open it in a browser; a hosted URL is not
configured yet. Use Overview, CPU path or GPU path, select a component, then
inspect its call trees or follow event T.

The viewer is a self-contained document and opens as a separate page. Its source
links point to the inspected commit, and its provenance stays with downloaded
copies. The transition example distinguishes the CPU path, current GPU limitations
and the proposed ordering contract.

For deeper references, see the [GPU pipeline design](../../psana/gpu/docs/design.md),
[GPU XTC parser](../../psana/gpu/docs/gpu_xtc_parser.md) and
[external task API](../../psana/gpu/docs/user_kernels.md).

See [viewer maintenance](event-flow/README.md) for generation and validation,
or return to [psana documentation](../README.md).
