# Independent Stage 2 review

Review the exact commit recorded in `status.json` (and passed to the helper).
Base: `1b7fd499e24feddf7839b3e200ead1332a8b2914`.
Scope: `notes/mfx_expand10_stage2_20261105/scope.md` and README. The user explicitly
requested Stage 2 and the session-managed implement/review loop. Review only;
do not launch another agent or review loop. No shell/edits/MCP permitted.

Inspect all source in `notes/mfx_expand10_stage2_20261105/`, the updated proposal
`psana/psana/gpu/docs/proposals/synthetic_mfx_expand10.md`, and relevant existing
framework contracts. No psana core code changed. Check descriptor ABI and bounds,
physical panel mapping, CUDA ownership/stream/result lifetime, materialized full
raw, calibration equivalence, source-derived transition schedule, fixture/SMD
integrity, provenance and honesty of validation claims. Check closure of Stage 1
F-002 and F-004. Historical Stage 1 code/artifacts remain intentionally frozen.

Relevant project guidance: generic fields use device descriptors backed by input
slot leases; tasks may not retain borrowed views past callback completion. Own
allocations must be kept alive/published before submitted CUDA work. Use supplied
stream and byte-safe reads for unaligned XTC. Physical panel IDs must map to full
calibration arrays. Input-slot memory budget does not cover every user buffer.
See current gpu_task_batch.py, gpu_events.py, gpudgram/batch.py and unchanged
examples/jungfrau_calibration.py. Source run 10 has only one native step; tail
fixture exercises actual closing transitions, not an internal step crossing.

`validation-summary.json` contains all final CPU/GPU checks, scratch artifact
paths/hashes and current source hash audit. Full reports are read-only accessible
in `/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage2-r1/`.
Coordinator ran eight focused writer tests; exhaustive CPU source/format checks
for 10, 3 and 1,000 events; and native GPU raw/calibration bitwise checks on all
pixels with both streams, multiple batches and partial final callbacks. Ten GPU
malformed/unaligned descriptor cases are included. Check supplied evidence;
unavailable evidence is a limitation, not a pass.

Earlier failed GPU attempts are retained in status.json: missing process-local
reader registration, too-late offline calibration loading, missing KvikIO in the
initial environment. One queued job was cancelled before running to correct CPU
reference shape handling. Successful final runs use the recorded qualified
CuPy/KvikIO environment and latest driver. tiny-single-result.json is historical
successful single-event-callback evidence predating only the bulk-read grouping
change; use the final summary for current-code provenance.

Limitations are intentional: one GPU, serial DataSource, KvikIO compatibility
mode ON; full D2H and CPU checking on every event; frozen constants through a
constructor-scoped test override (not live DB lookup). No GDS, MPI, cube, 35 kHz,
new public run, or calibration deployment acceptance. User arrays are bounded by
batch size <=16 and two streams; reported CuPy pool peak is not total GPU memory.

Return stable finding IDs, severity, exact reviewed SHA and verdict. Separate
blocking correctness/acceptance gaps from optional low-priority follow-ups.
