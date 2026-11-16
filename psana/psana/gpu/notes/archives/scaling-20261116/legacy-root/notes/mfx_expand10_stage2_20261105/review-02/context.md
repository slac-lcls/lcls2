# Focused evidence follow-up

Review the exact commit supplied by the harness. Implementation code remains
11f2f804a0f9968d6210d8a6bfed584691b101d3; this new commit changes only documentation,
status and evidence. Carry forward the completed code review in review-01/review.json.
No code or datasets were changed or rerun after that pass. Do not delegate.

Read notes/mfx_expand10_stage2_20261105/review-01/review.json and response.md,
README.md, validation-summary.json, commit-provenance.json and status.json.
Confirm closure of F-006 (six-event final pilot callback) and F-007 (generation
source_start_event is 0 / 36034 / 0). Inspect the evidence rather than assuming
that the prior denied scratch reads were successful.

Full final reports are now copied byte-for-byte under `build/stage2-review-evidence/`
IN THIS CHECKOUT, available to your Read/Grep tools. This ignored directory is
validation evidence, not source code. Do not attempt the original absolute
scratch paths; they are provenance references and outside your sandbox.
Use `index.json` to locate reports and coordinator-computed copy digests. Full
GPU reports include all event comparisons, actual callback order, guard tests,
code hashes and runtime provenance. Full CPU verification and generation
reports include source windows, all-stream schedule, hashes, sizes and checks.
These large files should be inspected selectively with Grep and bounded Read,
not consumed in entirety. No need to repeat the comprehensive code review unless
these evidence corrections expose a new inconsistency.

The reviewer cannot execute commands/hash computations; report that limitation.
Coordinator commit-provenance and supplied report digests bind source to evidence.
Check those values by inspection; never claim independent execution. Required
checks already completed: eight CPU tests; exhaustive source/format verification
for 10/3/1000 events; all raw and calibration bits exact on native GPU path; both
streams, partial batches, ten guard cases. Still no GDS/MPI/cube/live-DB/35kHz
claims. Return exact reviewed SHA, prior finding dispositions and any remaining
blocking evidence gap. Max three completed review rounds, this is round two.
