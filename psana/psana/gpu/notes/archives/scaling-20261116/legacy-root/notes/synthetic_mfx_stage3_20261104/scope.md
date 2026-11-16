# Stage 3: 60-event synthetic MFX pilot

Authorized by Mona: “Move on to stage 3 with review loop.” Base commit
`81c88de94`, branch `work/synthetic-mfx-scan`, remote `monarin/lcls2`.

Implement and run a bounded writer for prjdatpsana run 1: ten XTC streams and
matching SMD files, first 60 source events in steps [19, 20, 21], frozen source
SMD and the original public bigdata. Preserve event timestamps and payloads,
add standard scan Names/BeginStep data to s000, rewrite all BeginRun identities,
clone same-service per-stream transition templates, rebuild SMD offsets.

Validate every generated record, source-byte correspondence, offsets, scan and
run metadata, stream alignment, state machine, counts and strict timestamp
ordering. Exercise nanosecond rollover and insufficient-gap rejection. Generate
into a private staging directory; publish final names without overwriting files
only after structural validation and independent review. Save durable hashes,
source mapping, generator/runtime identities, verification and publication
records. Check current run/file collisions before generation and publication.

Do not change psana core or calibration DB records. Stage 4 retains calibrated
image comparisons, serial/MPI pipeline acceptance and collaborator-login tests.
Run 2/full production fixture is outside this stage. A small calibration-skipped
DataSource smoke check may supplement structural validation without claiming
Stage 4 completion. No logbook registration or messages to collaborators.

Use implement-review-loop, default three review rounds, Claude read-only review
of an exact committed implementation with validation evidence. Review feedback
may correct this writer/validator but does not expand dataset size or authorize
new deployments. Persist status and final reviewer SHA; stop on the skill's
round/access/scope limits without treating failed reviews as approval.
