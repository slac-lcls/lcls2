# Stage 1–2 audit response and review loop

User request: address the supplied Claude audit of the current synthetic MFX
progress, then continue with the implement-review-loop skill.

Base: `e118b3c7eec28203f06a1c9594306d59b58af4c5` on
`work/synthetic-mfx-scan`, remote `monarin/lcls2`.

Scope: refine the Stage 3 construction and Stage 4/5 acceptance requirements;
verify source prerequisites and current branch/access evidence; record a
response to each of the seven supplied findings; obtain independent Claude
review and address scoped corrections, up to three new review rounds.

Stages 1 and 2 remain complete. This task does not implement or execute the
Stage 3 writer, generate the large fixture, modify psana core, redeploy existing
calibration records, change ACLs/memberships, register files, or message others.
The user has not yet requested Stage 3 generation. A design requirement is not
evidence that generated XTC or EventBuilder behavior has passed.

Acceptance: all seven findings have evidence-based dispositions; the plan
specifies scan schema/payloads, all-stream run identity, full transition-header
construction, PV provenance/run-2 calibration, access gates, and validation
environment/discovery. Frozen SMD prerequisites pass; Git checkpoint is verified;
remaining operational acceptance checks are explicit. Claude reviews the exact
committed revision, with no blocking findings for this audit-response scope.

Review outputs and responses live here. Only the coordinator updates the task
branch. The reviewer uses a clean detached checkout and read-only file tools.
