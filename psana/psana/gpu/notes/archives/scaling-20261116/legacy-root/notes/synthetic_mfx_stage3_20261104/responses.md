# Stage 3 review responses

Round 1 reviewed `eef35c7a0977ae3c2ccf11131199a4de996ef4e3` and passed with
three nonblocking safeguards. All are accepted for the next revision.

| ID | Response | Validation |
| --- | --- | --- |
| S3-ADMIN-01 | Publisher now compares generator/helper hashes in generation.json and validator/helper hashes in the verification report against exact reviewed Git blobs. Verification now records its code hashes. | `verification_reviewed.json` passes a new full verification with the revised validator; generator/helper hashes remain unchanged, so no data regeneration is needed. Three focused rejection subcases pass for wrong generator, wrong validator and missing validator hashes before auth, journal or link operations. |
| S3-VERIFY-01 | Validator now checks the source record timestamp against the source index for both SMD and bigdata, independently of output timestamp checks. | New full all-record verification of existing staging passes. |
| S3-ADMIN-02 | Preflight now requires both target directories to exist and be writable/traversable before any links or publication journal. | Focused preflight checks for missing/unwritable directories pass before auth; fresh real preflight remains required by publication. |

Generated data and the generator/helper implementation stay unchanged. The
first verification report and round-1 pass remain preserved as historical
evidence. The revised validation report is used for final publication.

Round 2 independently closed all three findings and passed commit
`3c77c8b189fcdbc99a10760325f0727069e4f249`. No unresolved findings remain.
Publication exercised the real Git-blob provenance checks and completed all
twenty links; final-directory serial smoke passed. See `round-02/review.json`,
`publication.jsonl`, and `published_smoke.json`.
