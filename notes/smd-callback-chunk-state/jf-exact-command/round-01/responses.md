# Round 1 responses

Reviewed SHA: b6628b0735a0723af5e2c9897d6b42605c943690.
Verdict: pass. All findings low priority and non-blocking. No scripts or
production code changed after review. Subsequent changes are review records
and documentation clarifications only; no runtime rerun is necessary.

| ID | Response |
| --- | --- |
| JFEX-R1-001 | Deferred future-reuse hardening. The three reports confirm observation worked; analyze.py already requires datasource_class and exact callback counts. No harness change needed for this completed experiment. |
| JFEX-R1-002 | Added README clarification that view sizes include transitions and explain the tail chunk. The actual serial dispatch also uses the CLI's UtilsJungfrauCalibMPI dispatcher with MPI world size 1; no mpirun was used. |
| JFEX-R1-003 | Added round-01 context/review/responses and final status with exact reviewed SHA. |
| JFAB-R1-004 | Added README disclosure that exact chunk/callback count expectations were post-hoc and dataset-specific. MFX default1000 was known from source and the previous unset probe before these runs; the new runs confirm it. Oracle timestamp and numerical gate comparisons remain separate checks. |
| JFAB-R1-005 | Reviewer marks resolved: reproduction paths and prerequisites documented. |
| JFAB-R1-006 | Remains a documented, deferred application follow-up: empty-step ievt logging failure in serial/MPI loops. No application edit or additional issue was requested in this experiment. |
| JFAB-R2-001 | Reviewer marks resolved: committed environment setup plus initial/effective settings. |

Coordinator validation: all three expected outcomes passed, no timeouts, no
calibration-write helper calls; same ten streams; both successful cases read
exactly the oracle's first50 events in step2 across32panels and produced exactly
equal uint16 gate arrays. Byte compilation, whitespace checks, and commit hooks
passed. Reviewer independently inspected scripts/source/committed reports but
could not execute tests, load arrays, or recompute external artifact hashes;
runtime results remain coordinator evidence, as disclosed in review.json.
