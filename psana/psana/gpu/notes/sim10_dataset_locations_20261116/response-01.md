# Response to independent review round 1

Reviewed commit: `ba80cfe10ca51c8bc1671d4fbbbfdc9b95d19d73`.
Corrected commit: `be12fc6abecd5520fd9b355240e9e4e0b6ece353`.
Reviewer verdict: pass, with two low-priority, non-blocking suggestions.
Both suggestions are accepted and implemented; round 2 will verify closure.

| Finding | Response | Evidence |
| --- | --- | --- |
| DS-001 | Accepted/fixed: distinguish external 400k evidence from the tracked 100k archive, name the generation and acceptance records, and preserve both before source cleanup. | Coordinator read the existing SDF `sample10_400k/generation.json` header and `sample10_400k_evidence_20261111/{status,verification}.json`; these record 400,000 events derived from the 10,000-event fixture and passing acceptance. No data or receipts were changed. |
| DS-002 | Accepted/fixed: replace ambiguous "private" with "SDF project-scratch" in the regression template. | The published ten-stream runs supersede those SDF fixtures for new campaigns, not the separately retained private Perlmutter JF-only subset. |

No runtime, profile, or historical archive changes. No cleanup or transfer.
