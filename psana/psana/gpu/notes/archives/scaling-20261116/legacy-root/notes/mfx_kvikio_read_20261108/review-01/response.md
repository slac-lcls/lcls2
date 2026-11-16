# Round 1 response

Reviewed executable commit: 70359e3b894206f2e9c9f2d2abb4740805a3a87b.
Verdict changes_requested: KR-01 is the only blocking finding. The reviewer
found implementation/accounting sound and all measurements consistent.
Corrections are documentation/evidence only; executable files and native reports
remain unchanged. No additional GPU run is needed.

| ID | Response |
|---|---|
| KR-01 | Fixed: headline, confirmation label and interpretation use four-worker control with explicit task/bounce settings. README explicitly says old effective task/bounce sizes were not captured, and these values are now explicitly set/verified installed defaults. |
| KR-02 | Fixed: explicitly state the control and eight-worker row differ in worker and affinity counts, preventing isolation of either effect. Affinity IDs are logical CPUs; physical-core/sibling topology was not recorded. |
| KR-03 | Accepted the misleading-inference concern. README and new metric-definitions.json label both legacy fields as descriptor-flag censuses with no actual I/O-path evidentiary value. Native reports remain immutable. Executable schema/key rename is explicitly deferred (non-blocking), avoiding an unnecessary measurement rerun merely to rename a field. |
| KR-04 | Fixed: prefixed tests.log with coordinator-recorded exact interpreter, invocation, working directory and KVIKIO_READ_NATIVE_ROOT. Original unittest output is unchanged. |
| KR-05 | Fixed via reviewer's documentation alternative: mark size extrema as manifest-derived, not native report fields. Added request-size-evidence.json from fresh coordinator manifest readback with manifest SHA, per-file extrema and event byte totals. |

Post-commit independent audit reproduced results.json exactly. The executable
code/source hashes still match the measured cases. Round 2 should confirm the
blocking historical-settings wording is resolved and check these corrections.
