# Round 1 responses

Claude passed d5932e447b2ea63d7bdd3c666d2063a4926dc8cc with six non-blocking
findings. Documentation/evidence corrections are submitted for round 2; none
changes the runtime harness or production source.

| ID | Response |
| --- | --- |
| JFAB-R1-001 | Accepted need for default context; disputed generic-default inference. Both source revisions' datasource.py `_force_mfx_overrides` sets PS_SMD_N_EVENTS=1000 for MFX when absent. Generic tools.py fallback 20000 is not effective here. A new pre-fix stage1-step2 probe with the variable unset confirmed effective 1000, four chunks/callbacks, zero selected events and the same exit-1 UnboundLocalError. README now explains both defaults and the measured one-chunk workaround; default-probe.json and manifest retain evidence. Request reviewer agreement on this correction. |
| JFAB-R1-002 | Added explanation of resumed open step at callback-local index 0 and following BeginStep at local index 1 in the observed boundary chunks. Confirmed directly in the unfixed RunSmallData.steps implementation. |
| JFAB-R1-003 | Documented that per-rank resolved-path evidence covers two modules, with remaining modules selected by uniform overlay construction. Expanded provenance instrumentation is deferred to future runs; existing experiment unchanged. |
| JFAB-R1-004 | Documented that pre-fix acceptance branches were written after observing this run and are specific to this dataset/parameters, not a general regression test. Timestamp oracle and exact array comparisons remain independent of those branches. |
| JFAB-R1-005 | Documented hard-coded user prefix, overlay prerequisite and fixed artifact directory requiring a fresh run. No portability refactor in this experiment. |
| JFAB-R1-006 | Deferred. Empty-step handling is a separate application bug already described in README. This task tests the counter fix/workaround; no application edit or additional issue creation was performed. |

The initial review assumed the generic default while overlooking the MFX
override. Its pass verdict does not make that inference authoritative; the
follow-up review must assess the corrected source-backed explanation.
