# Round 1 response

Manifest reviewed: b8dd3c561f2a3cc33deacd46f004fdd664848a7ca29f53223d06c4692080edfd.
Claude returned plan-only pass, no blocking findings, two suggestions. Both
are accepted and incorporated; the amended plan goes to round 2 rather than
inheriting approval for changed text.

| Finding | Response |
| --- | --- |
| PLAN-R1-001 | Accepted. Explicit uncopied full-image ownership; prohibit image publication; compact allowlist capped at 64 KiB/callback; record the 64 MiB pinned cap, actual copy/pinned bytes and fallback count; fail on timed-loop ordinary-host fallback. Clarified that task publication is not disabled by gpu_d2h_chunk_size=0 and that separate bounded export/acceptance copies are permitted. |
| PLAN-R1-002 | Accepted. Pin PS_JF_SHARE_DERIVED=0 across the first cohort; require frozen smd0/MPI constant-buffer provenance, per-BD device-array hash verification before conditioning/timing, and setup-time/residency records. Any later derived-cache option is a separate matched cohort. |

Coordinator checked the referenced PR298 user-kernel publication contract and
MPI shared-calibration branch before incorporating these requirements. The
pinned source Git HEAD resolves to 67c327a148a08b2aaf972e67e1e02e367c2b298f.
Round 2 includes a copy of the native compatibility receipt and a compact
coordinator hash verification record so its provenance is readable within the
snapshot. No runtime tests, live node check, or job submission occurred.
