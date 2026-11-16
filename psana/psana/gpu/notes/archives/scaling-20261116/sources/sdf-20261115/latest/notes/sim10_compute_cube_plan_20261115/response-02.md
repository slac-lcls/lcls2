# Round 2 response and closure

Claude returned **pass, no blocking findings** for manifest
`c7e434de713ad13b7be14fd3e264a300a01ddfa0456dc777f2bf3c811b5d7231`.
Both round-1 suggestions are incorporated and confirmed by the re-review.
PLAN.md is unchanged after this verdict. The loop concludes after two rounds;
this record does not represent another revision of the reviewed proposal.

Three new low-priority, explicitly nonblocking notes are accepted for the
implementation-review checklist and deferred, not marked fixed or closed:

| Finding | Deferred follow-up before implementation acceptance |
| --- | --- |
| PLAN-R2-001 | Define harness-side publication-copy/fallback observation. PR298 has aggregate pinned capacity but no built-in per-group byte/fallback counters. A read-only instrumentation wrapper or derived audit must be reviewed; label derived metrics as derived, distinguish sampled pinned capacity from a true peak, and do not invent runtime counters. The no-image-publication and no-timed-fallback requirements already remain mandatory in the reviewed plan. |
| PLAN-R2-002 | Carry the inherited native provenance caveat into the compute harness receipt/report: the accepted staging receipt records different psana/meson.build hashes between the PR298 source and the frozen native build. The six Python/Git checks do not validate native rebuild compatibility. No new native build occurred or is authorized; implementation review must establish whether the inherited compatibility evidence covers the new application path. |
| PLAN-R2-003 | Pin and record PS_GEO_SHARE as well as PS_JF_SHARE_DERIVED. Prefer an explicitly disabled geometry-cache path for this synthetic no-geometry workflow, subject to implementation verification, or fully inventory any enabled shared geometry/calibc artifacts. Account for their setup time and host residency outside the clean interval. |

These are reviewer-permitted deferrals at the **plan-only** gate, not waived
implementation or runtime requirements. No future harness, job, numerical
result, hardware fit or node availability is accepted by this review.

The user decision remains whether to use four BDs on four GPUs for the assumed
full 100-bin cube, or request a separate single-GPU memory-bounded design.
Keep the requested I/O controls and nominal batches fixed unless a separately
declared cohort is approved. New job-submission authorization is still needed.
