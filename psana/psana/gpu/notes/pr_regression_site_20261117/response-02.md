# Response to round 2

PASS at b5ddc20f71de447bb8019131d68159e7bf04a6d3; SITE-001..005 closed by
independent reviewer. Proceed with the reviewed bounded placement validation.

- SITE-006 optional extra walltime: defer for this frozen validation. The job
  is deliberately bounded at 22 minutes; final coordinator accounting is required
  even if its own trap fails. Future jobs must budget the documented setup margin.
- SITE-007 optional GPU-error offline test: defer; success-path hardware evidence
  is the current next check. Import failure remains in native.log, not the GPU
  identity receipt. No claim that GPU error-path evidence is exhaustively tested.

No production changes after reviewed commit. Keep cache-residency test flakiness
as a documented limit. Neither normal placement probes nor code review qualify
cache behavior or timeout/signal cancellation.
