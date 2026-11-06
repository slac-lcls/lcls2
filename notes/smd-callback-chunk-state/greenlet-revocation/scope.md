# Remove greenlet and defer delivery during no-yield intervals

User request: revoke greenlet and create a separately reviewed issue with a
standard category label plus psana2-cpu as both label and title prefix. Continue
the implementation/review loop for code and issue creation. Existing permission
to send code/context to the configured Claude service and push the working
branch remains applicable.

Base: 570f954d96d23012fd14341405471582b164a728 on
tests/smd-callback-chunk-state. Narrow interpretation: retain the original
counter-continuity fix and remove the greenlet execution model and dependency.
An optional clarification offered full step-3 rollback as an alternative; the
default preserves the user's original bug-fix objective.

Replace stack suspension with ordinary synchronous generator iteration.
Serialize pending transitions and retain output packet size/destination rules,
but explicitly defer bounded pending memory and prompt transition delivery
while the callback does not yield. Preserve serial/MPI counter continuity,
physical-step resets, metadata, routing, early return and cleanup. Do not claim
that bounded emitted batches bound the pending transition queue.

Tests must run without greenlet. Keep the 24 production regressions passing;
keep finite reject-all, routing and lifecycle coverage, and replace greenlet
backpressure assertions with accurate synchronous-delivery characterization.
Run both required psana core groups on the final implementation SHA.

Review contract: Codex implements and drafts the exact issue payload; Claude
independently reviews both code and issue text/labels in a detached checkout.
Maximum three rounds, 15 minutes per invocation. No publication until both
deliverables have no blocking findings. Publish the exact reviewed issue body,
verify title/body/labels by readback, and push reviewed branch commits without
rewriting history. Historical step-3 review records remain as history, with
current documentation clearly marking the greenlet design as superseded.

Issue target: slac-lcls/lcls2. Standard category label: bug. User-requested
psana2-cpu label is absent from the repository and will be created only after
review. Proposed metadata is in issue.json; exact body is issue.md. Searches
found no existing issue matching the callback no-yield buffering case.
