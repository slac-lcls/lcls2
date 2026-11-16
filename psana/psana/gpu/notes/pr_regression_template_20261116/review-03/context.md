# PR regression template: final review, round 3 of 3

Reviewed SHA: bf0f93a9dcc63ac4920f28df34679ac27722048d
Original base: 39ea002a8ffc95546405139e50f2f45005514ee2
Prior reviewed code: a71922bfaf930c2c8633753672a8170274b3d5a6 (round 2 PASS).
Concurrent upstream: 1017d06e23b22b114e08980b3525faec43dcc0b9 (merged PR319).

User requests a maintained PR performance regression run template, run51 status
check against Confluence, candidate selection if unaffected, and agent instructions
with review loop. Preserve frozen historical harnesses. Scope is a reusable
controller/profile plus native adapter contract; native adapters must be written,
reviewed and qualified for each revision/site pair. No native benchmark, hardware
result, durable dataset guarantee or end-to-end runnable worker is claimed.

Read notes/pr_regression_template_20261116/scope.md, validation.md, integration.json,
review-01/ and review-02/ responses/findings, and the current controller/tests/profile
under psana/psana/gpu/scripts/performance/regression/. All notes paths above are
under psana/psana/gpu/. Read docs/performance/templates/pr-regression.md and the
merged docs/performance/README.md navigation. Candidate audit and selection
records remain under the same note directory. Review latest code for regressions.

Round 2 passed and accepted PRR-01..04 as fixed. It added low/nonblocking PRR-05:
post-acceptance timing edits could evade summary checks. This change writes an
accepted.json only after native result validation with result SHA256 and evidence
paths/digests, and requires it in summarize. A test changes loop_seconds after
acceptance and requires rejection. Existing within-case placement-edit test now
hits this earlier hash gate. Docs require immutable retained archives and explicitly
disclaim protection against coordinated rewriting of all receipts.

While round 2 ran, upstream PR319 published completed Sim10 S/E/EC/ECC study.
Merged cleanly at deb15848ac8bf8e62bb808104a4fefbf4f981353, no conflict edits.
Coordinator compared Git tree objects: template/controller/audit unchanged by
merge, and upstream Sim10 archive/catalog/analysis unchanged from 1017d06e2.
integration.json records exact identities. README combined both additions.
Upstream study has its own passing review at notes/sim10_results_recording_20261116/;
this task does not claim a new audit of all its native results or modify them.

Checks: 20/20 controller tests passed after PRR-05 fix (7.145 sec), including real
child PID termination, frozen/accepted evidence changes, phase/allocation/placement
gates, false qualification flags and missing receipts, full observed contracts,
balanced scheduling and statistical outcomes. Synthetic adapters only, no native
MPI/GPU jobs. Combined catalog checker: pass 10 campaigns, 2082 hashes, 275
arithmetic cases. Diff check passes. No more edits to runtime/examples/package
configuration/historical harnesses. No new status-payload download.

Dataset: Confluence warning belongs to mfx100852324 runs77-79. Direct run51 audit
selected status doc69d00aad4b71e8288a8c6b0e, uint64(3,32,512,1024), all3gain OR
and allbits reject gives 16768391 good/8825 bad=99.9474% good. Pixel_status only,
not full calibration or permanence. Audit source preserved as executed text,
copy-to-external-scratch instructions explicit; provenance is post-audit readback.

Read-only review; do not delegate or invoke another review loop. Coordinator
checks are evidence, not checks you performed. Return exact SHA, verdict and
stable findings, distinguishing blocking issues and optional suggestions. Prior
reviews are not approval of unreviewed changes; evaluate this final change and
merged navigation independently. Closure status/review receipts will be recorded
in an evidence-only commit after your result.
