Round2 re-review at exact commit 1af57a40073306763f013dd0876b57554fc53b2d.
Original base5a76c182f7d56a86d4dba43f5baf749400366bcf; first review64acad1438955ee1170ec304f376cbb9aa22295b.
Do not delegate, execute, modify, or start another loop. Same scoped implementation
acceptance and project guidance as the preserved context-01.md below; native
hardware qualification is explicitly outside this code step, not claimed.

Read psana/psana/gpu/notes/pr_regression_native_adapter_20261116/{scope.md,
context-01.md,review-01.json,response-01.md,validation-02.json,status.json}.
Recheck NAT-001..007 against code, and inspect for regressions/new issues.
Do not close disputed findings without evidence (none disputed).

Changes in scripts/performance/regression:
- DeliveryAccounting records separate EB source packet counts, yielded delivery
  windows including zero and final finish-drain population; requires population
  coverage. GPU deferred-retirement first-window timing is documented explicitly.
- SMD files must be root/smalldata and actual DataSource lists match both bigdata
  and SMD paths in stream order. Native-layout fixture plus rejection tests.
- Six-second total scheduler cleanup budget plus <=1 sec local waits fits the
  controller's 10 sec grace. Unknown cleanup stays failed and requires audit.
- process_seconds stops at srun wait, accounting confirmation time separate.
- Exact Slurm flag viability remains native qualification, now explicitly listed.
- Added requested failure-path tests and actual seal_native_index helper coverage.
- Removed overwritten provenance and stale manager inspection.
- Coordinator also fixed Slurm GPU visibility preservation/assignment checks and
  requires shared baseline/candidate Python/dependency/environment controls.

Validation independently executed by coordinator:24 adapter tests and20 existing
controller tests pass; catalog10campaigns/2082hashes/275arithmetic pass; diffcheck
pass. Exact git diff --exit-code from original base confirms profile.json and
PR187/208, PR299 and notes/archives inputs unchanged. These are offline checks,
not native API smoke, actual scheduler cancellation, hardware timing, overhead,
input qualification or component bounds. No job submitted. Default flags false.
No retained source/evidence edited. Please distinguish code acceptance from all
remaining qualifications listed in NATIVE.md. File-reading tools only; report
unavailable evidence rather than claiming independent test execution.
