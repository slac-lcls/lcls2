Review commit 15459eb09d2d99bcb8ea87d677b528fc8c9c038b, base 6e5d5f9b636b95be2cb79d699ea4f6bfd59a2bd2.
User requested next reusable regression driver and independent Claude review loop.
Scope in psana/psana/gpu/notes/pr_regression_site_20261117/scope.md.
New files under psana/psana/gpu/scripts/performance/regression:
prepare_site.py, site_probe.py, test_prepare_site.py, SITE.md.
Also read unchanged perlmutter_adapter.py (launch reused), native_worker.py identity/core helpers,
native_contract.py runtime/environment, prepare_runtime.py CPU guard and publication.
Only README/NATIVE links changed in existing files. No workload/profile changes.
81 offline tests passed; validation-01.json pins log. Hardware NOT yet run; code review
precedes bounded validation (single premium node, four tiny MPI placement probes,
no run51 read/cache work). Identify blockers to safe meaningful capture and usability.
No data/raw detector content in this public-repository code and non-sensitive task metadata.
Review must not claim offline mocks qualify real launch flags/Slurm placement/NIC selection.
Each future allocation requires new manifest. Current task excludes timeout/signal fault
injection, all-six-workload smoke, overhead/bounds and timed A/A/A/B; flags stay false.
User authorizes scoped validation after code review, commits/push. Three rounds max.
No nested review; file-reading only. Stable findings SITE-001 etc with blocking designation.
