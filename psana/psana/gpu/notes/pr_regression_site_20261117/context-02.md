Round 2 review commit b5ddc20f71de447bb8019131d68159e7bf04a6d3, base 15459eb09d2d99bcb8ea87d677b528fc8c9c038b.
Read previous context/review/response in psana/psana/gpu/notes/pr_regression_site_20261117.
Verify SITE-001..005 fixes and check regressions in the new code/docs/tests.
Sources in psana/psana/gpu/scripts/performance/regression: prepare_site.py, site_probe.py, test_prepare_site.py, SITE.md. Existing adapter/worker/runtime helpers unchanged.
Hardware pending. Bounded capture-01.sbatch now has22min limit,90sec per step, no detector data.
Review code readiness for that placement validation; actual flags/CPU/GPU/NIC behavior remain unproven until run.
Validation-02.json is honest: first expanded run had fixture shadowing error fixed; latest83-test suite passed82, failed an existing Lustre mincore cache test (0vs57resident bytes). Targetedrerun of all12site tests plus that cache test passed13. Focused-test output is included. Treat cache flakiness as limitation; no cache code changed or hardware qualification claimed.
Response-01 documents all5findings. Draft now has unaccepted schema; final links index(path,sha), load_contract does not independently validate new link; campaign evidence mustpinboth.
Before placement requires, rank observation records affinity, raw physical topology, /proc allowed mask, schedulerCPU/GPUvalues. GPU identity/error saved beforeexpectedUUIDcheck. No flags changed.
No nested review or commands; public source and non-sensitive metadata only. Final hardware evidence will be separately reviewed in round3 if this code is ready. Maximum3rounds.
