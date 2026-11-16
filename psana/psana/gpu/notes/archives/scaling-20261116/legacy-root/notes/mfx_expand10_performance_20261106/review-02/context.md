# Independent re-review: performance preparation

Review commit 5f4e7167b05748d440bafa64027fae7f86279a4b in detached checkout
/tmp/mfx-performance-review-20261106-r2. Previous review commit:
5727e727fd18de513988acaba66251a681dda25c (pass with four nonblocking findings).
Read review-01/review.json and response.md for the prior verdict and responses.
The original scope and project guidance are in review-01/context.md and scope.md.
Do not re-run a loop, execute tests, edit, use MCP or delegate; restricted reads
only. Unavailable evidence is a limitation, never a pass.

Re-review all PERF-PREP-01..04 and regressions in the fixes, not just the response.
Only four Python files changed since the reviewed implementation: runner.py,
timing.py, perf_cube.py, test_preparation.py. An exact git diff of those files is
provided at build/preparation-evidence/code-changes.patch for efficient review.
No kernels, core psana, runtime, fixtures or calibration constants changed.

Changes:
- Count-only admission_counts(window) patches the frozen runtime's pressure-only
  `_trim_gpu_caches` path in both modes; counts are separated into warmup/steady.
  It uses no clocks/CUDA. In profile mode the existing timed observer nests
  around it, both managed by exception-safe context managers. README and timing
  contract distinguish count-only observation from diagnostic timer patches.
- Profile metadata hook bypasses its device span on a closed metadata context,
  allowing the original RuntimeError to propagate.
- Fingerprint the actual imported Stage2 fixture/verify and Stage3 xtc_fixture
  helper files in addition to the existing code/runtime fingerprints.
- prepare() rejects existing kernels or previously seen events before CUDA import.

Fresh validation AFTER these changes: Slurm job43548551, COMPLETED exit0 in50s,
sdfampere011 A100-SXM4-40GB, lcls:data@ampere normal QOS. Nine host tests passed
(job-43548551.err). These include actual sealed-framework-context error semantics,
phase-specific pressure counting with timer calls forbidden and restoration on
failure, plus once-only prepare guard without CUDA. Four controlled GPU cases
again passed exact per-bank sums and invalid-panel rejection. All three native
1k cases passed exact per-bank event counts, 100 warmup/900 steady, callback
sizes1,2,5,6,7 and158 calls: throughput1, throughput2, profile2.

New count evidence: throughput1 warmup0/steady0; throughput2 and profile2 both
warmup15/steady142. Profile CPU cache-trim call counts agree exactly. Thus
PERF-PREP-01's previous possible one-stream-flush inference was not confirmed;
the observer now distinguishes both cases. No attempt was made to tune the
inherited4GiB budget or claim stable throughput from these small runs.

Latest compact evidence: validation-summary.json, execution-source-hashes.json,
profile-summary.json, device-checks.json, artifacts.json, slurm-accounting.txt,
job-43548551.out/.err. Coordinator independently rehashed every reported Python
source in all three native reports against this revision, including the three
new helpers. All matched. Commit hooks passed. Full reports/intervals are
copied/hash-checked under build/preparation-evidence/ with a copy-index.json.
The original smoke remains in its prior commit and review-01 artifact index.

Limitations unchanged: frozen B/install psana, CPU-fallback KvikIO, serial only,
small event/count smoke and controlled numerical tests rather than new 10k
full-pixel GPU acceptance. No full10k campaign, Nsight capture, MPI or GDS claim.
Cube bin-index upload and null-stream waits remain in inclusive call spans;
interval sums are not loop wall time. 10k fresh manifest SHA/size preflight was
done previously; file hashes remain historical unless --verify-files.

Return structured verdict for exact current SHA, explicitly close addressed
finding IDs and identify any new blocking findings or limitations. You need not
repeat extensive unchanged-kernel analysis already documented in round1, but
check the updated observer/context ordering and failure paths independently.
