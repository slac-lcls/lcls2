# B versus stream-grouped batched locators

This harness is an isolated adaptation of the read-only historical warm A/B
harness in the sibling `psana2-gpu-d2h-pipeline` worktree. B uses its frozen
`validation/perf-acceptance-20260916/sources/b/install_psana` prefix. O uses
this worktree's `install_psana`. Neither historical install is modified.

```bash
source setup_env.sh
./build_psana.sh -j 8
sbatch validation/batched-locators-20260917/run.sbatch
```

`run.sbatch` runs the focused unit/device correctness suite first. The controller
then stages bounded unmodified data prefixes, verifies CPU-reference raw/calib
hashes, alternates three repeated clean and CPU/NVTX runs per variant, and takes
separate Nsight traces. At least 99% page-cache residency before and after each
measurement is required. KvikIO compatibility mode is ON (not GDS). Keep the
source/install and scripts fixed during the measurement.

The copied `phase_timing.py` supports B's per-handle loop and O's batched block
under the same `xtc.locate_all` scope. `test_phase_timing.py` verifies that
removing timing contexts restores the original AST for both revisions. The
original parser/gather scopes and no-added-synchronization policy are retained.

```bash
python validation/batched-locators-20260917/summarize.py \
  validation/batched-locators-20260917/job-JOBID-warm-ab/results.json
python validation/batched-locators-20260917/trace_summary.py \
  validation/batched-locators-20260917/job-JOBID-warm-ab/O-trace-cpu-nvtx-r0.sqlite
```

After both traces have exported, generate their summaries and run `audit.py` on
the job directory. The completed evidence for this implementation is in
`job-38504647-warm-ab`; the report is
`psana/psana/gpu/docs/performance/batched_locators_sdf.md`. The audit checks counts,
cache residency, timestamps, memory samples, and unchanged installed-source hashes;
it does not impose a hardware speed threshold.

The trusted local calibration snapshot and CPU reference are read from the
historical job `job-38419641-primary`. Provenance records its checksum, source
revisions, the uncommitted implementation diff, install paths and script hashes.
Each case logs actual imported module paths and runtime versions.
`build-provenance.json` additionally verifies both installed parser sources and
identical native compiler options. GPU samples,
cache diagnostics, traces and results remain in the new job directory.

Main-suite logs are `core.log` (initial B-era stdout assertion failure),
`baseline-test-failure.log` (same failure on frozen B), and `core-fixed.log`
(after applying only the test correction already present in `599f856ae`).
The longer MPI suite is recorded in `byhand.log`.
