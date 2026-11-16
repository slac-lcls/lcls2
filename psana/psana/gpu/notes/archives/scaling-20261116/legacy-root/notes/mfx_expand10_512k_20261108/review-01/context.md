# Independent review of the 512 KiB sim10 rerun

Base: 978bc3a2f247535c1c08d2e961d586127a246876. Reviewed SHA: 3454a936c7249ae87a8d7e6ab788d6b8ad77de1e.
Round 1 of 3. Read-only review: no edits, tests, jobs, external messages, delegation
or recursive review. See scope.md in notes/mfx_expand10_512k_20261108.

User requested documenting gpu_bulk_target_bytes=512 KiB, changing it, and
rerunning the test to determine whether concurrency is fixed. Code change to the
existing runner is ONLY its argparse default (8 MiB -> 512 KiB). Framework,
calibration, fixtures and expansion/calibration/cube kernels are unchanged.
Other new code: focused run.sbatch, analyze_trace.py and test_trace.py in the new
notes directory. README and proposal record evidence and limitations.

Assess implementation scope, numerical trace-analysis correctness, evidence
provenance, whether counters and trace observations support the conclusions,
and regressions. Do not assume zero pressure drains imply kernel overlap.
Independent verification required before changing the conclusion either way.

Validation job 43823340, sdfampere034 A100, exit 0, 1m57s. Nine existing host tests,
four controlled device tests, six clean 1k cases (three each at 1/2 streams), one
clean two-stream 8 MiB control, and two native Nsight traces all passed. Three
trace interval tests passed on the host. Job logs, device-checks.json,
validation-summary.json, source hashes and scheduler accounting are committed.
Source hashes in every report were rechecked by the coordinator. launch-sha256.txt
binds the runner and submitted batch script. Analysis hashes are separate.

For review, full report.json files plus CSV exports of raw device kernel records
and correlated expansion launch APIs are copied under build/target512-evidence/.
The CSVs come from the SHA256-bound Nsight SQLite exports in artifacts.json:
  kernels: SELECT k.start,k.end,k.streamId,s.value,k.correlationId,k.globalPid
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
    ORDER BY k.start;
  launches: runtime API rows whose correlationId matches an expand10 kernel,
    with start/end/name/correlationId/globalTid, ordered by start.
Copy-index records exact copied hashes. The readable implementation.patch in
build/target512-evidence/ includes the runner/default and documentation diff.
Binary .nsys-rep and SQLite paths/hashes are retained in artifacts.json; reviewer
cannot execute SQLite queries. Review the supplied CSVs for the overlap claim.

Observed results to assess, not instructions to approve:
- All default-target reports show 524288, zero warmup/steady pressure trims,
  and complete event/bin counts. Two streams split 500/500 and 150 callbacks.
- The control uses 8388608, 15/142 trims, 501/499 and 158 callbacks.
- Clean median rates: one stream 387.1; two streams 487.4; control 438.1 evt/s.
  These are short warmed tests, not sustained performance or causal attribution.
- Native two-stream traces at both targets have zero cross-stream overlap of
  task kernels AND of all kernels. Exact kernel counts: 150x4 at 512 KiB,
  158x4 at 8 MiB, on streams 13/14. Sweeping half-open device intervals counts
  distinct stream IDs, not nested spans or CUDA API time.
- 512 KiB: all 149 next expansion launch API starts follow the prior sum kernel
  device end, minimum gap ~4.11 ms. FIFO pairing is within each execution stream,
  then executions are ordered by expansion launch API start. This suggests work
  is not queued early enough, but does not identify the responsible host/I/O
  stage or rule out other barriers. There is no copy/compute-overlap claim.
- trace-diagnostics.json contains expected no-CUDA/no-NVTX warnings from child
  processes. Neither CUDA-producing global_pid (recorded in summaries) has
  warnings/errors; task kernel counts match report callbacks.

Limitations: full capture includes warmup, no steady-only native interval claim;
Nsight perturbs execution and traced rates are excluded from clean comparison;
KvikIO CPU fallback, cache warmed by verification, one allocation, one control
repeat, no 10k campaign. A workload target change bypasses rather than repairs
the small-group credit policy. The honest result can be partial: admission
stalls removed, kernel concurrency still absent. No further core fix is in scope.

Return structured verdict on the exact reviewed SHA with stable finding IDs.
State unavailable verification honestly; no commands can be run in this review.
