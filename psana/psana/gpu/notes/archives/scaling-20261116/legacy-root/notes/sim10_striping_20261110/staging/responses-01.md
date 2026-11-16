# Staging round1 responses

S1 fixed: realpath containment for attempt/bundle/output, batch cwd equality,
noclobber preflight outputs, mkdir job-started reuse guard, refuse existing
datasets. Slurm chdir/stdout/stderr explicitly pinned to campaign attempt01.
S2 fixed: fadvise DONTNEED after fsync before close; receipt explicitly says
advisory eviction, cache residency and remote bytes unverified. The independent
read means a separately opened full-file hash, not verified remote traffic.
S3 addressed: SIGTERM handler publishes failed state immediately then raises;
Slurm sends step TERM120s before walltime. Keep2h bounded walltime; estimate
30-90min after start is provisional, first verified set refines it. If killed,
absence of complete.json forbids acceptance; Slurm accounting remains required.
S4 fixed: six new main-path tests cover all30 copies/6sets success, injected
second-set verification failure, reuse, bad manifest, outside-campaign output,
and a mismatched pinned manifest. Failure retains first verified set and partial
second set with failed status and no complete marker.
S5 fixed: seventh added test parses the five captured native source layouts.
Coordinator also parsed current native lfs output and checked all source sizes.
S6 fixed: stage.py pins SHA256 of accepted transfer-manifest.json. Source
snapshot bundle hashes will be compared against this reviewed commit before
submission and stored with provenance. Quota ample; recheck before submission.

17 tests pass in0.083s; shell syntax/diff checks pass. Nonfatal MUNGE environment
warning occurs after test OK, unrelated to fixture tests; no native job yet.
