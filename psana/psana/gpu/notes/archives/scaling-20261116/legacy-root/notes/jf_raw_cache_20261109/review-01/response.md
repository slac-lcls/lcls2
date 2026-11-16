Readiness review passed at 585111a4518a068fbb726dda0d4770c2839788c6. This is not final GPU/performance acceptance. No measured code is changed during job43983799.

| ID | Response |
|---|---|
| JFRC-01 | Accepted limitation. Python finally does not guarantee cleanup on SIGTERM/SIGKILL. Monitor this job through completion and verify cleanup; if killed, verify/remove only its exact private stage after all job processes terminate. Add signal-aware cleanup before any rerun; do not claim a kill-proof launcher. Final task closure requires a cleanup record. |
| JFRC-02 | Accepted as optional rerun improvement. Keep the submitted 600-second case limit unchanged; native reports retain conditioning.seconds, now also included in the independent summary. Failed/timeout cases cannot pass. |
| JFRC-03 | Accepted housekeeping gap: current cleanup.json covers only the 313GiB xtc copy, not the small parent/cache directories. Verify normal Slurm epilog cleanup or remove the exact private root after completion; record the distinction. |
| JFRC-04 | Exact sbatch command recorded in submission.txt, including log paths and positional arguments. |
| JFRC-05 | Fixed in post-run auditor only: require conditioning.kind to equal the case setting; add a negative test; report full cache_after and conditioning duration. This does not change running benchmark source hashes or timings. |
| JFRC-06 | Document that RunParallel still performs its empty shared-calibration setup; skip_calib_load=all and empty constants preserve the historical raw-only workload. |
| JFRC-07 | placement.json records 56 distinct physical cores available on sdfampere038. All psana ranks use this common mask; standalone uses the first four/eight. No claim that every future allocation has this mask. |

Final results and the changed auditor still require independent review on the final commit. No finding is treated as evidence of GPU acceptance.

R2/R3 updates before rerunning: JFRC-01/03 signal-aware launcher cleanup is
implemented and normal/TERM regression tests pass. R2 root-cleanup.json records
full removal after its failed warm probe. JFRC-02 per-case limit is1200s; the R3
allocation is bounded at90min because full interleaved warming is untimed but
substantial. JFRC-05 strict kind/per-file gates and partial-result classification
have negative tests. Final code/results still need re-review.
