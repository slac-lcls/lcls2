# PR298 fixed-profile stripe screen

Resume the user-authorized stripe sweep after accepted qualification61359629.
Run18cases: stripe counts1,2,4,8,16,32 in three randomized complete blocks,
100000events each,4BDs sharing oneGPU, one CUDA stream/BD,4workers/BD,batch200.
Preserve PR29867c327a, dataset identity, local-zero-residency gates, all31native
tests, all30file metadata/range validation, exact event/byte/handle/runtime
checks, disjoint34physical cores,6GiB/BD budget, sampled4GiB-free device floor.
No payload copies, framework rebuild, cube workload, multinode work, or changes
to shared filesystem settings. CPU-fallback staging; no GDS/storage ceiling claim.

Close EXEC-R1-02 through06 before launch. Include EXEC-R2-01 audit SHA binding.
Use implement-review-loop, up to3rounds, independent Claude code/plan review
before submission and results review afterward if rounds permit. Existing
external-review authorization applies to scoped code/results. No auto-retry.

Use lcls_g/premium, one exclusive GPU node. All five prior jobs cost3747node
seconds against the21600second phase cap, leaving17853seconds. Include any new
allocation in the ledger whether it passes or fails. Recheck Iris and squeue
before submitting. Generate/recompute the launch plan from the ledger and
sealed qualification receipts; Slurm limit must match. Submission is authorized
by the user's request to resume this stripe sweep, contingent on passing review.

Acceptance: meaningful harness checks, reviewed exact immutable runtime bundle,
all18cases passing independent audit and Slurm0:0, memory/cache/affinity/runtime
receipts, full three-block results with medians and paired-to-count2 comparisons.
Preserve partial results on failure, label them incomplete, never rank a partial
block as the completed sweep. No automatic promotion to further benchmarks.
