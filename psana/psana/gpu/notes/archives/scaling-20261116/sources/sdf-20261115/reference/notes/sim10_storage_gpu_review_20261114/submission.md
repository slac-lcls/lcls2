# Reviewed diagnostic completed

**Job 45028778**, submitted once on **2026-11-14 11:39:43 PST**.
Historical Slurm readback at 11:39:50 PST was **PENDING**, reason None, start
Unknown. Final accounting confirms **COMPLETED**, exit **0:0**, on sdfampere036:
**11:40:08–12:54:08 PST**, elapsed **1h 14m**, on November 14, 2026.
The [saved results and next screens](RESULTS.md) retain compact measurements.

- Round 2 independent Claude review: **pass**, no unresolved blockers.
- Frozen attempt02 manifest: `a06e3c9571cdfebde916bf7af2b3f5d83eb36a6ba36909ba2bae79a9c1bdfc98`.
- Exact snapshot host validation: **56 CPU-only tests passed**, 1.40s;
  Python AST, shell syntax, source/native/KvikIO integrity checks passed.
- Requested node **sdfampere036**, exclusive, 4 A100s, 6 tasks ×6 CPUs,
  192G, 3h, account `lcls:data@ampere`, QOS normal. Readback agrees.
- No commits, new branches, rebuilds, node fallback or duplicate submission.

Working artifacts:
`/sdf/scratch/users/m/monarin/gpu-validation/sim10-storage-gpu-20261114/attempt02/`.
The submission journal holds the exact command, review digest, manifest digest
and preflight readback. stdout/stderr: `slurm-45028778.out` / `slurm-45028778.err`.
Review contexts and full CLI evidence: sibling `review/round01` and `review/round02`.
Compact review, manifest and submission receipts are also preserved here.

At submission, 036 was still IDLE/no allocated TRES, but test-only estimated
**2026-11-30 21:41:43**. This is unexplained: no covering reservation or displayed
own-association/normal-QOS limits accounted for it; no SBATCH environment
overrides were present. That forecast is not the actual job's scheduled start,
which turned out to be **November 14 at 11:40:08 PST**, 25 seconds after
submission. The forecast did not reflect the actual start; its cause remains
unexplained. Final status comes from `sacct`, not the stale forecast or queue.

Completed allocation gates: live NVMe/free-space/topology qualification,
31 GPU tests/device, six traced pilots, 18 paired full cases, six Weka anchors,
physical NIC/disk checks, two CPU slab probes, final audit and owned-stage
cleanup. Both final audit and campaign status say pass, and Slurm exit is 0:0.
Independent result review remains pending. SG-007/008 reporting nits remain
explicitly deferred; see `disposition-round02.md`. The completion claim does
not rely on complete.json alone. No post-review modifications to submitted
source and no duplicate submission.
