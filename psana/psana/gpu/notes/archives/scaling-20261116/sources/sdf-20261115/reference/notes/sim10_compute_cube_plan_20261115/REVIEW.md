# Sim10 expansion and cube plan review

**Outcome: independent plan review passed in two rounds.** The latest reviewed
proposal is [PLAN.md](PLAN.md). This is not an implementation or launch pass.
No GPU job, rebuild, commit, branch, or change to the frozen benchmark code
was made. The pre-existing dirty baseline note remains unchanged.

| Round | Manifest SHA256 | Verdict | Follow-up |
| --- | --- | --- | --- |
| 1 | b8dd3c561f2a3cc33deacd46f004fdd664848a7ca29f53223d06c4692080edfd | Pass, no blockers | Two suggestions incorporated and re-reviewed |
| 2 | c7e434de713ad13b7be14fd3e264a300a01ddfa0456dc777f2bf3c811b5d7231 | Pass, no blockers | Three low-priority implementation/provenance notes explicitly deferred |

Read [round 1 findings](review-01.json), [response 1](response-01.md),
[round 2 findings](review-02.json), and [closure and implementation checklist](response-02.md).
The review uses an uncommitted SHA256 snapshot, not an invented clean commit.
Code/evidence snapshots and restricted Claude logs are on shared scratch:
`/sdf/scratch/users/m/monarin/gpu-validation/sim10-compute-cube-plan-20261115/`.
Durable manifests preserve the reviewed file identities.

Coordinator checks: [arithmetic](checks.json) reproduces worker rates, image and
cube sizes, all four memory subtotals, input bandwidth and case counts;
[runtime provenance](runtime-check.json) verifies six relevant Python files
against the existing receipt plus the pinned source Git HEAD. Plan links,
snapshot identity and whitespace checks pass. None is a live GPU test or a
fresh validation of the frozen native build. Claude read code/evidence with
Read/Glob/Grep only and did not execute tests or verify hardware.

Recommended decision: retain the user's I/O controls and compare S/E/EC/ECC
with fresh matched controls, initially at one BD/one GPU and four BDs/four GPUs.
The 100-bin cube and direct batch200 intermediates do not fit the conservative
shared-40GB-GPU memory plan. The four-GPU case is a candidate requiring full
memory/lifetime qualification, not a proven fit. Confirm this topology/100-bin
assumption before implementing the new MPI compute harness.
