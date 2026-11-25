# PR331 + PR326 + EndRun fix: local verification

The exact repeated-run callback MPI test passes with both fixes combined. Final code commit: `dc59d0b94bce73caa134b1c67b40781f6b5e5c65`. PR331 base `46c06415a3131d6a9b20247e373a49f85b164370`; PR326 head `98f344a32c66cd68c8c16af55779a879cdb55f38`; EndRun repair `fa695530a92b322ca96efb157ac506be4d04bfe1`. Local branch `fix/pr331-pr326-endrun-20261124`.

Final Interactive/lcls_g job **61947791** passes 1,438 candidate checks: 6 CPU lifetime, 17 callback (including the exact CI case), 1,398 GPU unit, 15 real-device component, and 2 PR326 multi-run checks. One unfixed-runtime negative control fails as expected on the new consumed-termination assertion. The existing default pytest filter deselects one slow 2,000-seed property sweep; selected checks have no skips. Both PR326 runs deliver exactly 10 + 10 events, at one EB/four ranks and two EBs/five ranks.

Initial job 61947711 passed the MPI/callback/device checks but its GPU unit fake communicator lacked Recv. The final test-only correction adds Recv and checks termination consumption. All final checks were rerun after that commit; original failures are preserved. Both allocations are terminal. Verification uses fresh synthetic fixtures and a separate installed prefix. Only mpi_ds.py and node.py differ from the verified PR331 runtime; all 37 native extensions are unchanged.

This is Perlmutter Python 3.11/Cray MPICH verification using an mpirun-to-srun adapter, not a fresh GitHub CI run. Device component coverage is not full MPI/GPU multi-run coverage. Existing broader performance decisions are unchanged. No remote branch has been pushed or merged. Claude round 2 **passed with zero findings** on evidence snapshot `c7f91b1457fb4bf5a40a326dc0b3554e2f4d5f89`, closing both prior findings. See review-round02.json and review-response.md.

See verification.json, response-to-round01.md, and evidence/. Prior review and diagnosed harness corrections are in prior-review/. SHA256SUMS covers the evidence snapshot. This commit adds verification records only; runtime/test sources are identical to the tested code commit.
