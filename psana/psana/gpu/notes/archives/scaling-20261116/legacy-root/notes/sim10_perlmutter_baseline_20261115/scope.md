# Sim10 Perlmutter baseline: GPU cube checks

User revision: run Expand → Calibrate → Cube without writing cube data files.
Check dimensions and reject garbage (zero populated bins, NaN, infinity).
Inspect each BD's resident FP32 (100,32,512,1024) GPU bank after drain; check
all pixels finite, every populated bin has a nonzero pixel, empty bins all zero,
and exact per-bin counts against delivered global event ordinals. Individual
zero pixels are valid. Only counts and scalar flags return to host. No full-bin
D2H, MPI cube reduction, cube export, CPU cube reference or reference-cost probe.
This is basic shape/content validation, not numerical-reference correctness.
The four BD banks remain distributed, with no globally merged cube output.

Reuse PR298 67c327a148a08b2aaf972e67e1e02e367c2b298f and job61359629 qualification;
no rebuild. Keep count2/1MiB layout, frozen constants, four workers/BD and fixed
1BD/1GPU/batch50 and 4BD/4GPU/batch200 configurations. Preserve lifetime, drain,
error, event/byte/range, actual-open identity, CPU/GPU affinity, and >=4GiB
physical memory gates. Small JSON receipts and diagnostic logs are retained.

Prepare bootstrap02r2: 70-minute single-node allocation, two ECC checks (201/1003
events), nine 100k staging cells (three repeats each of the two target profiles
and staging-only 4BD/1GPU bridge), existing native tests/descriptor guards.
Full same-node S/E/EC/ECC comparison remains a separately sized proposal after
bootstrap results; repeat staging on that node and apply this same cube policy.

New implementation-review cycle, at most three independent Claude rounds on
exact commits in detached worktrees. Continuing external review authorization:
AUTHORIZATION.md. Preserve historical bootstrap01 and job61373306 evidence;
its first ECC step timed out, with the exact blocked operation still unproven.
No new compute approval exists: show concrete reviewed plan and command before
submitting bootstrap02r2. No automatic retries or promotion to the full campaign.
