# Round2 response and final audit preparation

Claude PASS on199b5091c0482d015609ed728efa1aaaab0240ab, no submission blockers.
Attempt2 job61480775 uses exact reviewed bundle and new destination; frozen
inventory and blobs verified before submission. No preparation code changes.

PREP-005: audit explicitly refuses sys.flags.optimize before any acceptance gate,
so -O/PYTHONOPTIMIZE cannot remove checks and produce a pass.
PREP-006: audit checks account/QoS/time limit and actual cpu/memory/GPU allocation;
also exact attempt02 destination, started receipt/job/node/plan association,
read-only frozen files/directory and SHA256SUMS against reviewed blob hashes.
PREP-001/003: final audit verifies code/provenance and both SMD/bigdata layouts.
PREP-004: final evidence must contain successful scheduler state and elapsed;
stream logs lack timestamps, so do not invent per-stream completion times.
PREP-002: non-L1 gap timestamp/prefix checks remain outside claims.

Final audit changes and actual outcome will receive round3 review. No accepted
inputs or performance qualification claimed before completion.

Observed site allocation corrects PREP-006's suggested exact AllocTRES check:
ReqTRES=cpu8/gpu1/mem32G/node1, but AllocTRES=cpu128/gpu4/mem32G/node1 for both
jobs under premium. The audit now checks both separately and records actual
full-node resources. The 8CPU/1GPU descriptions elsewhere are requests, not
actual allocation; no GPU calculation is performed. Third review must confirm
this evidence-based adjustment rather than silently accepting a mismatch.
