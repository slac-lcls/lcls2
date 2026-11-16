# Round1 plan response

Review passed a9bc787d95a3fcc0d323d671174a15b78e5d8834 with no blockers.
All required reads succeeded; permission_denials was empty. All seven
nonblocking clarifications are accepted for round2 review:

- PLAN-R1-01: require all3complete blocks before ranking/finalist selection;
  incomplete campaigns remain descriptive. Full1BD/depth2/count2 sizing case
  counts against6hours and is excluded from rankings.
- PLAN-R1-02: confirmation target frozen now at4BDs/oneGPU,2streams/BD.
- PLAN-R1-03: explicit sequential RNG shuffles of canonical24cases, no rotation;
  same node/GPU/job for all3blocks, no silent pooling after interruption.
- PLAN-R1-04: require exact1/4device peers,6/24GiB aggregate explicit admission
  and measured overhead/headroom.
- PLAN-R1-05: actual descriptor fstat for KvikIO and DgramManager; all6counts
  get an untimed100event binding preflight; path-only checks are insufficient.
- PLAN-R1-06: exactly one immutable metadata view percount,reused unchanged.
- PLAN-R1-07: list six native integration files,save static pinned-PR inventory;
  actual collected node IDs/counts/hash manifest required at implementation review.

No runtime/build/data/cache mutation or Slurm job. Coordinator verified all30
existing payload identities,the72case arithmetic,and source evidence bytes
against the fetched PR SHA. git diff --check passes.
