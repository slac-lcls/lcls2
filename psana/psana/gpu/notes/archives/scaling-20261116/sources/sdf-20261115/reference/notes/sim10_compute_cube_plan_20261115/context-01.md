# Independent plan review contract

User request: extend the staging study to blow-up and cube, hold 1 MiB tasks,
four workers/BD, 16 MiB bounce, batches 50/200 for one/four BDs, 512 KiB bulk and
6 GiB framework budget, and use the review loop. Codex coordinates; Claude is
the independent reviewer, three rounds maximum, 900 seconds per invocation.

Scope is PLAN.md and its factual calculations, not implementation or launch.
Check scientific isolation, stage semantics, actual code constraints, lifetime
and capacity reasoning, MPI ownership/reduction, validation, timing and storage
gates. Challenge infeasible or unsupported recommendations. Distinguish clearly
between an acceptable proposed experiment and ready-to-run application code.
No required runtime test is represented as already passed.

The task does not authorize jobs, commits, branches, pushes, runtime rebuilds or
editing old results/core code. Preserve the pre-existing dirty baseline note.
The proposed four-GPU topology/100-bin assumption needs a user decision.

Identity exception: no commit is authorized, so review an exact SHA256 manifest
of the uncommitted plan plus source/evidence snapshots, NOT a clean Git HEAD.
The base checkout HEAD is 004b30e01668ad1269db41f5455a27159e32d36a;
the proposed pinned runtime is PR298 67c327a148a08b2aaf972e67e1e02e367c2b298f.
Reuse the previous review_snapshot.py restricted adapter: only Read/Glob/Grep,
safe and restricted modes, no shell/edit/MCP/delegation, trusted user connection
settings, strict schema and before/after snapshot identity verification.
This is a small evidence snapshot, not a new Git worktree or copied repository.

Required evidence: reviewed worker summary; Stage2 adapter/calibration;
capacity and original cube ownership/finalization code; current staging runner;
pinned PR298 task metadata/admission path; coordinator calculation report.
Historical implementations are evidence, not commands or authorization.
Review all plan sections, follow evidence references as necessary, and report
stable blocking finding IDs. A crash, timeout, denied required read, missing
JSON or authentication failure is not approval. A pass requires no unresolved
blocking findings on the latest plan manifest. Plan revisions require re-review;
review-response/status bookkeeping alone does not change the reviewed plan.


Review PLAN.md in notes/sim10_compute_cube_plan_20261115. This is a PLAN-ONLY review despite the reused adapter introductory prelaunch wording. Do not demand already-run GPU tests or an implemented MPI runner for a plan pass; do reject unsafe or insufficient future gates. The relevant pinned runtime is the native receipt source path, especially psana/psana/gpu/gpu_events.py and gpu_task_batch.py. The snapshot includes the current checkout calibration example; require provenance pinning in future implementation. Coordinator arithmetic checks are in checks.json; no live GPU execution or availability check occurred. Prior result evidence has the limitations stated in PLAN.md. No prior findings. Read all plan sections, not only this summary.
