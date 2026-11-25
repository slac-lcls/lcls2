# Round 1 response

Reviewed implementation: 64acad1438955ee1170ec304f376cbb9aa22295b.
All findings accepted; corrections await independent re-review.

| ID | Response / evidence |
| --- | --- |
| NAT-001 | Replaced misleading eb_delivery_sizes with separate EB packet count, yielded-event windows (including zero), final finish-drain population and GPU callback sizes. DeliveryAccounting observes the source iterator and snapshots the final no-start-marker window exactly once; sum(window counts)+drain must equal delivered events. First-batch timing is explicitly a first nonempty yield-window-end proxy, including drain and GPU deferred retirement. Stub-sequence test covers packet0=0, packet1=2, drain=2, repeated drain observation and missing-population rejection. |
| NAT-002 | Manifest requires root/smalldata SMD layout. Worker checks both actual DataSource file lists against manifest paths in stream order. Fixtures use the native layout; tests reject order/path mismatch and flat-layout SMD. |
| NAT-003 | Scheduler cancellation has a six-second total deadline with shrinking subprocess timeouts; local launcher waits add at most one second, within the controller's ten-second SIGTERM grace. Failure explicitly records no_active_steps=false if scheduler cleanup is not confirmed. Documentation requires allocation audit after forced kill, blocked filesystem receipt write or unknown cleanup. Offline interrupt path, cleanup failure and deadline tests added. Native cancellation qualification remains pending. |
| NAT-004 | Measured process_seconds stops at srun wait return. Scheduler confirmation duration is recorded separately, with an offline clock test showing five-second process versus four-second accounting lag. |
| NAT-005 | Added explicit qualification of --exclusive/--exact/mask_cpu compatibility and --gpus-per-node=0. Keep fail-closed affinity/GPU checks. No launch viability claimed before native smoke. |
| NAT-006 | Added nonzero srun, interruption, cleanup failure, scheduler identity mismatch, warmup placement, diagnostic/clean mismatch, wrong GPU, nonzero residency and generated native-index tests. seal_native_index is the actual helper used by main and the receipt/changed-file test. |
| NAT-007 | Removed redundant overwritten provenance call and always-None post-loop manager lookup. Reader construction hook retains accounting references until the single final snapshot. |

Coordinator findings addressed in the same correction:

- Preserve Slurm's CUDA visibility on GPU ranks; check one step GPU and valid job
  assignment, then verify the actual CUDA UUID. CPU/service ranks hide devices.
- Require baseline and candidate to share Python/version, dependencies/native
  libraries, pinned environment and GPU runtime versions. Only reviewed psana
  installation/revision differences vary. Tests cover mismatched shared runtime.

No native job, bulk input copy or qualification flag change. The code remains
unqualified for performance execution until the documented native checks pass.
