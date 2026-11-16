# PR performance regression template and agent instructions

Use the maintained [controller and candidate profile](../../../scripts/performance/regression/README.md)
to prepare a new comparison. Its implementation has CPU contract tests; native
psana execution requires a campaign-specific adapter and qualification. It is
not a new result or a portable replacement for historical execution scripts.

## What every campaign provides

| Component | Required output |
| --- | --- |
| Question | PR, exact baseline/candidate commits, intended change, supported workload, predeclared regression/equivalence margin |
| Frozen inputs | Dataset manifest, exact event/timestamp/step selection, stream/panel mapping, file or used-range hashes, constants and independent numerical-reference hashes |
| Comparable settings | One shared resolved profile; only declared runtime revision/install differs between the two sides |
| Placement | Site/allocation/node, CPU/GPU model, UUID/PCI IDs, MPI roles, BDs/GPU, physical CPU affinity, environment and actual runtime/import/build provenance |
| Workloads | Raw GPU input and CPU events/steps controls; separate calibrated CPU/GPU profiles after constants/reference qualification; application profiles remain separate |
| Correctness | Expected event identity/count, bytes, shapes, tail handling, nonzero/finite calibrated witnesses and numerical comparison; explicit step/EndRun coverage limits |
| Execution | Fresh processes, separate warmup followed by cache conditioning, balanced adjacent AB/BA pairs, bounded timeouts, retained failures and terminal Slurm accounting |
| Observations | Cache residency, requested/read bytes, physical NIC deltas, host/GPU/pinned memory, settings observed rather than merely requested |
| Timings | Setup, first completed batch, drained event loop, process wall time; clean throughput separate from diagnostics/profiling |
| Analysis | Complete matched pairs per cell, all repetitions, mean paired percent changes and interval, equivalence/regression/improvement/inconclusive outcomes |
| Archive | Exact controller/adapter/source bundles, profile, commands, logs, raw receipt index/hashes, report, independent implementation and results reviews |

## Dataset candidate: mfx100848724, run 51

The [November 16 audit](../../../notes/pr_regression_template_20261116/README.md)
selects run 51 as a candidate: the Confluence all-g1/g2-status warning concerns
mfx100852324 runs 77–79. Run 51's selected pixel_status leaves 99.9474% of pixels
good under an all-bits, all-three-gain-plane mask. This is not a guarantee about
future database selections or other constants. Freeze all required constant IDs,
payload hashes, shapes/dtypes, segment mapping and the exact mask policy.

Run51 has 17,872 events in the archived PR299 fixture. The starting profile uses
the first 10,000, one GPU for GPU cases, one/two BDs, batch 20, one CUDA stream,
four KvikIO workers, 1 MiB task/bulk targets, 16 MiB bounce buffer and 8 GiB/BD
input budget. Selected Jungfrau streams are 003,005,007,008,009. Verify these
against Configure/SMD and the chosen installed runtime, including resolved
fallback/GDS, sharing, SMD and D2H settings. The new client-cold profile is a new
measurement series; historical PR299 used uncontrolled cache.

The profile includes CPU events and nested-steps controls, but this 10k prefix
is not evidence of multiple native steps or full EndRun coverage. Use separate
correctness fixtures for those boundaries; do not silently change performance
inputs. The already-staged Sim10 100k fixture is a different workload, not a
substitute for this run. Consult the public source and explicit site paths in
profile.json. Public listing/scratch presence does not establish retention.

## Instructions for the implementing agent

1. Create a new campaign ID/root; state the PR question and choose exact base/head
   SHAs. Read applicable psana/GPU/site/review skills and inspect current source.
   Keep runtime feature edits on their feature branch. Do not merge runtime code
   into results/psana2-gpu-scaling to record a result.
2. Copy profile.json and choose **one site/allocation cohort**. Resolve every
   parameter explicitly. Freeze software versions/build options, callback and
   adapter revision, constants options, GPU/CPU placement, kernel compilation
   warmup, memory admission, and requested versus observed configuration. Never
   compare unmatched CPU and GPU workloads as if they were equivalent.
3. Qualify data before setting qualification flags true. Record the retention
   owner, authoritative recoverable source, permission/retention agreement and
   staging instructions; current scratch copies are not that agreement. Verify
   compute-node readability at both sites and identical logical data identities.
   Hash the full selected files or precisely indexed used ranges plus all SMD,
   Configure and reference inputs; record the coverage. Build the selected-event
   identity reference independently of the tested iterator and pin its digest.
   Set requested_bytes from the same explicitly defined input datagram ranges.
   Do not hash hundreds of GB on a login node.
4. For calibration profiles, pin required constants (including pedestal, gain,
   offset/status/extra when used), exact dtype/shape/order and full payload hashes.
   Count rejected pixels per gain and panel and after the chosen combination;
   reject an all-bad or nonfinite mask and all-zero/empty calibrated witnesses.
   Never disable masking or substitute constants merely to pass. Verify numerical
   results against independent CPU references with explicit common-mode, offset,
   threshold and tolerance policy. Raw-only profiles skip constants by design;
   label them raw-only and never claim calibration-regression coverage.
5. Implement a **standalone frozen native adapter** for the chosen revision pair
   using the contract below. Start from PR299 orchestration/accounting and PR187
   cache/first-batch concepts, but verify every API/instrumentation hook against
   both revisions. Do not import or edit frozen campaign harnesses. Keep shared
   parameters in the request; reject any unsupported option instead of falling
   back silently. Preserve adapter dependencies as hashed bundle artifacts.
6. Build isolated installs for both SHAs; freeze the compiler, MPI, Python,
   CuPy/KvikIO/CUDA libraries and environment. Verify imports, native library paths
   and install/source hashes inside every native sample. The adapter clears stale
   PSANA/KvikIO/CUDA overrides before applying the resolved profile. Reject
   uncontrolled environment differences. Sharing toggles need a separate profile.
7. Fill artifact entries with absolute paths and SHA256s, and adapter_command as
   an argv array including the exact hashed adapter path plus standalone
   `{request}` and `{result}` arguments. Include dataset_manifest,
   event_reference, baseline_runtime, candidate_runtime and adapter; add
   calibration_manifest for calibrated profiles and all dependency/reference
   artifacts. Each true qualification flag requires its hashed receipt:
   retention_receipt, cross_site_identity_receipt, compute_access_receipt,
   event_reference, adapter_review_receipt; calibrated profiles additionally
   require calibration_manifest, nondegenerate_mask_receipt and numerical_reference.
   Qualification flags summarize reviewed evidence; they do not prove its truth.
8. Run controller tests and prepare a fresh directory. Review plan/profile,
   native adapter, tests and evidence in a detached worktree using the established
   implement-review-loop. Resolve blocking findings on the latest code commit.
   Preserve the reviewed source bundle and use its controller for every phase.
9. Within task-authorized compute resources, run correctness for both revisions,
   then A/A of the baseline in both positions. Require valid numerical/event
   checks and the predeclared A/A loop-equivalence screen. If noise prevents it,
   report inconclusive and redesign prospectively; do not discard unfavorable
   pairs. Obtain any still-missing job authorization through the existing task
   workflow. Check Perlmutter allocation balance and current site requirements.
10. Run the approved A/B matrix, preserving adjacent pairs and their order. Warmup
    is a separate process before each sample's cache preparation. For client-cold
    use a private data copy/view whose cache can be safely conditioned, and prove
    measured-range residency is zero after setup and immediately before timing.
    Shared server caches remain uncontrolled. Warm requires complete measured
    residency. Do not flush system-wide caches or claim cold from fadvise alone.
11. Audit every configured sample and the scheduler's final status. Preserve
    failures/timeouts, native receipt hashes and raw artifacts. Analyze all pairs
    by workload/topology/site and distinguish primary loop from startup/process
    timing. Diagnostics/Nsight runs use `profile`, never the clean sample pool.
    Review **results** independently; an implementation pass is not a results pass.
12. Add a dated report and manifest under
    `docs/performance/campaigns/<site>/pr-regression/<date>-pr<number>/`, add its
    catalog entry and frozen script root under `scripts/campaigns/<id>/`, then run
    the existing catalog checker. Keep large inputs/arrays/logs in retained site
    storage with manifests. Never copy historical approvals or imply unmeasured
    performance. Publish within the task's existing authorization.

## Native adapter contract

The controller invokes adapter_command without a shell. A request contains the
profile digest, sample ID, phase, selected runtime SHA/install, case, a shared
`contract` and its digest, site, and frozen artifact paths/digests. Adapter work
must match that contract. Recompute `observed_contract_sha256` from actual
resolved settings and include the full `observed_contract` object; the controller
compares that object and computes its digest. Do not merely echo requested values. In A/A both sides use
the same baseline runtime, although their sample IDs retain left/right positions.

The adapter is responsible for site launch (`srun --mpi=cray_shasta` on Perlmutter,
a qualified SDF MPI launch), rank roles/binding, kernel warmup, cache conditioning,
read/byte accounting, native correctness, timing synchronization and cancellation
of outstanding scheduler steps. The controller's process-group timeout alone
cannot guarantee cleanup of remote MPI ranks; prove step termination in the
adapter and retain scheduler records. Return nonzero on any invalid/incomplete
sample. Never submit an asynchronous job and immediately report success.

Write a result JSON only after all ranks and GPU work drain. Required fields:

- Identity: `schema`, `profile_sha256`, `sample_id`, `phase`, `runtime_commit`,
  `observed_contract_sha256`, `status="pass"`.
- Work: global `events`, independently checked `event_sha256`, `requested_bytes`,
  `correctness_pass=true`, `warmup_completed=true`. Event digest serialization is
  fixed by event_reference and identical on both sites/runtimes.
- Timing seconds: positive finite `setup_seconds`, `first_batch_seconds`,
  `loop_seconds`, `process_seconds`. The adapter records rank-wise brackets and
  reduction rules in its native evidence. Do not sum independently maximized
  phases; first batch is included in loop. Process wall includes launch/teardown.
- `cache`: `policy`, `verified`, `resident_bytes_before`, `measured_bytes`.
  Controlled policies need verified resident byte counts for the frozen measured
  ranges. `uncontrolled` must be declared in a separate profile.
- `nic`: nonnegative `rx_bytes`/`tx_bytes`, or explicit null if unavailable.
  Retain interface names, counter units, time brackets and reset checks in native
  evidence. NIC totals are node/client traffic unless attribution is established.
- `placement`: node, per-rank physical CPU affinity map and GPU UUID list (empty
  for CPU-only cases). Include GPU PCI/model/NUMA and MPI roles in native evidence.
  Exact placement must agree across repeats/variants and phases for each case.
  All phases use the same allocation; a new allocation requires a new cohort.
- `evidence`: `runtime_provenance`, `correctness`, `cache`, `resources`, each an
  absolute `path` and `sha256`. Each must be a distinct file inside the sample
  folder (alongside result.json); outside paths and symlink escapes are rejected.
  Preserve additional files through a hashed native
  index. Resources include host RSS, GPU used/pool peaks, pinned-memory accounting,
  allocation/step IDs, final scheduler state and explicit unavailable metrics.

After validation the controller records result/evidence hashes in accepted.json.
`summarize` rejects changes relative to those acceptance receipts. Retain the
whole archive read-only; these hashes do not protect against someone rewriting
all receipts together. Result data and referenced files remain immutable.
`summarize` revalidates profile,
artifacts, requests, exit codes and native receipt hashes. It rejects incomplete
or extra sample directories. Assertions supplied by an adapter are a trust
boundary: independent review/native checks establish their meaning; structural
JSON validation alone does not prove numerical correctness or device placement.
