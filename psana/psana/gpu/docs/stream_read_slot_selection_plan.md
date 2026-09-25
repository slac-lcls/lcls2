# Step 1: bulk-on slot-selection optimization

Status: implemented and measured, 2026-09-25. Jobs 39070635 and 39071037
completed all 22 samples. Validation: 408 CPU unit tests, 15 harness tests,
and 10 A100 device tests passed. Baseline/candidate frozen manifests differ
only in `psana/gpu/gpu_input_group.py`. Selection CPU cost fell over 93%;
cold median loop time fell 5.7%. Warm results varied: +4.3% time in the first
campaign, then -5.9% across four additional pairs. The candidate is retained
for review; a consistent warm throughput benefit across allocations remains
unproven. See the [results](performance/stream_read_slot_index.md).
The original plan below records the scope and gates.

## Target and scope

Optimize `InputGroupPool.plan_slots` first. Its cumulative CPU profile cost
was 0.591–0.603 seconds per 1,000-event bulk-on run, the largest individual
candidate identified in [the profile report](performance/stream_read_cpu_profile.md).
This is measured instrumented cost, not a guaranteed recoverable speedup.

Change only how this bulk-on method chooses available raw slots. Pending-file
cleanup and legacy-plan rebuilding remain subsequent independent steps.
All performance comparisons in this step use **bulk on**: current versus
optimized slot selection, with both versions on the same allocation.

## 1. Freeze the baseline

Use the current worktree, including the validated BeginStep/EndRun deferred
input drain and the public bulk-target parameter. HEAD alone (`c8f6b6cdf`)
does not contain these working-tree changes. Save its source hashes and patch,
then derive a candidate differing only in slot-selection implementation.
Both variants must include the same transition fix and benchmark harness.
Keep frozen runtime copies, logs, profiles, and temporary builds on shared
scratch; preserve the existing dirty worktree and earlier evidence.

## 2. Replace repeated scans with a per-call index

Current selection reconstructs the free-slot list and scans every free cached
capacity for every group. Proposed selection:

1. Poll completion once, as today, then snapshot busy slots and small-stream
   credits. Enumerate free slots in ascending slot-ID order once.
2. Build a sorted list of `(cached_capacity, slot_id)` for free slots that
   actually contain an allocation. Read each cached capacity once.
3. For each group in the original order, use binary search for the smallest
   cached capacity that fits. Equal capacities choose the lowest slot ID.
4. If no cached allocation fits, choose the lowest remaining free slot ID,
   whether unallocated or undersized, exactly as the existing code does.
5. Remove the selected slot from local eligibility; advance the fallback
   free-slot cursor past used entries. If fallback picked an undersized cache,
   remove its entry from the capacity list too. Preserve small-stream credit
   checks and return `None` if the entire request cannot be planned.

The index exists only for one `plan_slots` call. It adds no persistent state to
keep synchronized with trim, allocation growth, completion, or failure. A
zero-byte cached allocation remains distinct from an unallocated slot.
Planning does not allocate device storage, submit reads, or reserve credit.
The existing combined byte hold and full replacement-cost calculation remain
in `_issue_group_reads`; actual issue still verifies availability.

Sorted-list removal may move entries, so this is not a claim of logarithmic
total selection cost. The intended reduction is repeated Python scans and
capacity-property lookups. Measure the actual implementation before accepting.

## 3. Correctness gate

- Compare complete selected-slot tuples against the original selection rule
  over deterministic generated cases: variable capacities, ties, busy slots,
  empty caches, zero-size groups, undersized fallbacks, exhaustion, and multiple
  small requests from the same stream. Verify failure does not consume credits.
- Exercise successive calls after allocation growth, trim, and out-of-order
  completion to prove the per-call snapshot cannot become a stale global index.
- Run the existing CPU GPU suite and targeted A100 group pixel/launch,
  delayed-consumer, partial-failure, and transition regressions. Preserve
  current best-fit choices, byte accounting, holds, and reuse dependencies.
- Keep timing assertions out of pytest. A small standalone CPU benchmark may
  isolate selection cost; the real workload decides the throughput result.

## 4. Bulk-on-only A/B campaign

Keep the current workload fixed: 1,000 JF+feespec events, batch 100, depth 1,
8 GiB device budget, eight KvikIO workers, 4 MiB bulk target and task size,
compatibility mode ON, one BD and three MPI ranks, same private Weka FFB data
and frozen references. Use one A100 allocation for both builds.

- Eight controls: baseline/candidate × cold/warm × two rounds. Reverse build
  and cache ordering in round two.
- Four separate measured-loop profiles: baseline/candidate × cold/warm.
  Compare `plan_slots` time and call counts and check where time moves.
- Two separate warm pipeline diagnostics: one per build, retaining subbatch,
  kernel-launch, reservation, and charged-memory checks.

That is 14 samples, all with bulk on. Profiles and pipeline diagnostics do not
enter control throughput. Preserve warmup raw/calibrated pixel checks,
timestamp/feespec checksums, 5,019 requests and 33,566,911,424 payload bytes,
cache residency, cold NIC bytes, and before/after placement/source provenance.
Record effective runtime settings instead of inferring them from arguments.

## 5. Decision and next step

Require exact correctness and unchanged slot decisions first. Then report
both control rounds, median loop time, events/s, instrumented selection cost,
and memory/launch counters. Accept a performance improvement only if the
selection cost drops materially and the paired controls support a repeatable
benefit without a cold/warm regression. If results are small or inconsistent,
add balanced repetitions before deciding; do not combine another optimization
to hide an inconclusive result.

Review candidate changes and save concise findings with source hashes/job IDs.
Pending-file ownership scanning becomes the next isolated experiment after
this step's result is understood. Broader Stage 4 correctness, 10k acceptance,
and legacy-removal gates remain separate.
