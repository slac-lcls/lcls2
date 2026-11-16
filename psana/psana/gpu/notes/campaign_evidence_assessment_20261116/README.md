# November campaign evidence assessment

Assessment date: November 16, 2026. This guidance concerns the four completed
scaling screens; the Sim10/XPP-cube application snapshot remains pending.
It adds interpretation and follow-up priorities without changing frozen reports,
measurements, scripts, manifests or historical review verdicts.

## What complete means

- **Execution complete:** the planned measured cases finished. Scheduler success
  alone does not establish correctness or acceptance of throughput.
- **Archive verified:** selected artifacts match their recorded hashes, inventory
  and source mapping; documented arithmetic and derivative replay checks pass.
  This does not prove raw measurement correctness or continued external-data access.
- **Independent results reviewed:** a separate review assessed the stated results
  against evidence within an explicit scope. Implementation, migration and results
  reviews are different; one does not silently upgrade another.

The existing catalog already records these states separately. Its top-level
`status: complete` must be read with `lifecycle` and the source review record.
The consolidation reviews checked scientific summaries and arithmetic, while
preserving pending historical results-review states. This assessment is not
itself a new independent campaign-results verdict.

## Follow-up decisions

| Issue | Action for this historical archive | When additional evidence is needed |
| --- | --- | --- |
| SDF original inputs absent from transfer | Retain the explicit gap. Recover the eight missing inputs if readily available, as a hashed supplement; no new job is needed to search for existing files. All 156 normalized rows and 48 summary cells are retained. | Reconstructing the original source-to-normalized-data pipeline or making a full raw-evidence re-audit claim requires the relevant originals. Derivative table/heatmap replay is already supported. |
| SDF final three cohort reviews and synthesis review pending; psana stripe/scaling results reviews pending or unrecorded | Recommend one focused review of existing receipts, sample selection, arithmetic and conclusions before adopting these screens as formal regression baselines. Preserve original pending labels; a later verdict must state its own evidence scope. | Formal results acceptance requires the applicable checks/review; a limited archive review cannot certify missing raw evidence. This documentation task provides neither a new verdict nor baseline qualification. |
| NIC traffic unavailable in C++ and psana stripe screens | Keep unavailable values explicit. Do not rerun historical jobs merely to add this metric; new runs would be a new campaign. Record NIC counters in future campaigns. | Network/storage attribution and bottleneck claims need suitable additional instrumentation; whole-NIC counters alone do not isolate storage traffic. |
| Backend cache uncontrolled | Keep the qualification. Verified local page-cache conditioning does not establish cold backend storage. | Cold physical-storage or storage-ceiling claims require a suitably controlled experiment. Existing reports support observed reader/application throughput under the recorded conditions. |
| Limited repetitions/placement and differing CPU/BD policies | Keep sample counts, ranges, placement and topology visible. Do not require a larger campaign just to archive exploratory screens. | Universal configuration rankings, cross-site causal comparisons or isolated GPU scaling require additional matched experiments. |

## Conclusions supported within the recorded scope

- The [SDF parameter screen](../../docs/performance/campaigns/sdf/sim10-staging/2026-11-14-parameter-screen/README.md)
  supplies a practical staging starting profile. Its storage/GPU comparison holds
  four BDs total fixed and shows little observed gain from adding GPUs.
- The [C++ stripe screen](../../docs/performance/campaigns/perlmutter/cpp-read/2026-11-11-stripe-screen/README.md)
  reports measured aggregate reader rates. Count 1 across five files spans five
  OSTs; 11.42 GiB/s is not a single-OST ceiling or a matched psana overhead control.
- The [psana stripe screen](../../docs/performance/campaigns/perlmutter/sim10-staging/2026-11-16-stripe-screen/README.md)
  supports choosing count 1 as the best tested configuration in that screen.
- The [psana GPU screen](../../docs/performance/campaigns/perlmutter/sim10-staging/2026-11-16-gpu-scaling/README.md)
  shows nearly twice the aggregate node throughput with its tested four-GPU
  configuration versus its own one-GPU control. GPUs, BDs, workers and CPU
  allocation increase together; the result does not isolate the limiting resource.

These are historical exploratory findings and starting configurations, not a
qualified maintained regression baseline or a universal hardware optimum.

## Branch retirement and subsequent work

Retirement requires preserving original source heads with published archive tags,
verifying their exact targets, checking for source-head changes, verifying the
selected import/relocation coverage, and resolving open-PR dependencies. Keep the
existing review and evidence records accessible. This assessment does not perform
or approve deletion of any specific branch.

Missing historical NIC measurements and pending scientific review are not, by
themselves, branch-retirement blockers. Preserve the limitations and reachable
source history rather than requiring new experiments before branch cleanup.

Recommended follow-ups are a bounded search for existing missing SDF inputs and
a focused review of existing results. They are not prerequisites to publication.
Dataset retention/retrieval qualification and the maintained regression template
remain the final planned step; formal future comparisons require their own
qualified workload, checks and review. No jobs, data recovery or new campaign
results review were performed for this assessment.

## Evidence and review scope

The [migration review](../scaling_migration_20261116/review-01.json),
[follow-up review](../scaling_migration_20261116/review-02.json) and
[validation](../scaling_migration_20261116/validation-02.json) record the scope of
prior checks. The linked campaign records retain their native audit and review
status. See [scope](scope.md), [current review status](status.json), and the
[implementation plan](../performance_campaign_organization_20261116/IMPLEMENTATION.md).
