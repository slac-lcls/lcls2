# Campaign consolidation implementation

The user selected **results/psana2-gpu-scaling** on November 16, 2026.
This is the local branch and published upstream branch in slac-lcls/lcls2.
It supersedes results/psana2-scaling wherever that name
appears in the previously reviewed organization proposal and review records.
Those historical records remain unchanged.

Base: 1995752058bb9aaa3f16ccd86c22e84153257538
(features/psana2-gpu, including PR298).

Completed: establish the clean feature-derived local branch, retain the
reviewed design and its evidence, apply the user-selected branch name, and
import both PR documentary packages. The import passed independent Claude
review (round 2, no findings) at 4d462252b80370fde35b12af9d490ecef5c28fbc; see
[import records](../pr_campaign_import_20261116/README.md). The packages contain
703 selected source paths: 700 source-identical files and three adapted
navigation pages. Frozen campaign scripts and evidence are unchanged.
The scaling migration is also complete: 1631 mapped source paths, including all
359 original additions and 1272 inherited dependency paths; 1612 files remain
byte-exact. The unified catalog has nine entries. Claude round 2 passed
9c4efbde2109ccccb56fe13edbb47ffbabc3db5d with no findings; see
[migration and validation records](../scaling_migration_20261116/README.md).
The worktree remains at
/global/u2/m/monarin/lcls2_worktree/benchmarks/psana2-scaling-results.
The directory name does not determine the Git branch name.

Implementation steps from the agreed plan (steps 1–5 complete; step 6 next):

1. **Complete:** selectively import the PR187/208 and PR299 documentary packages,
   preserving frozen bytes and PR299 archive paths; integrity, isolation and
   independent import review passed.
2. **Complete:** migrate the old fork scaling catalog and its selected dependency
   closure into the reviewed GPU layout, with complete path/hash accounting.
3. **Complete:** validate archive integrity, all 240 arithmetic rows, SDF replay,
   links, test/lint isolation and Meson/Hatch package selection; independent
   implementation review passed. Full native build/production wheel and new
   performance measurements were outside this migration.
4. **Complete:** publish the verified branch to slac-lcls/lcls2 and verify the
   remote tip, campaign index and review receipt. See the
   [publication receipt](../scaling_migration_20261116/publication.md).
5. **Complete:** preserve all three exact source heads with published annotated
   archive tags in their original repositories, verify the tag objects and peeled
   commits, recheck source heads and PR dependencies, then retire all three remote
   branches and obsolete local references. The clean source scaling worktree was
   removed; the canonical results worktree remains. Claude round 1 passed the
   retirement procedure with no blocking findings. See the
   [retirement record](../campaign_branch_retirement_20261116/README.md).
6. Last: qualify a retained/retrievable dataset on Perlmutter and SDF, then
   implement and review the maintained regression template. This step must not
   delay preserving or consolidating the completed campaign evidence.

The completed local work includes both PR-study import and scaling-catalog
migration/validation, upstream publication, archive tagging and source-branch
retirement. Dataset qualification and the maintained regression template remain
pending. Runtime and example code remain unchanged; no new benchmark was submitted.

## Campaign PR and issue cleanup

Completed November 16: documentary PR245 and PR314 are closed without merging
because their campaigns moved to results/psana2-gpu-scaling. Their bodies and
conversation notices link to the consolidated results. PR299's performance
reference is updated; that implementation PR remains open. Searches across
slac-lcls/lcls2 and monarin/lcls2 found no standalone obsolete-campaign issue
requiring closure. Review findings and historical commit citations are preserved.

The [cleanup record](../campaign_link_cleanup_20261116/README.md) contains scoped
search evidence, exact reviewed payloads, Claude round-2 pass and verified
execution receipts. Source branch tagging and deletion are now complete; the
[retirement receipt](../campaign_branch_retirement_20261116/execution.json) records
fresh source-head/dependency checks, archive verification and local/remote cleanup.

## Historical evidence follow-ups

The November 16 assessment adds the four completed scaling campaigns to the
history page without changing their original results or review status. Recover
missing SDF source inputs if readily available; preserve any recovered inputs
as a separately recorded supplement, leaving the frozen transfer unchanged.
A focused results review is recommended before adopting these screens as formal
regression baselines. Neither follow-up requires rerunning the historical jobs
or blocks preservation and branch retirement by itself. No recovery or new
results review has been performed by this documentation update.

## Reusable regression campaign (final implementation step)

User direction, November 16: the two PR studies should inform one maintained
performance regression campaign that can compare revisions on the same
experiment/run and parameters. This is a planned extension of the organization
work, not an implemented runner or a newly reviewed runtime change. The user
subsequently specified this as the last step, after campaign consolidation,
and requested dataset qualification on both Perlmutter and SDF (sdfiana),
including evaluation of the artificial prjdatpsana XTC fixture.

Code comparison used these pinned sources:

- PR187/208: 66072235d3cd18c002a13c581af82e111a418758,
  psana/psana/gpu/scripts/pr187_pr208/{ada,perlmutter}/.
- PR299: e8d69a7e6604e1d459ad04c633de50a56b46ac9a,
  psana/psana/gpu/scripts/pr299_perlmutter/.

The Perlmutter check_runtime_provenance.py helpers are byte-identical:
SHA256 89506e59c60dca4cd47192cd8222d2d6ebf1e4fc474b9b206844912daec0aceb.
Both orchestrators freeze runtime/harness inputs, use fresh MPI samples,
perform preflights, balance variant order and retain per-rank timings,
placement and resource observations. They contain fixed historical paths,
case counts, versions and acceptance assumptions; neither is directly reusable.

Important differences to retain in the historical records:

- PR187 Perlmutter startup: mfx101232725/run442, calibrated Jungfrau,
  A/B-off/B-on, four BDs sharing one GPU, 1k/10k events, six balanced blocks,
  batch 20, one stream, eight KvikIO workers, client-cache-cold checks.
  The broader PR187/208 archive also has Ada raw/calibration A/B/C comparisons.
- PR299 Perlmutter: mfx100848724/run51, raw-input GPU events and CPU
  events/nested-steps controls, baseline/head, one/two BDs sharing one GPU,
  10k events, four alternating pairs, batch 20, one stream, four workers,
  uncontrolled cache. It does not provide calibration or multi-step acceptance.

Implement a maintained scripts/performance/regression/ runner, with shared
site/provenance/analysis helpers in the existing proposed performance tree.
Archived PR harnesses stay byte-for-byte unchanged, including campaign-specific
helpers: do not redirect their imports to maintained code. The maintained runner
is a template for creating new frozen campaigns, not a replacement for the
scripts that generated historical measurements. Every future campaign retains
the exact runner/harness revision, any local changes as a hashed source bundle,
the fully resolved profile, launch commands, environment and analysis version.
Template updates never retroactively change a recorded campaign.
Campaign reports and receipts use the
common docs/performance/campaigns/ catalog, rather than a separate results tree.

A versioned profile pins experiment/run, data identity and selected event IDs,
streams/detectors, workload/callback and calibration-reference hashes, iterator,
event count, topology, physical CPU/GPU assignment, batch/stream/bulk/task/
bounce/worker/budget settings, cache policy, environment/build options,
correctness checks, warmup, timing definitions and analysis policy. The runner
resolves defaults before execution and rejects baseline/candidate mismatch.
Only declared experimental factors (normally runtime SHA/install) may differ.
A sharing-toggle study is an explicit separate experiment. Compare each
workload/topology cell only with its matched counterpart; CPU and GPU controls
are not automatically equivalent workloads. Record observed placement and
runtime settings as well as the requested values.

Use PR299's baseline/head orchestration and integrity/placement/accounting
checks as a starting point; add PR187's explicit cache conditioning and
setup/first-batch/loop/total timing where compatible. Verify every adapted
instrumentation hook against the chosen revisions; do not carry old private
API assumptions forward. Clean throughput, correctness and profiling are
separate modes. Preserve missing observations as unavailable rather than zero;
NIC counters describe node/client traffic unless attribution is established.

Run51 is one candidate, not the selected permanent dataset. Its historical
profile is 10k events, one GPU, one/two BDs, raw GPU input plus CPU events/steps
controls. Run51's archived fixture has 17,872 events;
it is not a 100k profile and its timed prefix does not validate multi-step/EndRun.
Qualification must establish current data availability, exact input identities
and constants before adopting this as the maintained baseline. Use one fixed
cache policy for new comparisons; controlled-cold results are a new profile,
not a continuation of the historical uncontrolled-cache PR299 measurements.
Calibration can use that same dataset only after references/constants are
qualified; otherwise publish a separately identified profile, never silently
switch datasets. Multi-step correctness also needs an explicitly qualified
fixture. Sim10 100k/application scaling remains a separate workload profile.

Before first performance reuse: CPU tests for manifest equality, scheduling,
missing/failed cases and paired statistics; native correctness on each runtime;
then an A/A control on the selected site/profile to check noise and harness
symmetry. A/A completion is not proof of universal equivalence. After independent
review, conduct a separately authorized matched A/B pilot/campaign and preserve
all pairs and execution order. Report primary loop performance plus startup and
process wall time separately; permit inconclusive results. Hardware throughput
thresholds remain outside pytest. No jobs or new runner code were produced by
this source comparison.

### Dataset qualification before adopting the template

Current availability is not a guarantee of future retrieval. At the final
implementation step, inspect the real-run candidates and the existing
prjdatpsana XTC fixture before deciding whether to reuse or create artificial
data. No availability or suitability claim for prjdatpsana has been made yet.

For each candidate, record:

- Exact experiment/run and fixture version; XTC and SMD files, event selection,
  transitions, detector/configuration metadata, sizes and checksums.
- Actual locations and read permissions on Perlmutter and SDF, verified from
  compute jobs as well as the access/coordinator host. Login visibility on
  sdfiana alone does not establish compute-node access.
- A retention owner, durable authoritative source, permission to retain the
  inputs, and a documented retrieval/staging procedure for both sites. Scratch
  copies alone do not establish durable availability.
- Calibration constants and independent numerical/event references needed by
  the selected workload; availability must not depend on an unrecorded live
  calibration-database response.
- Event count, payload sizes, segmentation, step/EndRun coverage and supported
  workloads. Artificial fixtures must be representative for the regression
  being tested; a tiny fixture passing correctness is not a storage benchmark.

Prefer an existing qualified fixture. If prjdatpsana can supply the required
workload, preserve its identity and generator provenance. If a new synthetic
fixture is needed, version the generator, seed, parameters, dependencies and
output manifest; verify reproducibility or preserve authoritative generated
bytes when byte-identical regeneration cannot be guaranteed. Validate the
same logical dataset and references on both sites. Site paths, hardware and
storage remain explicit configuration; cross-site absolute rates are not a
matched runtime comparison. Baseline/candidate comparisons use identical
settings within each site's allocation.

If the real and artificial datasets cover different workloads, keep separate
named profiles and state their limits. Never silently substitute one dataset
for another. Each campaign checks availability and identity before execution,
and records a retrieval failure as a blocked run rather than changing inputs.
Dataset transfers/generation and qualification jobs belong to that later step;
this plan update performs none of them.
