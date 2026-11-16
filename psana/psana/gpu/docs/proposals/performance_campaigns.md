# GPU performance and application campaign organization

Status: proposed organization and migration contract; the data migration and
script refactoring have not been performed. This design is on the new local
`results/psana2-scaling` branch. Its clean base is GPU feature commit
`1995752058bb9aaa3f16ccd86c22e84153257538`. The published
`results/psana-scaling` branch remains unchanged and is a migration source.

## Decision

Use the existing `psana/psana/gpu/` tree (the source path for `psana.gpu`).
Keep user documentation in `docs/`, public examples in `examples/`, runnable
tools in `scripts/`, and development/archive records in `notes/`. Make
`results/psana2-scaling` the integration branch for future performance,
regression, and application campaigns. It is a campaign hub, not the runtime
that every campaign must execute.

Do not retain a second top-level `benchmarks/scaling/` hierarchy. Preserve the
existing curated performance pages and add a campaign index beneath them.
Promote small, independently reviewed tool/documentation improvements to
`features/psana2-gpu`; do not merge the campaign branch wholesale into runtime
development.

## What the repository already provides

The [source inventory](../../notes/performance_campaign_organization_20261116/sources.json)
pins the inspected trees and selected source files by commit and SHA-256.

| Source | Existing organization and consequence |
| --- | --- |
| `features/psana2-gpu` at `1995752058bb9aaa3f16ccd86c22e84153257538` | `docs/performance/` contains current summaries; `examples/` contains public science examples; `scripts/jf_scaling`, `stage1_regression`, `stage5c`, and `feespec_bulk_benchmark` already separate harness families. Extend these conventions. |
| `docs/gpu-performance-pr187-pr208-20261028` at `66072235d3cd18c002a13c581af82e111a418758` | Adds the performance landing page, hardware/history pages, `pr187_pr208.md`, compact evidence, and 47 frozen harness files in `scripts/pr187_pr208/`. Import the documentary additions with their provenance; retain the harness snapshots unchanged. |
| `docs/gpu-performance-pr299-perlmutter-20261115` at `e8d69a7e6604e1d459ad04c633de50a56b46ac9a` | Report is `notes/pr299_perlmutter_20261115.md`, evidence is `notes/evidence/pr299_perlmutter_20261115/`, scripts are `scripts/pr299_perlmutter/`. Its offline verifier checks repository-relative paths and exact package coverage. Preserve this package at these paths. |
| Published `results/psana-scaling` at `9674e3e803354b65c86e42d616af50b23d2c6e3a` | Added 359 files to a research benchmark branch: 323 imported files plus 36 catalog/tool/review files. That is not its full difference from the feature branch: the latter is 2,394 changed files and includes runtime differences. Import selected campaign material, never merge this entire branch. |

The historical PR studies are distinct from Sim10 scaling. PR299 ran Run51
with 10k events, CPU and GPU controls, and uncontrolled cache state. Its
performance acceptance did not approve the functional PR: C1 was unfixed at
the measured head and Run77 was deferred. PR187/208 compares frozen revisions
and sharing configurations, with unresolved remaining-loop equivalence.
Neither becomes a generic regression pass by entering the index.

## Target layout

All paths below are relative to `psana/psana/gpu/`. Existing paths are kept
unless the migration table explicitly says otherwise. New directories are
created when their first real campaign or reusable tool needs them.

```text
README.md                         user entry point; link to performance index
docs/
  design.md, user_kernels.md, ...  standard runtime/API documentation
  proposals/                      designs, including this proposal
  performance/
    README.md                     main campaign and current-summary index
    read_staging.md                curated current staging report
    user_kernels.md                curated current application report
    input_group_bookkeeping.md    existing measured optimization report
    pr187_pr208.md                 preserve reviewed report path
    hardware.md, history.md        preserve the report branch's pages
    FORMAT.md, catalog.json        campaign contract and searchable inventory
    evidence/                     existing compact evidence stays in place
    campaigns/
      <site>/<workload>/<date>-<study>/
        README.md                 findings, scope, status, source links
        manifest.json             execution identities and artifact index
        runs.csv, summary.json    compact measurements, when available
        figures/                  report figures, when available
        review/                   readiness/results reviews, when available
examples/                         supported user-facing examples
scripts/
  README.md                       tool index: maintained / experimental / frozen
  jf_scaling/, stage1_regression/, stage5c/, feespec_bulk_benchmark/
                                  existing harness families and helper tests
  pr187_pr208/, pr299_perlmutter/  unchanged historical execution packages
  performance/
    README.md                     shared campaign tooling contract
    common/                       proven shared receipts, timing, validation
    sites/                        explicit sdf/perlmutter adapters
    analysis/                     catalog check and offline report replay
  campaigns/
    <campaign-id>/                 new campaign configuration and orchestration
                                  includes workload callbacks if experimental
notes/
  ...                             existing development records
  evidence/pr299_perlmutter_20261115/  unchanged PR299 package
  archives/
    scaling-20261116/              preserved legacy Sim10 source/evidence bundles
                                  old relative hierarchy + relocation manifest
```

Keep the present site/workload/date directory convention for the five Sim10
catalog entries; do not churn their stable campaign IDs. `kind` is an indexed
field (`scaling`, `regression`, `application`, or `diagnostic`), not another
directory layer. A multi-site synthesis has its own `site: multi-site` record
and links separate site campaigns; it never pools their trials implicitly.

This extends standard docs/scripts/notes responsibilities. There is one
authoritative copy of each report or evidence file. A campaign README may
summarize and link an existing reviewed report (PR299, for example) without
copying its body or evidence. Legacy source bundles are explicitly frozen
archives; they are not maintained code simply because they contain Python.

## Concrete migration map

| Current material | Destination/action |
| --- | --- |
| `benchmarks/scaling/campaigns/{sdf,perlmutter}/...` | Move authored reports/manifests to `docs/performance/campaigns/` with the same site/workload/date suffix. Rewrite authored relative links and repository-relative artifact paths. |
| `benchmarks/scaling/{README.md,FORMAT.md,catalog.json}` | Integrate into `docs/performance/{README.md,FORMAT.md,catalog.json}`. Reconcile the PR187 landing page so PR187/208, PR299, historical staging/user kernels, Sim10, and application work are all discoverable. Preserve dated qualifications; do not label an old revision study as the latest overall result. |
| `benchmarks/scaling/tools/` | Adapt into `scripts/performance/analysis/`; remove fixed-depth root inference and assumptions that exactly the five initial campaigns exist. Keep arithmetic/negative checks and SDF replay. |
| `benchmarks/scaling/templates/` | Move to `docs/performance/templates/`; use one template for the common record plus kind-specific fields. |
| `benchmarks/scaling/review/` and `validation.json` | Preserve under `notes/archives/scaling-20261116/catalog-review/`. These attest to the old catalog SHA, not the relocated package. Record fresh migration verification separately. |
| `benchmarks/scaling/sources/` | Move the exact historical package under `notes/archives/scaling-20261116/sources/`. Preserve `reference/` and `latest/` as distinct snapshots with their original bytes, package receipt, missing-input record and hashes. |
| Root `notes/sim10_perlmutter_baseline_20261115/` (95 newly imported files) | Archive under `notes/archives/scaling-20261116/legacy-root/notes/sim10_perlmutter_baseline_20261115/`; the relative tree is evidence, not an executable installation. Never copy a live dirty worktree in place of the pinned snapshot. |
| Inherited root `notes/sim10_striping_20261110/`, `notes/psana_striping_pr298_20261111/`, `notes/psana_gpu_scaling_20261116/` and transitive source dependencies | Enumerate from the old manifests, verifier paths, bundle/import code and report links. Import the complete selected dependency closure under the same `legacy-root/` prefix. These are required beyond the 359 added files. Account for every selected path and mark historical missing inputs explicitly. |
| PR187/208 doc branch | Select only report/index/hardware/history updates, compact evidence, supporting GPU notes and `scripts/pr187_pr208/` (47 snapshots plus archive metadata). Three-way reconcile edits to existing GPU README/current reports; never overwrite newer API docs from that branch. Add site campaign records plus one multi-site synthesis linking the existing report. |
| PR299 doc branch | Import its added `scripts/pr299_perlmutter/`, GPU note and `notes/evidence/pr299_perlmutter_20261115/` intact. Add a `perlmutter/pr299-run51/2026-11-15-regression/` catalog wrapper pointing to them; do not relocate files covered by `SHA256SUMS`/`archive-map.json`. |
| Existing feature-branch reports and `validation/input-bookkeeping-20261110/` | Retain existing locations and links. Index them as existing evidence; do not create a second copy beneath the new campaigns tree. |

Archive relocation must add an old-to-new mapping with source commit, old
path, new path, byte size and SHA-256. Original receipts and embedded paths
stay byte-for-byte unchanged. New manifests identify relocated files and
their unchanged hashes. A separate source digest describes any subsequently
adapted script. Identical reference/latest files may be recorded as duplicates
but are not silently deduplicated; no symlink or content-store scheme is needed
for these small historical bundles.

The migration manifest must classify every path in the old 359-file addition
as moved, integrated, or retained through an explicit immutable reference,
and separately enumerate inherited dependencies. Its accounting must close
before the old landing page is retired. Old URLs remain valid on the old
branch; new links target the new branch after publication. Do not force-push,
delete, or rewrite the old published branch to achieve the new layout.

## Scripts: consolidate by responsibility, not by filename

The existing scripts are not all performance tests. They include timing
harnesses, diagnostic tracing and read-plan previews, source-complexity
inventory, correctness/reference preparation, campaign orchestration, and
application callbacks. `examples/` has the supported public API examples;
production implementation remains in the GPU package modules and its normal
unit/integration tests remain in `psana/psana/tests/gpu/`.

| Class | Treatment |
| --- | --- |
| Frozen `pr187_pr208`, `pr299_perlmutter`, SDF/Sim10 execution snapshots | Preserve exact bytes and dependencies. Never redirect historical imports to a new common helper or submit their old Slurm wrappers directly. |
| Maintained `jf_scaling`, `stage1_regression`, `stage5c`, `feespec_bulk_benchmark` | Retain callable paths initially. Inventory actual imports/callers before extracting helpers; several families already depend on each other. Preserve compatibility entry points when paths eventually change. |
| Shared cache residency/conditioning, runtime provenance, CPU/GPU placement, NIC and memory observations | Candidate `scripts/performance/common/` and `sites/` modules. Require two real consumers and a written measurement contract before extraction. Do not create a generic framework of empty directories. |
| Timing/paired statistics, archive checking, plotting | `scripts/performance/analysis/`; pure offline checks run without CUDA/MPI. Preserve original paired blocks, signs, statistics, missing values and timing scopes. |
| Diagnostics (`preview_stream_read_plan.py`, fallback tracing, complexity inventory) | Keep in `scripts/` with an indexed purpose. Do not relabel them as throughput/regression tests. |
| New Sim10/XPP-cube experimental callback, reference and runner | `scripts/campaigns/<campaign-id>/`; pin the application source independently of runtime and harness. Promote a small portable user example to `examples/` only after its API and numerical policy are validated. |

First extraction candidates are read-only artifact hashing/provenance and
offline summary logic. Cache eviction, NIC selection and launch placement need
site-specific tests and a new controlled validation run before reuse in a
measured campaign. Similar names are insufficient: SDF local scratch, Weka
and Lustre have different cache meanings; node NIC traffic is not synonymous
with application storage bytes. Full cache eviction/preflight must precede
each new cold case and be recorded, but historical PR299 remains explicitly
uncontrolled rather than being retrospectively called cold.

Keep throughput thresholds out of pytest. CPU tests validate harness
contracts, negative paths, pairing and archive integrity; GPU tests validate
correctness. Regression judgments come from a separately authorized matched
campaign and an evidence-based report, including inconclusive outcomes.

## Campaign record and lifecycle

The common manifest records a stable ID, kind, site, workload, date, and
separate execution, correctness, results-review and archive-verification
states. Use statuses that allow planned, running, failed, complete-unreviewed,
accepted and inconclusive results; do not infer acceptance from Slurm exit 0
or an implementation review. Record snapshot time and superseding campaign ID.

Pin runtime source and native-build receipt, harness source, application
source, analysis version, dirty/package hashes, dependencies, data identity,
job/node/topology, GPU mapping, CPU affinity, storage/stripe placement,
cache/warmup, GDS/compatibility state, batch/worker/stream/budget parameters,
correctness gates and raw-artifact availability. Record omitted evidence as
unavailable, not zero. `artifact_index` names paths, sizes, hashes and durable
locations; large XTC data, installs, traces, event arrays and full cubes remain
outside Git. Scratch alone is not durable retention: accepted new campaigns
must identify an owner and retained evidence location, or state the limitation.

Per-case rows preserve case ID, paired/block/repetition IDs, execution order,
variant, exact comparison commits, timing bracket, global events and byte
denominator, stage and resource observations. Derived metrics name units and
formulas. CPU/GPU controls with different work are not equal-work speedups.
One-OST-per-file does not mean one OST for all five input files. Raw C++ file
bytes, psana requested payload bytes and NIC bytes remain separate.

Regression records additionally specify baseline/candidate, A/A controls when
needed, pairing/order, cache policy, primary metric and any predeclared
decision threshold and uncertainty method. Report mean/median of paired
changes explicitly; preserve startup, first-batch, loop and total-wall scopes.
Uncontrolled storage and small sample counts can leave a comparison
inconclusive even when every case completes.

Application records additionally specify expansion/calibration/cube semantics,
shape/dtype, bin definition, reduction/export policy, numerical reference and
tolerances, correctness outputs and memory peaks. Report S/E/EC/ECC separately.
Do not advertise a full application result from a staging test. The archived
Sim10/XPP-cube record remains a pending historical snapshot: its bootstrap
retained distributed cubes with shape/content/bin checks, without CPU-reference
parity or export/reduction. Consult the active task before importing later
results; this organization task does not update its job status.

New campaign work branches start from `results/psana2-scaling` and return
reviewed scripts/configurations and records to it. Scratch execution uses
explicit separately built/pinned runtimes; it must not accidentally import the
hub checkout's psana via cwd/PYTHONPATH. Runtime feature work stays on its
implementation branch and is referenced by immutable identity. Freeze each
attempt; do not edit a queued/running bundle. New attempts get new identities.
Execution readiness, result review and archive relocation review remain
separate. Creating a campaign directory never submits a job.

## Branch and cleanup policy

Start the replacement hub from the inspected feature head, not by renaming or
merging the old research-derived results branch. This design branch establishes
that base; importing the campaign material is a subsequent implementation.
Import only the selected report/script/evidence changes from the three source
branches. Record source SHAs and path inventories so the import can be audited.
Do not merge either historical docs branch wholesale simply to recover its
archive. Reconcile conflicting README/current-report edits by hand.

Keep the hub's runtime aligned with a documented feature commit. Update it
through explicit reviewed synchronizations that preserve campaign documents
and do not relabel old measurements; every campaign still pins what it ran.
The hub is not a second runtime-development branch. On each synchronization,
diff runtime paths against the selected feature SHA and account for any
exceptions. Runtime changes do not enter through archive imports.

Cleanup for `features/psana2-gpu` is a separate small patch series: tool index,
portable shared helpers, compatibility entry points, tests, current-summary
links and packaging policy as appropriate. Cherry-pick those reviewed changes
back to the hub after integration. Do not cherry-pick frozen receipts or old
campaign status into production just to get a helper. Full campaign history
stays on the results branch; standard docs/examples remain meaningful on the
feature branch. Cross-branch links must be explicit and use published immutable
URLs when reproducibility matters, rather than relative paths to absent files.

Packaging needs an explicit cleanup: `psana/meson.build` currently excludes
`gpu/notes`, but not new campaign docs or scripts. Do not claim that placement
alone keeps them out of installed packages. Before building/distributing the
consolidated tree, define source-only campaign paths (including historical
`scripts/pr187_pr208`, `scripts/pr299_perlmutter` and `scripts/campaigns`), check
Meson and wheel inclusion independently, and verify maintained docs/examples
and callable tools remain available. Never add campaign modules to public
`psana.gpu` exports as a side effect of consolidation.

## Migration sequence and acceptance

1. Establish the clean hub and review this design. No runtime edits, job
   submissions, historical archive mutation, or remote branch replacement.
2. Import the two PR documentary packages selectively. Run their native
   offline archive checks, preserve caveats, and reconcile the performance
   index with current feature documentation.
3. Migrate the five scaling records, their source packages and inherited
   dependency closure. Produce a complete relocation ledger. Update only
   authored link/path metadata and new adapters; preserve source hashes.
4. Adapt the catalog validator and SDF replay to explicit roots and extensible
   campaign dispatch. Unknown kinds/schemas fail clearly. Test missing/corrupt
   artifacts, duplicate IDs/cases, wrong summaries and escaped paths. Recompute
   the existing 240 archived cases (156 SDF, 54 C++, 18 stripe, 12 GPU scaling)
   without claiming recovery of missing original SDF cohort inputs.
5. Check all authored links, figures, hash coverage, source/package mappings,
   exact runtime/example equality to the selected feature base, and declared
   documentary changes. Run PR299's strict verifier plus tampering tests and
   PR187's 47-file hash check plus existing CPU analysis tests. Replay original
   PR187 aggregation only if its required raw inputs are available; otherwise
   retain the prior receipt and explicitly limit new verification. Source
   snapshot tests with absolute site paths are not assumed portable.
6. Review the relocated implementation independently, with an exact commit and
   manifest of excluded/unavailable inputs. Publish the replacement branch
   after that task's publishing authorization. Keep the old branch as an
   archive; a later authorized pointer update can mark its index superseded.
7. Extract reusable tools incrementally, each with its own consumers/tests and
   review. Any helper change affecting timing, placement, cache, I/O or GPU
   ownership needs a fresh controlled validation before performance reuse.

The completion criterion for migration is a discoverable, verified campaign
hub with no accidental runtime changes and no lost provenance. The completion
criterion for this task is the reviewed design, source inventory and review
record; migration and new performance measurements are not claimed here.
