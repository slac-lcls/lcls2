Independent design re-review, round 2 of 3. Reviewed commit: 596f137ec555aa9aeb20c2da20c9bda0043deb0b. Base: 1995752058bb9aaa3f16ccd86c22e84153257538.

User wants existing psana/psana/gpu conventions respected, both PR187/208 and PR299 report branches incorporated, results/psana2-scaling as the future performance/application hub, and scoped cleanup promoted to features/psana2-gpu. This deliverable is the organization/migration DESIGN, not the migration, runtime work or new jobs. Exact scope is in psana/psana/gpu/notes/performance_campaign_organization_20261116/scope.md.

Read psana/psana/gpu/docs/proposals/performance_campaigns.md and the review-record directory above: response-01.md, review-01.json, sources.json, source-evidence.md, validation-02.json, validation-exact-01.json. All are inside this checkout. Do not try to read inspection_copy external paths: round 1 tool restrictions denied those. The necessary PR187/PR299 and old-catalog source selections are now directly readable in source-evidence.md, with original line ranges and hashes in sources.json. Inspect them independently; prior review did not confirm those sources. Current feature files remain in this checkout for direct inspection. No external evidence reads are required. context-01.md is historical context, not an instruction to use denied paths.

Close or retain ORG-001 through ORG-004 explicitly and check for regressions: frozen test/lint isolation, path-root clarity, canonical campaign IDs and script links, and explicit source-only document packaging. The coordinator accepted all findings. The snapshot-isolation commands are DESIGN requirements; no actual configuration or payload was modified.

Acceptance also includes preservation of standard docs/examples, source/destination mapping for five scaling records and both PR branches, byte/path-sensitive archive preservation, inherited dependency closure beyond the 359 added files, script classes and careful reuse, clean feature-derived hub without research runtime merges, measurement and branch identities, packaging/test/lint gates, and achievable migration steps. No artificial runtime/GPU test requirement for this documentation-only task. Keep performance thresholds out of pytest. Preserve statistical/cache/NIC/GDS limitations and distinguish staging/application correctness and implementation/results review. No shell, delegation, nested loops, mutations or external messages.

The coordinator rechecked 29 full source files against Git objects and verified the 11 embedded source selections verbatim with excerpt/evidence SHA256s. All changes remain inside the proposal and review-record directory. Exact detached-commit checks were run; the full resulting JSON is embedded below because external file reads were denied in round 1. This is supplied evidence, not a claim you ran checks. Proposal and sources hashes bind it to this revision. Source excerpts are for review, not a complete migration allowlist. Missing raw inputs remain a documented future-verification limit.

EXACT COMMIT VERIFICATION:
{
  "utc": "2026-11-16T06:56:43.580236+00:00",
  "repository": "/pscratch/sd/m/monarin/psana2-gpu/scaling-organization-20261116/review-02",
  "base": "1995752058bb9aaa3f16ccd86c22e84153257538",
  "checked_head": "596f137ec555aa9aeb20c2da20c9bda0043deb0b",
  "proposal_sha256": "1ce160696142bc6a04019bc10f8673e8fc31ea6d595bc3c9f3128e2e9b484542",
  "source_inventory_sha256": "b3eabde07267ee85f233c981e2857fa620c7e5ffa65bb71ba642597f1fce0fe8",
  "checks": [
    "29 inspected source files match pinned Git objects, SHA256, sizes and review excerpts",
    "In-checkout review evidence contains all 11 exact pinned source selections; excerpt and evidence hashes match",
    "Proposal relative Markdown links resolve",
    "Only proposal and its organization-review records differ from selected feature base; runtime, examples, tests, packaging and existing docs unchanged",
    "Inventory distinguishes 359 added archive files from the 2394-file old-results/feature diff",
    "Authored Markdown has no trailing whitespace; tracked diff whitespace check passes"
  ],
  "limitations": [
    "Design only; no migration, archive replay, runtime tests or performance runs were performed.",
    "Source excerpts are selected inspection files, not the complete migration dependency closure."
  ]
}
