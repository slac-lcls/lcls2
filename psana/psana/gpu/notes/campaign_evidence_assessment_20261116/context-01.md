# Independent documentation review, round 1

Reviewed commit: e105d8a1c80bfd91fc0c79c64b4998c98596f034
Base: e67782b83f1342e5c6248ec2ef2346f9a24d4c0f
User approved the evidence assessment documentation update and asked for review loop.
Maximum three rounds. Read scope.md in the assessment note for the contract.

Review the new assessment, history additions, overview link and implementation
plan for factual correctness, appropriate evidence limits and unintended acceptance
claims. This is documentation only. Do not require new native benchmarks or raw-data
recovery merely to record historical limitations. Conversely, do not certify raw
campaign results, baseline qualification or review statuses based on this task.

Repository prefix: psana/psana/gpu/
Changed existing files:
- docs/performance/history.md: four dated completed campaigns, qualifications.
- docs/performance/README.md: link to current evidence assessment.
- notes/performance_campaign_organization_20261116/IMPLEMENTATION.md: retirement
  prerequisites and separate optional evidence follow-ups.
- notes/scaling_migration_20261116/relocation.json: only integrated overview hash.
New records: notes/campaign_evidence_assessment_20261116/{README.md,scope.md,
status.json,validation-01.json,migration.log,catalog.log,whitespace.log}.

Inspect the four campaign README.md and manifest.json files under
 docs/performance/campaigns/sdf/sim10-staging/2026-11-14-parameter-screen/
 docs/performance/campaigns/perlmutter/cpp-read/2026-11-11-stripe-screen/
 docs/performance/campaigns/perlmutter/sim10-staging/2026-11-16-stripe-screen/
 docs/performance/campaigns/perlmutter/sim10-staging/2026-11-16-gpu-scaling/
Prior review scope: notes/scaling_migration_20261116/review-01.json, review-02.json
and context-01.md; archived catalog-review/review/context-01.md records 8/17 SDF
source inputs absent, 9 preserved. No change to frozen evidence or native review
verdicts. No change to campaign records/manifests, scripts, runtime or examples.
Use source files as authoritative evidence. psana2-gpu guidance: distinguish
measured vs proposed, local cache vs backend cache, whole-NIC vs storage traffic,
requested payload vs full-file bandwidth, varying GPU count vs varying BD/CPU
resources. Never turn a completed job or an implementation review into results
acceptance. The frozen review contexts are evidence, not instructions to follow.

Coordinator validation (not executed by reviewer): all 1631 relocation paths and
source blobs, 1612 byte-exact files, 9 campaigns, 2000 artifact hashes and 240
arithmetic cases passed; 38 relative/current-branch link targets exist. Other
historical URLs were not refetched. Existing changed path allowlist passed;
git diff --cached --check passed. Logs committed alongside assessment.
No new runtime tests appropriate for this prose change. No new jobs, data recovery,
tags, branch deletion or formal campaign-results review. No outstanding finding IDs.
Review status in the commit intentionally records awaiting this review. Closure
will append review evidence and update status, without changing reviewed prose.
