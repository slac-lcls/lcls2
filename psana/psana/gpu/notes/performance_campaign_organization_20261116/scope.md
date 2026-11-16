# Organization design review scope

User request: comply with existing GPU organization, account for the PR299 and
PR187/208 performance report branches, make results/psana2-scaling the home for
future performance/application campaigns, and distinguish cleanup suitable for
features/psana2-gpu. Use the Codex/Claude review loop.

Deliverable: a concrete, source-backed organization and migration design.
Proposal: ../../docs/proposals/performance_campaigns.md.
Base: 1995752058bb9aaa3f16ccd86c22e84153257538.
Branch: results/psana2-scaling; preserve published results/psana-scaling.
Exclusions: data migration, runtime refactoring, benchmark jobs, remote writes.
Acceptance: inspect both named branches; preserve docs/examples conventions;
give source/destination and ownership rules; address historical archive paths,
script reuse, measured-runtime identity, install packaging and verification;
review exact committed proposal with independent Claude; fix blocking findings.
Default maximum: three review rounds, 15 minutes each.
Tests: source/path/hash and document consistency checks only. Runtime and GPU
tests are unnecessary for a design-only change; no new benchmark is claimed.
