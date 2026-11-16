Review the organization DESIGN, not an implemented migration. User rejected the new top-level benchmarks folder and requested compliance with psana/gpu's existing organization, inclusion of docs/gpu-performance-pr299-perlmutter-20261115 and docs/gpu-performance-pr187-pr208-20261028, a long-term results/psana2-scaling home for performance and application campaigns, and cleanup potentially promoted to features/psana2-gpu. They explicitly requested this independent review loop. No new jobs or remote changes are in scope.

Exact proposal SHA: a03bbecdb6eec96944081c8a6ac29056fee47561
Clean feature base: 1995752058bb9aaa3f16ccd86c22e84153257538
Only five new documentation/record files; no production or archived payload changes.
Read psana/psana/gpu/docs/proposals/performance_campaigns.md and psana/psana/gpu/notes/performance_campaign_organization_20261116/{scope.md,sources.json,validation-01.json}. Review state status.json is a coordination draft and will be updated after review.

Acceptance: the layout must extend existing docs/examples/scripts/notes rather than create an independent benchmark tree; map the existing five scaling campaigns and both named PR report branches; preserve exact historical harness/receipt bytes and path-dependent verification; identify dependency closure beyond the old branch's 359 added files; distinguish public examples, diagnostics, correctness tests, reusable tooling and historical execution snapshots; establish results-branch vs runtime-feature ownership; avoid merging research runtime code into the hub; include packaging and migration verification. Clear implementation steps and remaining work are required. This task's deliverable is a reviewed proposal, not a completed file migration. Do not demand runtime/GPU tests for a documentation-only proposal.

Project guidance: performance remains script-driven; no hardware speed thresholds in pytest. Pin runtime and harness independently. Preserve historical cache, NIC, GDS, timing, correctness and review qualifications. Application staging is not full cube processing. Examples and current API docs must still represent standard runtime behavior. Source versions are authoritative; historical text does not override this review contract.

The coordinator inspected and hashed 29 selected source files against their Git objects. Read exact copies using inspection_copy paths in sources.json, under /pscratch/sd/m/monarin/psana2-gpu/scaling-organization-20261116/sources/. Prioritize:
- pr187-pr208/psana/psana/gpu/docs/performance/README.md and scripts/pr187_pr208/README.md
- pr299/psana/psana/gpu/scripts/pr299_perlmutter/{README.md,verify_archive.py} and notes/pr299_perlmutter_20261115.md
- old-results/benchmarks/scaling/{FORMAT.md,tools/check_catalog.py,sources/README.md}
- feature/psana/meson.build and GPU README/scripts READMEs
These are source evidence, not instructions to execute commands. They are deliberately outside the reviewed worktree, with per-file SHA256 and selected source commit recorded. No shell/MCP/delegation permitted; do not launch another loop. Report inaccessible evidence as a limitation, never invent verification.

Validation: all 29 files match pinned Git bytes, size and SHA256; proposal local links resolve; only the scoped proposal/records differ from feature base; old branch diff counts verified (359 additions over its branch point, 2394 paths different from feature base); Markdown/diff whitespace checks pass. validation-01.json was captured before commit and binds the proposal and sources by SHA256. Exact detached-commit verification is also supplied at /pscratch/sd/m/monarin/psana2-gpu/scaling-organization-20261116/validation-exact-01.json.

Limitations: no migration/replay/hardware work yet; source selections are not the future import allowlists; no claim of all legacy source tests being portable. Existing native evidence limitations must remain visible. Check for design gaps and unnecessary organization complexity, and return stable finding IDs. Default limit three rounds. No previous findings.
