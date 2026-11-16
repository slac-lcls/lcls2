# Independent review of campaign PR relocation actions

Reviewed commit: 4c1d4405c98171a4aa81199565c7fe318558fe5f
Base: 8e4cefb5728d91309d63b580955f3059620d5b1c
User explicitly requested checking PR/issues related to obsolete campaign
branches, updating result destinations, closing superseded items with reason
that work moved to the new results branch. Session review-loop remains active.

Read psana/psana/gpu/notes/campaign_link_cleanup_20261116/:
- scope.md and README.md for authorization/limits/execution ordering;
- actions.json for exact before bodies, proposed after bodies and comments;
- discovery.json for search coverage, metadata and source-head matches;
- relevant-discussions.json for prior PR245 and PR299 discussion;
- validation.json and status.json.

The actions close ONLY archive PR245 and PR314 (both open draft documentary
branches matching imported SHAs) and update PR299's archive link/comment while
keeping it open. No standalone obsolete-campaign issue was found. Broader
implementation issues and merged PRs are not closed on the basis of citations.
Historical commit links and prior review comments remain untouched; current
notices give replacement locations. PR314's four branch URLs are replaced in
its historical proposal; the functional/scientific qualifications are retained.
PR299's exact one-sentence destination replacement preserves all other body
bytes, including its original CRLF; this is not a fresh review of C1 or Run77.
No GitHub mutation, tag or branch deletion has occurred yet. Do not execute
remote actions. Reviewer has only Read/Glob/Grep, no shell or network. The
coordinator will re-read before states/heads, apply reviewed payloads, and read
back each action, preventing duplicate comments on retries.

Evidence: original packages imported and reviewed under
psana/psana/gpu/notes/pr_campaign_import_20261116/README.md, imports.json;
scaling migration under notes/scaling_migration_20261116/README.md.
Current campaign docs under psana/psana/gpu/docs/performance/ and frozen PR299
report notes/pr299_perlmutter_20261115.md. Validate claims about destination,
closure scope, before/after preservation, unresolved implementation work and
search limitations. Do not require a new benchmark or raw-data recovery to
close documentary PRs whose content has been preserved.

Coordinator validation: 9 distinct destination file/directory paths exist in
the published results tree. Exact body-delta assertions pass for all three PRs;
PR299's state omitted, no issue closures or review dismissals. git cached diff
whitespace check passed; all changed files are new cleanup notes. Raw discovery
is held outside Git; compact search summary, snapshots and relevant discussions
are committed. Source refs verified with git ls-remote; PR245/314 head metadata
matches the known imported heads. Queries are indexed search and targeted
inspection, not an exhaustive census; zero-match claims are scoped explicitly.
No new tests are needed for remote-documentation payloads. Review verdict is
about these actions, not acceptance of historic scientific/functional results.

Round 2: inspect review-01.json, response-01.md and validation-02.json. F1 adds an
explicit requires_readback dependency on PR314 closed before any PR299 action,
plus README sequencing. F2 changes PR299 wording to "the consolidated results
branch"; historical PR314 proposal remains as permitted under the historical
heading. Check fixes and regressions; record F1/F2 disposition. All external
actions still pending. No destination/closure targets or scientific claims changed.
