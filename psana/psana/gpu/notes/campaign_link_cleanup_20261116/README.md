# Campaign PR and issue link cleanup

The user requested checking issues and PRs related to the obsolete campaign
branches, correcting campaign-result destinations, and closing superseded work
with the reason that it moved to another branch. The destination is
`slac-lcls/lcls2:results/psana2-gpu-scaling`.

## Discovery and disposition

| Item | Disposition |
| --- | --- |
| [PR245](https://github.com/slac-lcls/lcls2/pull/245), PR187/208 documentary branch | Close without merging as superseded. Add current report/script/history links and a relocation explanation to the body and conversation. |
| [PR314](https://github.com/slac-lcls/lcls2/pull/314), PR299 documentary branch | Close without merging as superseded. Replace four obsolete branch URLs, retain the original proposal under a historical heading, and add a relocation explanation. |
| [PR299](https://github.com/slac-lcls/lcls2/pull/299), implementation | Keep open. Replace its draft-PR archive destination with the current report link and add a conversation update addressing the earlier review reference. Keep all functional review and scientific qualifications. |
| Standalone issues and fork scaling PR | None matched the obsolete campaign scope in the searches; no issue closure planned. Broader implementation issues are not superseded by moving reports. |

The three source heads still match the imported source SHAs. No PR was found
using any of those refs as a base. Head searches found only PR245 and PR314.
Searches covered both repositories, separate PR/issue queries and targeted
comments; the [discovery record](discovery.json) states coverage and limits.
Historic review comments and commit-specific citations remain intact. A current
notice supplies the replacement destination without rewriting earlier findings.

## Execution contract

[actions.json](actions.json) contains exact proposed body/comment payloads and
before snapshots. A focused independent review precedes GitHub mutations. Re-read
bodies and states immediately before each action; stop/reconcile if a body changed.
Check source PR head identity again through a head-filtered query before closure.
Update body before adding the notice and closing. Read back each mutation. On any
retry, inspect existing state/comments first to prevent duplicate notices.

This task does not tag/delete source branches, merge a PR, close implementation
work, resolve reviewer threads, change historical measurements or submit jobs.
Archive tags, final source-head/dependency rechecks and branch retirement remain
a later step. See [status](status.json), [scope](scope.md), and the
[current implementation plan](../performance_campaign_organization_20261116/IMPLEMENTATION.md).
