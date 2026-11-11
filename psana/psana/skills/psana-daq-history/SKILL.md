---
name: psana-daq-history
description: Persist a psana-daq sweep report, triage its findings into local issue drafts, and file/comment on GitHub issues on slac-lcls/lcls2. Use when a psana-daq sweep has just emitted a report and the human has confirmed they want to hand it off — persisting and triage need no setup at all; GitHub filing uses a GitHub MCP server when one is configured, or hands the human a ready-to-paste draft otherwise. Not for general DAQ diagnosis, which is psana-daq's job, and not invoked mid-sweep.
---

# Skill: psana-daq-history — Persisting, Triaging, and Filing DAQ Diagnostic Findings

You are the **only skill in this suite that writes anything, anywhere.**
`psana-daq` is read-only end to end — it searches, diagnoses, and reports,
then hands its output to you. You take that report (plus the search
results it already gathered) and do everything downstream: save the
report, decide which findings are worth tracking, stage issue drafts, and
— on request — turn them into GitHub issues and comments.

**Related skill:** `psana-daq` owns diagnosis, the sweep, and the search
that finds prior issues before you're ever loaded. Load it first if you
haven't already run a sweep — you have nothing to do without one. Its
"Diagnostic store" section documents the store's *shape* from a reader's
perspective; this skill documents the *write* side.

**Reference material** (cite, don't restate): `reference/
github-mcp-setup.md` covers the one-time MCP setup, the label-API
mechanics, and the Slack-thread rationale. Load it when you need the
"why," not just the policy stated below.

## Read-only with respect to the DAQ only

You never touch the DAQ — no state, no configuration, no processes.
Everything else is fair game, on confirmation:

- **Local files** (`reports/`, `issue-drafts/`) — written on confirmation,
  once, at hand-off. The hand-off itself is the confirmation, not a
  per-file prompt.
- **GitHub** — every write (issue, comment, reopen) is confirmed
  **per-action**, never batched.

Issues and comments filed this way carry the invoking user's own GitHub
identity — there is no shared service account.

## Input contract

Handed, from `psana-daq`'s step-9 hand-off:

- The **full emitted report** (verbatim — all sections).
- The **post-legs search results** (GitHub matches per finding, the
  existing `issue-drafts/` listing).
- The scope that produced it: `hutch`, `platform`, the window.

You don't re-diagnose or re-search — that already happened. Your job
starts at "what should happen to this report."

## Workflow

```
1. PERSIST — save the report to reports/. One confirmation (the hand-off
   itself). Create the file exclusively (see below).
2. TRIAGE — gate 1 (mechanical) then gate 2 (one prompt) to decide which
   findings become new issue drafts, occurrence comments, or recurrence
   comments.
3. STAGE — write any new issue draft.
4. FILE — on request, turn a staged draft or a matched issue into a real
   GitHub write. Only this step benefits from a GitHub MCP server; steps
   1-3 need nothing, and this step still works without one via the
   manual-paste fallback.
```

## Step 1 — Persist the report

Write the handed-off report verbatim to `<store-root>/reports/<hutch>/
<date>-<window>.md`, plus frontmatter:

```yaml
---
hutch: rix
date: 2026-09-23
window: 24h
session: 23_09:57:56   # every session actually examined, comma-separated;
                        # "none" if none ran in the window
components: [alvium_0, teb0, meb0_first]
symptoms: [configure-no-response, disconnect-no-response]
---
```

Filename is date-first (a session prefix is awkward in a filename and
dates collide predictably), but `session` records what was actually
examined — a window may contain zero, one, or several sessions, comma-
separated. Collision on the same hutch/date/window → append `-2`, `-3`,
... — never overwrite. **Written once, never edited afterward.**

State which store root you wrote to (`psana-daq/SKILL.md`'s "Diagnostic
store" section lists the three-tier resolution — cited, not restated).

### Report file creation

Create the file **exclusively** (`O_CREAT|O_EXCL` — fail if it already
exists, rather than silently overwriting). On collision, increment the
suffix and retry the exclusive create, rather than falling back to a
non-exclusive write.

### Coverage: does a prior report cover this session?

A prior report is safe to cite instead of re-investigating a session only
if that **session's last log file's mtime predates the report's own
timestamp** — i.e. the session had already stopped writing by the time it
was reported on. If the session kept writing after the report, treat it
as not fully covered: re-run the legs for it this time, citing the prior
report for context rather than ignoring it. This is `psana-daq`'s own
check before it starts a sweep (see its "Autonomous sweep" step 2) — this
skill doesn't need to re-derive it, just be aware that a report persisted
here is exactly what that future check will read.

## Step 2 — Triage

**Gate 1 — mechanical, no prompt.** Reduce the report's findings to
issue-draft candidates via four structural rules, applied silently:

1. `## Also Possible` entries fold into the `## Hypotheses` section of the
   issue draft for the finding they explain — never their own draft.
2. Exclude absence/idle observations ("DAQ hasn't run in the window").
3. Exclude operator-behavior observations (restart churn, config editing
   mid-session).
4. Exclude negative/ruled-out findings ("links are fine").

**Explicitly not confidence-based** — a `Confidence: low` finding is
exactly the kind of recurring-and-unexplained item this store exists to
keep visible; filtering on confidence would strand it.

**Gate 2 — one prompt.** Present gate-1 survivors as a single numbered
list, each with a suggested filename and the handed-off `issue-drafts/`
listing + GitHub matches, so reuse is visible. One question: *"which of
these should become or update issue drafts? (numbers, or none) — confirm
or replace each suggested name."* Outcome depends on what the search
found for that finding:

| Search found | Outcome |
|---|---|
| No match | Stage a new draft locally — suggested filename, human's name wins |
| Matches an **open** GitHub issue | Occurrence comment (see "Filing" below) |
| Matches a **closed** GitHub issue | Recurrence comment + reopen suggestion |

## Issue drafts (`issue-drafts/<name>.md` — mutable, staging)

```yaml
---
hutch: [xpp]
components: [epix100_0]
symptoms: [register-read-timeout, drp-abort]
first_seen: 2026-09-22
last_seen: 2026-09-29
github_issue: null       # set once filed; file then reduces to a stub
report_refs:
  - reports/xpp/2026-09-22-24h.md
  - reports/xpp/2026-09-29-24h.md
---

## What's happening
Deterministic pyrogue register-read timeout on `epix100_0`, aborting the
DRP every time it is enabled.

## Hypotheses
<Also Possible entries that explain THIS fault fold in here — see gate 1
above. Not for competing faults on a different symptom.>

## What's been tried
- Commented out of the live config. Worked around, not fixed — the
  underlying cause is still unknown.
```

**`issue-drafts/` is staging, not a durable tier, but it is kept, not
dropped** — a finding matching an existing draft condenses in a later
sweep's `## Known Issues` the same way a filed GitHub issue does (see
`psana-daq/SKILL.md`'s "Autonomous sweep" step 6). `issue-drafts/` is the
name, not `cases/` (implies a durable local record this isn't) or
`issues/` (implies the issue already exists). A draft carries
`github_issue: null` until filed; filing reduces it to a stub pointing at
the issue, and occurrences from then on are logged as issue comments, not
appended locally.

**A finding matching an existing draft merges in, never duplicates** —
bump `last_seen`, append to `report_refs`, fold any new detail into the
existing narrative rather than creating a second file for the same fault.

**Naming — suggestion only, no enumerated vocabulary.** Propose
`<component>-<symptom>.md` — component segment copied **verbatim** from
the log/metric label (`epix100_0`, underscore intact), symptom segment
lowercase-hyphenated. The human's typed name is authoritative within the
constraint below. No fixed symptom list is maintained — message/symptom
text drifts between releases faster than a static list can track (see
`psana-daq-logs/SKILL.md`'s "Interpreting `<C>`/`<E>` messages" for why
this suite already rejected a pre-built catalog once). Reuse over
duplication is handled by *displaying* the existing `issue-drafts/`
listing at the point a name is chosen (gate 2), not by matching strings.

**Path safety on the typed name** — a sentence, not a threat model: reject
(ask for a different name) anything containing a path separator, a
leading absolute-path marker, or a `..` segment. Guards against an
accidental paste landing outside `issue-drafts/`, not against a malicious
actor.

**Linkage is one-directional:** issue drafts cite reports; reports never
point back.

## Store location and write mechanics

Same three-tier resolution `psana-daq` searches with (cited from its
"Diagnostic store" section, not restated): `$DAQ_DIAG_REPORT_DIR` →
`/sdf/group/lcls/ds/tools/daq-diagnostics/` (shared default) →
`${XDG_DATA_HOME:-~/.local/share}/daq-diagnostics/` (per-user fallback).
State which one you wrote to.

`/sdf/group/lcls/ds/tools/` is `drwxrwsr-x+ root:ps-pcds` with ACL
`group:ps-data:rwx` — verified group-writable, multi-user in practice.

**Permissions set explicitly, not left to ambient umask/ACL inheritance:**
- New files under a default `umask 0022` land at `644` despite the
  directory's setgid bit (setgid fixes the inherited *group*, not the
  *mode*). **Create every file at `664`.**
- The directory ACL doesn't propagate as a `default:` entry to new files.
  **Create any new hutch/issue-drafts subdirectory at `2775` explicitly.**
- **No git in the store** — a shared-FS git repo throws `fatal: detected
  dubious ownership`. Plain markdown, read and searched with `rg`.
- Concurrent edits to the same draft file have no locking — an accepted
  risk given drafts are short-lived and single-author in practice.
  `reports/` is written once and never edited, so this doesn't apply there.
- Pruning stale entries is a manual, human action.

## Filing

### Repo

**Hardcoded: `slac-lcls/lcls2`.** Public, reachable via `api.github.com`
with no auth for reads. A deliberate choice — no repo-selection logic
needed.

### Setup — per-user, optional

Filing needs a GitHub MCP server with issue read/write access, configured
in the current harness. **Persisting and triage need none of this.**
Without one, every filing action falls back to showing the human a
ready-to-paste draft — nothing else in this skill degrades.

**Check before assuming setup is done**: if no GitHub MCP tool is
available, use the fallback path — don't error out or ask the human to
configure MCP mid-task. Filing without MCP is supported, not degraded.

If MCP isn't configured and the human wants it, point them at `reference/
github-mcp-setup.md` for the one-time setup (opencode, Claude Code, and
Codex covered explicitly). On request, not proactively.

**Tokens are personal — never shared.** Issues/comments filed via MCP
carry the invoking user's own identity, which is the provenance trail this
store exists to build.

### Filing a new issue

1. **Draft the issue body from the issue-draft file** — self-contained.
   **Never cite the local draft's path** — it means nothing to someone
   reading on GitHub.
2. **Suggest labels** (see "Labels" below). Show the suggestion and the
   repo's actual existing labels; the human may add, remove, or replace
   anything.
3. **Confirm** — show the full drafted body and label selection before
   creating anything.
4. **File**: MCP available → `issue_write` (method: `create`). MCP
   unavailable → tell the human plainly, point at `reference/
   github-mcp-setup.md`, show the drafted title/body/labels to paste
   themselves right now.
5. **On success**: write `github_issue: <url>` into the draft's
   frontmatter and reduce it to a stub. If filed manually, ask for the
   resulting URL so this still happens.
6. **Tell the human a Slack thread may now exist** (see "Slack thread"
   below) — GitHub's own app posts it automatically; this skill does
   nothing to cause it.

### Posting an occurrence comment

Draft what was observed this time, when, in which session; confirm; MCP
available → `add_issue_comment`; MCP unavailable → show the drafted text
to paste themselves.

### Posting a recurrence comment + reopen request

Draft a comment stating explicitly this is a **recurrence after a
believed fix** — the single highest-value signal this system produces.
Ask whether to also request reopening. Confirm, then: MCP available →
`add_issue_comment`, then `issue_write` (method: `update`, `state: open`)
if confirmed; MCP unavailable → show the drafted text and tell them to
toggle the issue themselves.

### Labels

No enumerated vocabulary — message/symptom text drifts faster than a
static list tracks.

1. **Query the repo's existing labels** (`GET /repos/slac-lcls/lcls2/
   labels`) and show them — unauthenticated REST, no MCP needed. Re-query
   every time rather than trusting a remembered list.
2. **Suggest**: `daq-diagnostics` (always); `hutch-<name>` per hutch in the
   draft; `component-<subsystem>` — **subsystem-level, never alias-level**
   (`epix100_0`/`epixuhr3x2_1` → `component-epix`; verified: 34 distinct
   DRP aliases at one hutch reduce to ~14 subsystem types — alias-level
   labeling would mean a new label every card racked). Not `bug` — adds no
   filtering power over `daq-diagnostics`, and findings aren't uniformly
   bugs.
3. **The human picks** — final selection is what gets filed.
4. **A chosen label that doesn't exist yet** → ask: create it, or file
   without it? Never create a repo label silently.
5. **On consent, create it explicitly** with a chosen `color`/
   `description` via `POST /repos/slac-lcls/lcls2/labels` **before**
   `issue_write` — see `reference/github-mcp-setup.md` for why ordering
   matters (implicit creation via `issue_write`'s `labels` array defaults
   to flat grey, no description).

## Slack thread

`#lcls-daq-diagnostics` is subscribed to `slac-lcls/lcls2`'s issue
activity — GitHub's native Slack app auto-creates a thread when an issue
is filed. No bot token, no code in this skill. This skill never reads
that thread (see `reference/github-mcp-setup.md`'s rationale). Convention,
not enforced by any tool: if the thread reaches a conclusion, **post it
back to the issue as a comment** — the issue, not the thread, is what a
later sweep's search finds.

## What this skill does not do

Does not diagnose or search on its own — consumes `psana-daq`'s report
and search results, already gathered. Never loads mid-sweep — the
hand-off happens once, after the report, sweep path only. Does not create
GitHub labels without explicit human consent. Does not read Slack, ever.
Does not seed drafts/issues from historical data — the store grows
organically from real sweeps.
