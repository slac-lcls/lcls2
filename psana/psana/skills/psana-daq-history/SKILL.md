---
name: psana-daq-history
description: Persist a psana-daq sweep report, triage its findings into local case files, and file/comment on GitHub issues on slac-lcls/lcls2. Use when a psana-daq sweep has just emitted a report and the human has confirmed they want to hand it off — persisting and triage need no setup at all; only GitHub filing needs a personal access token, and falls back to a hand-over command if none is configured. Not for general DAQ diagnosis, which is psana-daq's job, and not invoked mid-sweep.
---

# Skill: psana-daq-history

# Persisting, Triaging, and Filing DAQ Diagnostic Findings

You are the **only skill in this suite that writes anything, anywhere.**
`psana-daq` is read-only end to end — it searches, diagnoses, and reports,
then hands its output to you. You take that report (plus the search
results it already gathered) and do everything downstream: save the
report, decide which findings are worth tracking, stage case files, and —
on request — turn them into GitHub issues and comments.

**Related skill:** `psana-daq` owns diagnosis, the sweep, and the search
that finds prior issues before you're ever loaded. Load it first if you
haven't already run a sweep — you have nothing to do without one. Its
"Diagnostic store" section documents the store's *shape* (what the two
file trees look like from a reader's perspective, and why the shared path
was chosen); this skill documents the *write* side — schemas, naming,
permissions, filing — and doesn't restate what it already covers.

---

## Read-only with respect to the DAQ only

You **never touch the DAQ** — no state, no configuration, no processes.
Everything else is fair game, on confirmation:

- **Local files** (`reports/`, `cases/`) — written on confirmation, once,
  at hand-off. Not per-file confirmed individually; the hand-off itself is
  the confirmation.
- **GitHub** — every write (issue, comment, reopen) is confirmed
  **per-action**, never batched. Filing three cases means three separate
  confirmations, not one blanket "yes."

**Issues and comments filed this way carry the invoking user's own
GitHub identity** — there is no shared service account. Expected, not a
gap: since GitHub actions are always per-action confirmed, accountability
for what gets filed is already the confirming human's either way.

---

## Input contract

You are handed, from `psana-daq`'s Step 4 hand-off:

- The **full emitted report** (verbatim — all sections, including `##
  New Crashes` and `## Known Issues`).
- **Step 1's search results**: matched GitHub issues (open/closed) per
  finding, and the existing local `cases/` listing.
- The scope that produced it: `hutch`, live-vs-past, the window.

You do not re-diagnose or re-search anything — that already happened.
Your job starts at "what should happen to this report."

---

## Workflow

```
1. PERSIST — save the report to reports/. One confirmation (the hand-off
   itself); no separate prompt.
2. TRIAGE — gate 1 (mechanical) then gate 2 (one prompt) to decide which
   findings become new case files, occurrence comments, or recurrence
   comments — see "Triage" below.
3. STAGE — write any new case file per "Case files" below.
4. FILE — on request, turn a staged case or a matched issue into a real
   GitHub write — see "Filing" below. This step alone needs a token/MCP;
   steps 1–3 need nothing.
```

---

## Step 1 — Persist the report

Write the handed-off report verbatim to
`<store-root>/reports/<hutch>/<date>-<window>.md`, plus frontmatter:

```yaml
---
hutch: rix
date: 2026-09-23
window: 24h
session: 23_09:57:56       # every session actually examined, comma-separated; null if none ran
components: [alvium_0, teb0, meb0_first]
symptoms: [configure-no-response, disconnect-no-response]
---
```

Filename is date-first, not session-based, because a session prefix
(`DD_HH:MM:SS`) is awkward in a filename and dates collide predictably —
but **the sweep's unit is the session(s) actually examined, not the
window**: `window` records the scope the human asked for, `session`
records what was actually examined, comma-separated if the sweep covered
more than one (a window may contain zero, one, or several sessions, and a
mixed-freshness window may examine only the ones not already covered by a
prior report — see `psana-daq`'s "Raw Findings" → "`Session:`" for the
merge behavior). The `session` field is what a later search should trust
for coverage, not `window`. Collision on the same hutch/date/window →
append `-2`, `-3`, ... — never overwrite. **Written once, never edited
afterward** — this is what lets `reports/` skip the mutability handling
`cases/` needs (below).

**State which store root you wrote to** (see `psana-daq`'s "Store
location" for the three-tier resolution — cited, not restated here).

---

## Step 2 — Triage

**Gate 1 — mechanical, no prompt.** Reduce the report's findings to case
candidates via four structural rules, applied silently:

1. `## Also Possible` entries fold into the `## Hypotheses` section of the
   case file for the finding they explain — never their own case (by
   `psana-daq`'s own discrimination rule, Also Possible is an alternative
   explanation for the *same* symptom, not an independent fault).
2. Exclude absence/idle observations ("DAQ hasn't run in the window").
3. Exclude operator-behavior observations (restart churn, config editing
   mid-session).
4. Exclude negative/ruled-out findings ("links are fine").

**Explicitly not confidence-based** — a `Confidence: low` finding
(e.g. a buffer-occupancy signal flagged as "probably benign but
unconfirmed") is exactly the kind of recurring-and-unexplained item this
store exists to keep visible; filtering on confidence would strand it.

**Gate 2 — one prompt.** Present gate-1 survivors as a single numbered
list, each with a suggested filename (see "Case files" → "Naming" below)
and the handed-off `cases/` listing + GitHub matches, so reuse is visible.
One question: *"which of these should become or update case files?
(numbers, or none) — confirm or replace each suggested name."* Three
outcomes, decided by what the handed-off search results found for that
finding:

| Search found | Outcome |
|---|---|
| No match | Stage a new draft locally — suggested filename, human's name wins |
| Matches an **open** GitHub issue | Occurrence comment (see "Filing" below) |
| Matches a **closed** GitHub issue | Recurrence comment + reopen suggestion |

---

## Case files (`cases/<name>.md` — mutable, staging only)

```yaml
---
hutch: [xpp]                 # list — failure modes aren't hutch-scoped;
                              # an accelerator-side BLD fault would hit
                              # every hutch at once
components: [epix100_0]
symptoms: [register-read-timeout, drp-abort]
first_seen: 2026-09-22
staged_since: 2026-09-26
github_issue: null           # set once filed; file then reduces to a stub
report_refs:
  - reports/xpp/2026-09-22-24h.md
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

**`cases/` is staging, not a tier.** A case file carries `github_issue:
null` until filed, at which point it's reduced to a stub pointing at the
issue — occurrences from then on are logged as issue comments, never
appended locally. An unfiled case older than about a week is surfaced as
such the next time `psana-daq`'s Step 1 searches this store, so aging
drafts stay visible rather than quietly becoming a second permanent store.

**Naming — suggestion only, no enumerated vocabulary.** Propose
`<component>-<symptom>.md` — component segment copied **verbatim** from
the log/metric label (`epix100_0`, underscore intact — the stable,
greppable key), symptom segment lowercase-hyphenated. This is a suggested
shape, not a validated format; **the human's typed name is authoritative.**
No fixed symptom list is maintained, deliberately — see "Interpreting
`<C>`/`<E>` messages" in `psana-daq-logs/SKILL.md` for why this suite
already rejected a pre-built catalog once (message text drifts between
releases faster than a static list can track). Reuse over duplication is
handled by *displaying* the existing `cases/` listing at the point a name
is chosen (gate 2 above), not by matching strings.

**Linkage is one-directional:** cases cite reports; reports never point
back — this is what lets `reports/` skip the mutability handling below.

---

## Store location and write mechanics

Resolved in the same three-tier order `psana-daq` searches (cited from
its "Store location" section, not restated): `$DAQ_DIAG_REPORT_DIR` →
`/sdf/group/lcls/ds/tools/daq-diagnostics/` (shared default) →
`${XDG_DATA_HOME:-~/.local/share}/daq-diagnostics/` (per-user fallback).
**State which one you wrote to.**

`/sdf/group/lcls/ds/tools/` is `drwxrwsr-x+ root:ps-pcds` with ACL
`group:ps-data:rwx` — verified group-writable and already multi-user in
practice.

**Permissions must be set explicitly, not left to ambient umask or ACL
inheritance:**
- New files under a default `umask 0022` land at `644` (author-writable
  only) despite the directory's setgid bit — setgid fixes the *group* a
  new file inherits, not its *mode*. **Create every file at `664`.**
- The directory ACL does not propagate as a `default:` entry to new
  files. **Create any new hutch/case subdirectory at `2775` explicitly.**
- **No git in the store** — a shared-FS git repo throws `fatal: detected
  dubious ownership`, the same defect class fixed elsewhere in this
  release for DAQ release trees. Everything here is plain markdown, read
  and searched with `rg`, never git.
- Concurrent edits to the same case file on wekafs have no locking. Given
  staged files are short-lived, single-author drafts rather than a
  long-lived accumulating corpus, this is an accepted risk, not solved
  here. `reports/` is written once and never edited, so this risk doesn't
  apply to it at all.
- Pruning stale entries is a manual, human action — not something this
  skill performs autonomously.

---

## Filing

### Repo

**Hardcoded: `slac-lcls/lcls2`.** Public, reachable via `api.github.com`
with no auth for reads. A deliberate choice, settled directly rather than
left open — no repo-selection logic needed here.

### Setup — per-user, one-time

Filing needs a GitHub personal access token. **Persisting and triage
above need none of this** — only the steps below do. Without a token,
every filing action falls back to a hand-over command; the skill still
does everything up to that point unaided.

**Tokens are personal — never shared between people.** Issues and
comments carry the invoking user's own GitHub identity (see "Read-only
with respect to the DAQ only" above), which is the provenance trail this
store exists to build; a shared token collapses every filing to one
name. If a shared identity is ever genuinely wanted, the supported
pattern is a dedicated machine account owning its own token, not a
circulated personal one.

1. **Create a fine-grained PAT** at
   https://github.com/settings/personal-access-tokens/new:
   - **Set "Resource owner" to the target org (e.g. `slac-lcls`) before
     anything else.** It defaults to your own account, and while it's
     there the repository picker lists only repos you personally own —
     the target repo never appears, making it look like you lack access.
     This is the single most common setup failure.
   - Repository access → **Only select repositories** → the target repo
     (e.g. `slac-lcls/lcls2`).
   - Permissions → Repository permissions → **Issues: Read and write**.
     Leave everything else at *No access*. `Metadata: Read-only` is added
     automatically and is mandatory.

   **If the org isn't offered as a resource owner**, it hasn't enabled
   fine-grained tokens; an org owner can allow this under Organization
   Settings → Third-party Access → Personal access tokens. A classic
   token works as a fallback, but note it has **no issues-only scope** —
   `public_repo` is the narrowest that can file an issue, and it also
   grants code-push on every public repo you can write to. Prefer
   fine-grained. Don't use `repo`: it adds private-repo access this skill
   never needs.

2. **Store it where only you can read it:**

   ```bash
   printf '%s\n' '<token>' > ~/.config/opencode/github_token
   chmod 600 ~/.config/opencode/github_token
   ```

   Mode `600` matters — this token can write to a shared repo. Avoid
   trailing whitespace: the file's contents are substituted literally
   into an `Authorization` header, so a stray character makes it
   malformed.

3. **Add a `github` entry to your own
   `~/.config/opencode/opencode.jsonc`** (per-user — not centrally
   provisioned). It goes *alongside* any MCP servers you already have,
   inside the existing `"mcp"` block — don't replace the block, and
   remember the comma after the preceding entry:

   ```jsonc
   "mcp": {
     // ...existing entries, each followed by a comma...
     "github": {
       "type": "remote",
       "url": "https://api.githubcopilot.com/mcp/x/issues",
       "enabled": true,
       "headers": {
         "Authorization": "Bearer {file:~/.config/opencode/github_token}"
       }
     }
   }
   ```

   The `/x/issues` suffix loads **only** the issues toolset. Plain
   `/mcp/` also works but additionally loads repos, PRs, actions, and
   code scanning — context cost on every session for capability this
   skill never uses. (`verified-live`, 2026-09-29: with `/x/issues` the
   session exposed issue, label, and sub-issue tools and no repo, PR, or
   actions tools, and successfully authenticated a real filing against
   `slac-lcls/lcls2` — see "Filing a new issue" below.)

4. **Restart the opencode session.** MCP servers connect at startup, so a
   running session won't pick up the new entry.

**Verifying, without writing anything:** request a label on the repo
(e.g. `get_label` on `bug`, which exists by default on every repo). This
exercises the token end to end and creates nothing. Note that seeing
`github_*` tools in a session proves only that the *server* connected —
the tool manifest loads before any authenticated call, so a bad token
still shows tools and fails only at call time.

If a known-good token is rejected by `api.githubcopilot.com`, an active
Copilot entitlement may be required — take the fallback path below rather
than debugging the token.

**Not wired up for AMI-spawned agents today.** AMI's `mcp_server.py`
registers only its own `ami` MCP server for agents it spawns — a personal
`github` MCP entry in your CLI's `opencode.jsonc` does not extend to a
session spawned from inside AMI. The hand-over-command fallback below
works in both contexts; this is a known gap for agent-side filing
specifically from within AMI, not a blocker.

**Check before assuming setup is done:** if no `github` MCP tool is
available to you, use the fallback path — do not error out or ask the
human to configure MCP mid-task. Filing without MCP is a supported, not
degraded, path.

### Filing a new issue

1. **Draft the issue body from the case file** — title, description,
   evidence — self-contained. **Never cite the local case-file path**; it
   resolves differently per user and per fallback tier, so it means
   nothing to someone reading the issue on GitHub.
2. **Suggest labels** — see "Labels" below. Show the human the suggestion
   and the repo's actual existing labels; let them add, remove, or replace
   anything before filing.
3. **Confirm with the human** — show the full drafted body and label
   selection before creating anything.
4. **File:**
   - **MCP available:** call `issue_write` (method: `create`) with the
     confirmed body/labels.
   - **MCP unavailable:** display the drafted body and hand over a
     ready-to-run `curl` for the human to execute themselves:
     ```bash
     curl -X POST https://api.github.com/repos/slac-lcls/lcls2/issues \
       -H "Authorization: Bearer $(cat ~/.config/opencode/github_token)" \
       -H "Accept: application/vnd.github+json" \
       -d '{"title":"<title>","body":"<body>","labels":["daq-diagnostics","hutch-mfx","component-epix"]}'
     ```
5. **On success:** write `github_issue: <url>` back into the local case
   file's frontmatter and reduce the file to a stub (a one-line pointer at
   the issue — no longer a draft to edit locally).
6. **Tell the human that a Slack thread may now exist.** `#lcls-daq-diagnostics`
   is subscribed to this repo's issue activity (see "Slack thread" below)
   — GitHub's own app will have posted a thread-starting message for the
   new issue automatically there. You did nothing to cause this and
   cannot control it, but the human should know discussion may now be
   happening there too.

### Posting an occurrence comment

1. Draft a short comment: what was observed this time, when, in which
   session — enough for someone skimming the issue's history to see this
   is the Nth occurrence.
2. Confirm with the human.
3. **MCP available:** `add_issue_comment`. **MCP unavailable:** hand over
   a ready-to-run `curl`:
   ```bash
   curl -X POST https://api.github.com/repos/slac-lcls/lcls2/issues/<N>/comments \
     -H "Authorization: Bearer $(cat ~/.config/opencode/github_token)" \
     -H "Accept: application/vnd.github+json" \
     -d '{"body":"..."}'
   ```

### Posting a recurrence comment + reopen request

1. Draft a comment stating explicitly that this is a **recurrence after a
   believed fix** — this is the single highest-value signal this system
   produces (a fix that didn't hold), so don't understate it.
2. Ask the human whether to also request reopening.
3. Confirm, then: **MCP available:** `add_issue_comment`, then
   `issue_write` (method: `update`, `state: open`) if the human confirmed
   the reopen. **MCP unavailable:** hand over the comment `curl` above,
   plus:
   ```bash
   curl -X PATCH https://api.github.com/repos/slac-lcls/lcls2/issues/<N> \
     -H "Authorization: Bearer $(cat ~/.config/opencode/github_token)" \
     -H "Accept: application/vnd.github+json" \
     -d '{"state":"open"}'
   ```

### Labels

**No enumerated vocabulary is shipped here** — case-file naming already
rejected a closed list for the same reason (message/symptom text drifts
faster than a static list tracks), and the same argument applies to
labels for symptoms and detector aliases.

**What to do instead, every time you file:**

1. **Query the repo's existing labels** (`GET
   /repos/slac-lcls/lcls2/labels`) and show them to the human. This is
   plain unauthenticated REST — no MCP tool exposes label listing, but
   none is needed; this works even before any token is configured. As of
   2026-09-26 these were the 7 GitHub defaults (`bug`, `duplicate`,
   `enhancement`, `help wanted`, `invalid`, `question`, `wontfix`) with no
   custom labels and no labeled issues. As of the first real filing
   (2026-09-29, issue #134), `daq-diagnostics`, `hutch-mfx`, and
   `component-epix` now also exist — re-query rather than trusting either
   snapshot; this list only grows as real findings get filed.
2. **Suggest**, derived from the case file's frontmatter:
   - `daq-diagnostics` — always suggested. This is the one label that
     matters for finding these issues again later; everything else below
     is a nice-to-have on top of it.
   - `hutch-<name>` for each hutch in the case file's `hutch` list.
   - `component-<subsystem>` — **subsystem-level, never alias-level.**
     `epix100_0` → `component-epix`; `epixuhr3x2_1` → `component-epix`;
     `hsd_2` → `component-hsd`. Verified against a real corpus: 34
     distinct DRP aliases at xpp alone reduce to ~14 subsystem types.
     Labeling at alias granularity means a new label every time a new
     card is racked; labeling at subsystem granularity does not.
   - Do **not** suggest `bug` — it adds no filtering power over
     `daq-diagnostics` in a repo where nothing is labeled yet, and this
     system's findings aren't uniformly "bugs" (some are hardware faults,
     some are operator-sequencing issues, some are unconfirmed anomalies).
3. **The human picks.** Show the suggestions and the existing label list
   together; let them accept, remove, add, or replace anything. Their
   final selection is what gets filed — you do not enforce your own
   suggestions.
4. **If a chosen label doesn't exist in the repo yet**, say so explicitly
   and ask: create it, or file without it? **Never create a repo label
   silently** — that's a shared-repo configuration change, not a
   per-issue action, and it should be a visible decision. (`verified-live`,
   2026-09-29: the skill surfaced this exact prompt for `daq-diagnostics`,
   `hutch-mfx`, and `component-epix` on issue #134's filing, and the human
   confirmed before any label was created.)
5. **On consent, create the label explicitly** with a chosen `color` and
   `description` via `POST /repos/slac-lcls/lcls2/labels` (needs only the
   Issues:Write permission the token already has for filing) — **do not**
   rely on the label appearing implicitly in `issue_write`'s `labels`
   array to create it. Passing a nonexistent name there does create the
   label as a side effect, but with no color/description chosen (GitHub
   defaults it to flat grey, `#ededed`) — verified on all three labels
   created for issue #134. An explicit `POST` first gives the label a real
   color and description, so it reads as a deliberate part of the
   vocabulary rather than an accident of the API, before `issue_write`
   ever references it.

**`slac-lcls` also defines org-level custom issue fields** (`Priority`,
`Start date`, `Target date`, `Effort` — all `ORG_ONLY` visibility, as of
2026-09-29). This skill does not set any of them; labels remain the only
metadata it writes.

---

## Slack thread

`#lcls-daq-diagnostics` is subscribed to `slac-lcls/lcls2`'s issue
activity (one-time, human-performed setup, done 2026-09-29) — **GitHub's
native Slack app auto-creates a thread the moment an issue is filed.**
No bot token, no `chat:write` scope, and no code in this skill are
involved — it's entirely GitHub's own integration, external to anything
this suite does. (`verified-live`, 2026-09-29: filing issue #134 created
a thread in that channel with no action from this skill beyond the
filing itself.)

**This skill never reads that thread.** Agent-side Slack search was
considered and rejected during planning: Slack's `search.messages` API
requires a user token (impersonating a specific human, with access to
whatever that human can see, including private channels), not a bot
token — a materially larger trust decision than this suite makes anywhere
else, for a capability (reading back informal chat) that's reachable more
simply by searching the issue itself.

**The mitigation, stated as a convention, not enforced by any tool:** if
discussion in the thread reaches a conclusion — a cause identified, a fix
that worked, a hypothesis ruled out — **post it back to the issue as a
comment.** The issue, not the thread, is what a later `psana-daq` sweep's
search actually finds. A conclusion that lives only in Slack is invisible
to every future sweep.

---

## What this skill does not do

- Does not diagnose anything, and does not search on its own — it
  consumes `psana-daq`'s sweep report and Step 1's search results, both
  already gathered by the time it's loaded.
- Never loads mid-sweep. `psana-daq`'s hand-off happens once, after the
  report is emitted, on the sweep path only — never on targeted dispatch.
- Does not create GitHub labels without an explicit human decision to do
  so.
- Does not read Slack, ever, regardless of MCP/token availability.
- Does not seed case files or issues from historical data (`daq_logs.db`
  or otherwise) — the store grows organically from real sweeps, on
  purpose.
