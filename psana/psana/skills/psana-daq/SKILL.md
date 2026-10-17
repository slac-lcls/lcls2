---
name: psana-daq
description: Entry point for diagnosing LCLS-II DAQ problems end-to-end. Use when the user reports a general DAQ issue without a specific angle yet — e.g. "the DAQ is broken", "we're losing data", "high deadtime", "damage events", "DAQ won't start", "what happened in last night's run", "give me a full assessment" — and you need to figure out which diagnostic angle (state/control, metrics, logs, or configuration) to pursue, or run a full autonomous sweep across all of them. Routes to psana-daq-control (run-control state machine and transition failures), psana-daq-monitor (Grafana/Prometheus metrics), psana-daq-logs (raw DAQ log files), and psana-configdb (read-only detector/system configuration lookups). For a vague report with no specific angle named, runs an autonomous sweep across all applicable angles, searching prior GitHub issues and a local diagnostic store after the legs run, and returns one ranked report, offering to persist/stage/file new findings afterward.
---

# Skill: psana-daq — LCLS-II DAQ Diagnosis Entry Point

Entry point for diagnosing LCLS-II DAQ problems. Establishes the target
(hutch, platform, and a time window) once, then either **dispatches** to a
sibling skill for a targeted question, or runs an **autonomous sweep**
across every applicable angle for a vague report and returns one ranked
report.

**Every question is about a hutch and a time window** — there is no
separate "live" code path. "Live" is simply a window ending now, scoped to
the newest launch. Every angle answers from retained evidence (log files,
Prometheus history, ConfigDB history, GitHub issues); none requires a
reachable DAQ control process. See "Establish the target" below.

**Reference material** (cite, don't restate):
- `reference/report-template.md` — the full sweep report template and
  field-level rules.
- `reference/release-source.md` — locating the exact source tree a running
  DAQ was built from.

## Read-only posture

Read-only everywhere, no exceptions — never mutates DAQ state, never writes
a local file, never calls a GitHub write endpoint. **Every write in this
suite lives in `psana-daq-history`** — reports, issue drafts, GitHub
issues/comments, all of it. This skill only ever hands that skill a report
to persist and triage.

## How this works

Four sibling skills, one angle each:

- **`psana-daq-control`** — run-control state machine and transitions,
  read from `control.log` (DAQ won't start, stuck in a state,
  component-didn't-respond errors, readout-group questions).
- **`psana-daq-monitor`** — Grafana/Prometheus metrics (event rate,
  deadtime, damage, buffers, event builder, MEB).
- **`psana-daq-logs`** — raw DAQ log files on disk (error/critical
  messages, process command lines, crash context).
- **`psana-configdb`** — read-only ConfigDB lookups (detector/trigger/
  timing configuration, alias/device history).

None load automatically — load the relevant one by name before attempting
any of its tool calls. Do not guess at state queries, Grafana queries, log
paths, or ConfigDB URLs from this router's summary alone.

## Establish the target

Before routing or sweeping, establish `hutch`, `platform`, and the window.

### Config discovery and identity

List `~<hutch>opr/daq/scripts/*.py`. Two tiers: leaf endstation configs
(e.g. `3rix.py` — `from <base> import*` + `config.select([...])`, what
operators launch) and base configs (e.g. `rix.py` — `procmgr_config =`,
where `platform`/`collect_host`/`hutch` live). Ask the user which config
they're running — it's a launch argument, not discoverable from a resident
process.

Read the base config **textually, never import it** — a syntax error in a
real config would otherwise crash the agent; report it as a finding
instead. Extract near the top:

```
platform = '0'
collect_host = 'drp-srcf-mon002'
hutch, station, user = ('rix', 2, 'rixopr')   # rix/mfx — station present
hutch, user = ('xpp', 'xppopr')               # xpp/tmo — no station field
```

When `station` is absent, `station = platform` (matches `control.py:808`).
Ignore `p*.cnf.last` files in the same directory — stale `procmgr`
leftovers, not current state.

**`-C` is a homograph**: `-C COLLECT_HOST` (e.g. the DRP/TEB/MEB command
line's `std_opts = f'-P {hutch} -C {collect_host} -M {prom_dir}'` in
`psdaq/psdaq/cnf/rix.cnf:35`) is the collection host; `-C CONFIG_ALIAS`
(`control.py:2608`) is the ConfigDB alias. Don't copy a log's `CMDLINE:
-C BEAM` as the collect host — that's the alias (see "Configuration
alias" below for where it *is* wanted). Use the config's `collect_host`
field, or `control.log`'s `# HOST:` line.

### Configuration alias

Every ConfigDB endpoint is scoped by alias (`<hutch>/<alias>/<device>`), and
a hutch commonly has more than one — `psana-configdb` must never guess by
taking the first `get_aliases` result. Establish it alongside `hutch`/
`platform` whenever a ConfigDB handoff is possible, via this fallback
chain, taking the **last** match at or before the window's end time:

1. `control.log`: the last `Loaded configDb <hutch>/<alias>/<trigger>_0`
   line, inside `condition_configure` — this is the alias actually
   applied at that Configure. Logged at INFO as of a recent `control.py`
   change (`condition_configure()`); only present in logs from releases
   that include that change, absent in older/already-captured logs.
2. `teb0.log`: the last `Trigger loaded from <lib> using configDb
   <hutch>/<alias>/<trigger>_0` line (already logged at INFO today,
   `teb.cc:1050`; verified-against-real-logs today in real rix, mfx, and
   tmo logs). This line is only written when control is run with `-t
   trigger`, which is true of the production `rix`/`tmo`/`mfx`/`txi`/
   `ued` configs but should not be assumed universal.
3. `control.log`'s `# CMDLINE:` header, the `-C <alias>` flag — labelled
   explicitly as the alias **at launch time only**, since an alias change
   made via the GUI or a scan script before the above `control.py` change
   was logged only at debug level and may not be reflected here.
4. If none of the above are found, ask the user which alias was active.

If a `config alias changed: <old> -> <new>` line is present (only in logs
from releases including the above `control.py` change), use it to
establish when a switch happened, corroborating steps 1–3.

### The window

Establish `[start, end]` in UTC. Ask once, alongside `hutch`:

> **"What time range? (e.g. 'right now', 'yesterday', 'last 2 hours', an
> explicit date/range)"**

Accept free-form input without validating or parsing it here — that's
`psana-daq-logs`' "Resolving a vague date" job. Infer instead of asking
when the report's own phrasing already settles it ("what's the deadtime
right now?" → `end = now`; "what happened last night?" → last night).
"Right now" / "live" / "current" all mean the same thing: a window whose
`end` is "now," scoped to the newest launch — not a different mechanism
from a past window, just a different value for `end`.

**One exception, sweep only:** if the resolved window's intersecting
session count is unexpectedly large, ask once, before any leg starts,
whether to narrow it — never mid-sweep.

### Handing off to leaf skills

State `hutch`, `platform`, and the resolved window explicitly when invoking
a leaf — e.g. "Loading `psana-daq-monitor` for hutch=rix, platform=2,
window=2026-09-17T20:00Z..2026-09-18T08:00Z" — rather than letting it
re-derive or re-ask for any of them. Hand off the configuration alias too
when established.

`hutch` ≡ Prometheus's `instrument` label, and `platform` ≡ its
`partition` label — but both equivalences are more specific than that,
and the `partition` label means something different on XPM metrics than
on DAQ-process metrics. See `psana-daq-monitor/SKILL.md`'s "Establish
instrument, partition, and window" section for the full table (DAQ-
process metrics, XPM per-group metrics, XPM link metrics, `drp_deadtime`)
rather than restating it here. Hand off `hutch` and `platform` as
established; `psana-daq-monitor` applies the right filter per metric
family itself.

### Which config is running

Not directly discoverable at runtime. Corroborating signals only, not a
decision rule: whether the config's selected process set matches
`control.log`'s rollcall selected/NOT-selected lines (same citations as
`psana-daq-control/SKILL.md`'s rollcall interpretation); the config
file's own content. If uncertain, ask — the user knows which config they
launched.

## Dispatch vs. sweep

- **Specific angle named** → use the routing table below, start with that
  one leaf. Cheap, fast path — don't sweep for "what's the deadtime right
  now?"
- **Vague report** → run the autonomous sweep across all four angles.

**"Start with one" is not "stop at one."** A targeted dispatch may hand off
to a *second* leaf when the first leaf's own result calls for it (a metrics
spike needing a log-grep window; a log finding needing a config-history
check) — decided by what's found, not a fixed two-skill sequence.

## Symptom → skill routing table

| Symptom / question | Load skill |
|---|---|
| DAQ won't start / stuck in a state / failed transition / "X did not respond to \<transition\>" | `psana-daq-control` |
| "Why isn't detector X being recorded" / readout group questions | `psana-daq-control` |
| deadtime, damage %, event rates, buffer occupancy, MEB/TEB timing | `psana-daq-monitor` |
| "what actually failed", error messages, crash traces, "why did component X die" | `psana-daq-logs` |
| "was detector X misconfigured", trigger/timing config, "did a parameter change mid-run" | `psana-configdb` |
| correlate a metrics spike with a failing component | `psana-daq-monitor`, handing off to `psana-daq-logs` if the result pinpoints a window/component |
| "is this new behavior, or did a config change cause it" | `psana-daq-monitor` or `psana-daq-logs`, handing off to `psana-configdb` if a specific device/alias is implicated |
| vague report, no other detail | run the **autonomous sweep** below |

## Autonomous sweep (vague reports)

One ranked diagnostic report for the established window, by running every
applicable leg and the store search. This is not a new diagnostic
technique — it's the four leaf skills' existing methods, cited and invoked.

**The sweep has exactly one deliverable: the report in `reference/
report-template.md`, emitted as the mandatory last step.** Narrating the
legs is preparation for it, not the deliverable — do not stop at prose.

Steps:

1. **Pin the exact session(s)** the window intersects (`psana-daq-logs`'
   session-selection). A window may span more than one launch; investigate
   every intersecting session without per-session prompts (the one
   exception above).
2. **Check local `reports/` for coverage** — before any leg runs. A prior
   report covers a session only if that session's **last log file's mtime
   predates the report's own timestamp**; otherwise treat it as not yet
   covered and re-run legs for it (citing the prior report for context).
3. **Run each uncovered leg**, in order — logs, control, metrics, configdb
   — narrating briefly before each. Fixed order, not branch-on-state: logs
   are cheapest to check and most often name what the other three need to
   look for.
   - **Logs**: `psana-daq-logs`' canonical read + `<C>`/`<E>` interpretation.
   - **Control**: `psana-daq-control`'s state-machine reconstruction from
     `control.log` (this is now the skill's normal mode, not a degraded
     past-session fallback — there is no live/past split left to degrade
     from).
   - **Metrics**: queries cover the full `[start, end]` window per
     `psana-daq-monitor`'s query-shape rules (range, plus whole-window
     `increase()` totals). Cite its diagnosis-order table, don't re-paste
     PromQL.
   - **ConfigDB**: gated by the implicated-detector rule — only query for a
     detector named in a control/log error, a `<C>`/`<E>` line, or the
     `detname` label on a breached metric. No detector implicated →
     `skipped — no detector implicated`.
4. **Derive search terms from what the legs found** — component aliases,
   stable error-string fragments, exception types, transition names,
   failure categories. Not from the user's original vague wording.
5. **One batched GitHub + `issue-drafts/` search**, after every leg has
   run: `is:issue label:daq-diagnostics` against `slac-lcls/lcls2`
   (unauthenticated REST — `is:issue` is required, `search/issues` returns
   PRs too) plus an `rg` of local `issue-drafts/`. One query per sweep, not
   one per finding (10 req/hour unauthenticated).
6. **Match and condense.** A matched finding renders as a one-line pointer
   plus any new occurrence/session delta under `## Known Issues` — never
   dual-listed under `## New Crashes`, never re-derived in full elsewhere
   in the report. A match against an unfiled draft keeps its real evidence
   tier (`verified-against-real-logs`, typically); `documented-in-issue` is
   for filed GitHub issues only.
7. **Account for every leg** — every `## Raw Findings` line is a finding or
   an explicit `skipped — <reason>`; a leg pruned by the ConfigDB gate or
   already covered by a prior report must say so, never simply be absent.
8. **Emit the report** per `reference/report-template.md`, verbatim in
   shape, as the final action. Mandatory — a vague report that ends in
   unstructured prose has not completed the sweep.
9. **Offer the hand-off to `psana-daq-history`** — once, after the report
   is shown, never after targeted dispatch. One combined offer (persist +
   triage + stage, not separate prompts). Declining leaves nothing written.

### Crash definition

Python exception (`Traceback (most recent call last)`, a terminating
`<Type>Error:` line) or segfault (`Segmentation fault`, `(core dumped)`).
Full corpus derivation and rejected signatures are in `psana-daq-logs/
reference/crash-signatures.md` — cite it, don't restate the derivation
here.

### What the sweep does not do

No writes of any kind — issue search is unauthenticated REST; persisting,
staging, and every GitHub write are `psana-daq-history`'s job. No store
offer after targeted dispatch (prose, not a structured report; usually
nothing worth tracking).

## Diagnostic store

Read-only from here — `psana-daq-history` owns writing both trees.

```
reports/<hutch>/<date>-<window>.md   immutable — one file per sweep
issue-drafts/<name>.md               mutable staging — a draft issue body
GitHub issue                         the durable, collaborative record
```

Location: `$DAQ_DIAG_REPORT_DIR` → `/sdf/group/lcls/ds/tools/daq-diagnostics/`
(default, shared) → `${XDG_DATA_HOME:-~/.local/share}/daq-diagnostics/`
(per-user fallback). State which root was searched.

A finding matching a draft in `issue-drafts/` condenses the same way a
filed GitHub issue does — staging is not a "re-derive every sweep" tier;
see `psana-daq-history/SKILL.md` for why. The two search steps above (local
coverage before legs, GitHub+drafts after) must stay in this skill — if
condensing logic ever moved to `psana-daq-history`, it would break for
anyone who declines or never sees the hand-off offer.

## Prerequisites / preflight

Check before routing or sweeping; tell the user upfront if an angle is
unavailable rather than discovering it mid-sweep:

- **Control**: no external dependency — it's a plain file (`control.log`),
  with no reachability or `PATH` requirement at all.
- **Logs**: a readable log directory (`psana-daq-logs`' path convention).
  `test -d` before assuming it exists.
- **Metrics**: `grafana_*` MCP tools present in the current session.
- **ConfigDB**: `curl -sf -o /dev/null -w '%{http_code}'
  https://pswww.slac.stanford.edu/ws/configdb/ws/configDB/get_hutches/` →
  `200`.

## AMI / DAQ ownership seam

**AMI is a shared-memory client of the MEB** — true regardless of whether
this agent happens to be running inside an AMI-spawned session.
`ami-performance-monitor` owns AMI-side symptoms (graph latency, worker
starvation, GUI lag); this suite owns the DAQ side. The MEB is the
boundary: if AMI isn't updating, check whether the MEB is delivering events
(this suite) before blaming AMI's graph processing.

## What this skill does not do

No PromQL, log grep patterns, or ConfigDB endpoints of its own — those stay
in the leaf skills so they're authoritative in one place. Load the
relevant sibling before doing any real diagnostic work. No "questions for
reviewers" section — that lives in the plan directory, not shipped text.
