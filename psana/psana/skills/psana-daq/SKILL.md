---
name: psana-daq
description: Entry point for diagnosing LCLS-II DAQ problems end-to-end. Use when the user reports a general DAQ issue without a specific angle yet — e.g. "the DAQ is broken", "we're losing data", "high deadtime", "damage events", "DAQ won't start", "what happened in last night's run", "give me a full assessment" — and you need to figure out which diagnostic angle (state/control, metrics, logs, or configuration) to pursue, or run a full autonomous sweep across all of them. Routes to psana-daq-control (run-control state machine and transition failures), psana-daq-monitor (Grafana/Prometheus metrics), psana-daq-logs (live raw DAQ log files), and psana-configdb (read-only detector/system configuration lookups). For a vague report with no specific angle named, runs an autonomous sweep across all applicable angles and returns one ranked report.
---

# Skill: psana-daq

# LCLS-II DAQ Diagnosis Entry Point

You are the entry point for diagnosing LCLS-II DAQ (Data Acquisition) problems.
This skill establishes the target (hutch, and live vs. a past session) once,
then either **dispatches** to a single sibling skill for a targeted question,
or runs an **autonomous sweep** across every applicable angle for a vague
report and returns one ranked report. This skill itself queries nothing for
the dispatch path — those tool calls live in the sibling skills. For the
sweep path, it composes the sibling skills' existing methods, cited and
invoked rather than duplicated.

## Read-only posture

This skill is read-only, no exceptions. Gather evidence and recommend; the
**human executes every remediation.** See `psana-daq-control/SKILL.md`'s
**"READ-ONLY POSTURE — NON-NEGOTIABLE"** section for the canonical
may-run/must-not-run table — do not restate it here.

## How this works

There are four sibling diagnostic skills, each covering one angle:

- **`psana-daq-control`** — run-control state machine and transition failures
  (DAQ won't start, stuck in a state, component-didn't-respond errors,
  "why isn't detector X being recorded", readout-group questions). Works with
  no Grafana, no MCP, no ConfigDB reachability — talks directly to the
  `control` process over ZMQ.
- **`psana-daq-monitor`** — Grafana/Prometheus metrics for the running DAQ
  (event rates, deadtime, damage, buffers, event builder, MEB monitoring).
- **`psana-daq-logs`** — raw DAQ log files on disk for the current/recent
  session (error/critical messages, process command lines, crash context).
- **`psana-configdb`** — read-only lookups against the ConfigDB web service
  (detector/trigger/timing configuration, alias/device history).

None of these load automatically. Each is discovered but dormant until you
call the `skill` tool with its exact name. When you decide which angle(s) are
relevant, explicitly invoke `skill(name="psana-daq-control")` (or
`psana-daq-monitor` / `psana-daq-logs` / `psana-configdb`) before attempting
any of that skill's tool calls — do not guess at state queries, Grafana queries,
log paths, or ConfigDB URLs from this router's summary alone.

## Establish the target

Before routing or sweeping, establish **which DAQ/configuration** you are
looking at, and **which session** (live or a specific past one).

### Config discovery

The configuration files live at `~<hutch>opr/daq/scripts/*.py`. List them:

    ls ~<hutch>opr/daq/scripts/*.py

Two tiers exist:

- **Leaf endstation configs** (e.g. `3rix.py`, `crix.py`) — contain `from <base> import*` and
  a `config.select([...])` call. These are what operators launch.
- **Base configs** (e.g. `rix.py`) — define `procmgr_config =` and hold the identity
  assignments at the top. These are what you read for `platform`/`collect_host`/`hutch`.

Ask the user which config they are running (or list them and ask). The user knows; the
agent cannot determine it from a resident process because `daqmgr` takes the config as a
launch argument but is not a persistent resident process. The `p*.cnf.last` files in the
same directory are stale 2023/2024 `procmgr` leftovers — ignore them.

### Identity extraction

Read the **base config textually** (never import it — as of 2026-09-18, `3rix.py` had a
syntax error on line 44 (unclosed paren), and importing it would crash the agent). Extract
these assignments near the top of the base config:

```
platform = '0'
collect_host = 'drp-srcf-mon002'
hutch, station, user = ('rix', 2, 'rixopr')   # rix/mfx variant — includes station
hutch, user = ('xpp', 'xppopr')                # xpp/tmo variant — no station field
```

Two schema variants exist:
- `hutch, station, user = (...)` — rix and mfx; station present
- `hutch, user = (...)` — xpp and tmo; no station field

When `station` is absent, use `station = platform` (matches `control.py:808`).

If the config file has a syntax error (e.g. unclosed paren), report it as a finding
rather than crashing the agent.

### Derived values

With `hutch`, `platform`, and `collect_host` from the config:

- **`daqstate`**: `daqstate -P <hutch> -p <platform> -C <collect_host>`
  Use the **bare `hutch`** (e.g. `rix`, not `rix:2`). The log CMDLINE may show `-P rix:2`
  but `daqstate -P rix:2` exits with `Error: instrument name 'rix:2' does not match 'rix'`
  — `control.py:801` (`handle_getinstrument` at `:1688` returns the bare instrument). The
  config's `hutch` field is already stripped.
- **`-C` is a homograph**: `-C COLLECT_HOST` in `daqstate.py:18` means the collection
  host. `-C CONFIG_ALIAS` in `control.py:2608` means something entirely different. Do NOT
  copy `-C BEAM` from the control CMDLINE in the log — that is the config alias, not the
  collect host. Use the config's `collect_host` field. The collect host also appears as
  `# HOST:` in `control.log`.

### Scope: live or a past session

Ask, once, alongside (or immediately after) establishing `hutch`:

> **"Live/current, or a specific past date/time range to look at?"**

Free-form — accept whatever the user gives back (a date, "yesterday", a
session prefix, an explicit start/end) without validating or parsing it
here; that is each sibling skill's job. Infer instead of asking when the
report's own phrasing already settles it (e.g. "what's the deadtime right
now?" is live; "what happened in last night's run?" is past) — only ask when
genuinely unclear.

If the user gives something vaguer than an exact session, the receiving
skill still does the final pinning — e.g. `psana-daq-logs`' own
**"Session selection — present a list and ask"** mechanism narrows "last
night" to an exact `DD_HH:MM:SS` prefix. This section does not replace that;
it just avoids asking the live/past question a second time downstream.

### Handing off to leaf skills

Once `hutch` and scope are established here, **state both explicitly when
invoking a leaf skill** — e.g. "Loading psana-daq-monitor for hutch=rix,
scope=past, 2026-09-17 evening" — rather than relying on the leaf skill to
re-derive or re-ask for either. `hutch` and `instrument` are the same
identifier: Prometheus's `instrument` label is set directly from a
component's `--hutch` argument (`psdaq/psdaq/cas/epics_exporter.py:29-51`,
`self._hutch = hutch` fed into `g.add_metric([self._hutch, self._id], ...)`
under the `instrument` label), so no translation is needed between this
router's `hutch` and `psana-daq-monitor`'s `instrument` label.

### Which config is running

Not directly discoverable at runtime. Corroborating signals to show (but not decide on):
- Config file mtime (recently modified configs are more likely to be current)
- Whether the config's selected process set matches `showPlatform` output

If uncertain, ask the user — they know which config they launched.

---

## Dispatch vs. sweep

Once `hutch` and scope are established, decide which of these two paths to
take:

- **The report names a specific angle** (state, metrics, logs, or
  configuration) → use the **symptom → skill routing table** below and load
  exactly **one** sibling skill. This is the cheap, fast path — do not run
  the full sweep for a targeted question like "what's the deadtime right
  now?"
- **The report is vague** — no specific angle named (e.g. "the DAQ is
  broken", "we're losing data", "what's wrong with the DAQ", "give me a full
  assessment", "why did this run go bad") → run the **autonomous sweep**
  (below) across every applicable angle and return one ranked report.

## Symptom → skill routing table

| Symptom / question | Load skill |
|---|---|
| DAQ won't start / stuck in a state / failed transition / "X did not respond to \<transition\>" / "X did not respond to \<transition\> phase 2" | `psana-daq-control` |
| "Why isn't detector X being recorded" / readout group questions | `psana-daq-control` |
| deadtime, damage %, event rates, buffer occupancy, MEB/TEB timing, "what does Grafana show" | `psana-daq-monitor` |
| "what actually failed", error messages, crash traces, "why did component X die", current/live run diagnostics | `psana-daq-logs` |
| "was detector X misconfigured", "what was the trigger/timing config", "did a parameter change mid-run" | `psana-configdb` |
| "correlate a metrics spike with a specific failing component" | `psana-daq-monitor` then `psana-daq-logs` |
| "is this new behavior, or did a config change cause it" | `psana-daq-monitor` (or `psana-daq-logs`) then `psana-configdb` |
| vague report, e.g. "the DAQ is broken" / "we're losing data" with no other detail | run the **autonomous sweep** below |

## Branch-on-state workflow (live sessions)

Check DAQ **state first, always** — it tells you whether the DAQ reached
running at all, which determines every subsequent step. This tree applies to
**live** sessions; a past session has no live signal left to branch on (see
"For a past session" under the sweep below).

```
state first, always  (psana-daq-control: bare daqstate call)

  └─ NOT running / stuck at a state boundary
       └─ logs next  (psana-daq-logs: control.log is timestamped and names the
       |              culprit; no meaningful metrics from a DAQ that never
       |              reached configured — checking metrics first can mislead)
            └─ configdb  (psana-configdb: did a config change cause the
                          component to fail?)

  └─ RUNNING but degraded  (deadtime, damage, slow event rate, etc.)
       └─ metrics next  (psana-daq-monitor: only source of a time window —
       |                 C++ component logs carry no timestamps, so you cannot
       |                 time-grep them without a metrics-derived window)
            └─ logs  (psana-daq-logs: grep around the metrics-derived window)
                 └─ configdb  (psana-configdb: correlate with config history)
```

**Why this branch, not a fixed order:**
367 of 400 xpp log files across one month carry no timestamps at all. C++
components (`SysLog.hh`) emit raw, untimestamped lines. Only `control.log` and
`control_gui.log` (Python logging) timestamp their output. This asymmetry is what
drives the split:

- For startup failures, `control.log` IS timestamped and names the culprit
  outright (e.g. `2026-09-02 08:26:57,015 xpp-control: <E> hsd_1 did not
  respond to alloc`). Metrics contribute nothing — the DAQ never reached
  `configured`.
- For running-degraded symptoms, component logs have no clock, so a
  metrics-derived time window is the only way to scope a log grep.

**Transition note:** this branch is partly a workaround for the
untimestamped-C++-logs defect being fixed in this same PR via
`psalg/psalg/utils/SysLog.hh`. Once that fix is deployed and log rotation turns
over, component logs will carry timestamps and the running-degraded path can be
simplified.

This order is a recommendation for **targeted dispatch**, not a requirement.
If the user's question already targets one angle specifically (e.g. "what's
the deadtime right now?"), load that skill directly. The **sweep** below
consumes this same tree as its live-session ordering, rather than restating
it.

---

## Autonomous sweep (vague reports)

You are producing a **single ranked diagnostic report** for one DAQ session —
live or a specific past one, per the scope already established above — by
sweeping every applicable diagnostic angle yourself, with no further
questions to the user after the scope was established. This is not a new
diagnostic technique: it is entirely composed of the four sibling skills'
existing methods, called in sequence and narrated as it goes.

**The sweep has exactly one deliverable: the structured report defined in
"Final step — emit this report" below.** Running the legs and narrating
them is not the deliverable — it is preparation for it. Do not stop after
narrating the legs and answering in prose; the sweep is not complete until
you have emitted that report, verbatim in shape, as your last action.
Treat the sequence below as ordered steps ending in that report, not as a
set of independent sections to consult:

1. Run each applicable leg below (State, Logs, Metrics, ConfigDB), narrating
   briefly before each one (e.g. "Checking DAQ state...", "Reading logs for
   session `<prefix>`...", "Querying metrics...", "Checking config for
   implicated detectors...") — this is not a silent background sweep, the
   user should see progress as each leg completes.
2. Once every applicable leg has run (or been explicitly skipped with a
   reason — see "Account for every leg" below), stop narrating and emit the
   final report per "Final step — emit this report." This is mandatory,
   not optional formatting — a vague report that ends in unstructured prose
   instead of this report has not completed the sweep.

### Branch-on-state ordering

For a **live** session, follow the **"Branch-on-state workflow (live
sessions)"** section above to decide whether logs-then-configdb or
metrics-then-logs-then-configdb is the more informative order. This
citation assumes the live branch has already been established (state vs.
running-degraded) by the time you reach this point.

For a **past** session there is no live signal left to branch on — the DAQ
reached whatever state it reached, and that is now history, not something to
branch a *decision* on. This is a **deliberate divergence from the live
case, not an oversight**: for a past session, run every leg below that is
still applicable (state-from-log, logs, metrics with a derived window,
configdb if implicated) regardless of what branch-on-state would have
recommended live.

### State leg

- **Live session**: call `daqstate`/`showPlatform` exactly as documented in
  `psana-daq-control` (Section 1 for the state machine and path arithmetic,
  Section 6 for the tool invocations) — do not restate its read/mutate
  command tables here, cite them.
- **Past session**: the state machine cannot be queried retroactively — there
  is no live API for historical state. Degrade to reading that session's
  `control.log` for its timestamped transition history (the same file
  `psana-daq-control` reads live, but after the fact). **Label this leg's
  output "reconstructed from control.log, not live"** — both in the sweep
  narration as you report it, and again in the Raw Findings section below.
- **`showPlatform`**: runs immediately after state, for **live sessions
  only** (registration/readout-group info). There is no past-session
  equivalent — state that plainly in the Raw Findings rather than silently
  omitting the line.
- **The State leg is not skippable just because the user framed the request
  as "historical" or "past."** "Past" describes the *scope* (which session
  to look at), not permission to omit a leg. If a user's phrasing sounds
  like it's asking only about metrics or logs, that does not exempt this
  leg — run it and report a real `control.log`-derived finding in Raw
  Findings, not `skipped — not requested`.
- **If no session touched the requested window at all** (e.g. a 24h window
  with no DAQ activity in it), that absence *is* the State leg's finding —
  report it as such (e.g. "no session ran inside the window; last session
  ended <N> days earlier at <timestamp>") rather than marking the leg
  skipped. Do not fold this finding into Independent Issues or another
  section instead of reporting it here — Raw Findings should show the
  State leg actually ran and what it found, even when the answer is "DAQ
  was idle throughout."

### Logs leg

Scope to the chosen session: live → current/newest active session; past →
the session prefix pinned during scope establishment. Cite and reuse, rather
than re-deriving:

- `psana-daq-logs`' **"Session selection — present a list and ask"** section
  (used to pin down an exact prefix for the past-session case, if not
  already exact; for live, the current/newest session is used directly).
- `psana-daq-logs`' **"Interpreting `<C>`/`<E>` messages"** section for how
  to read whatever Critical/Error lines turn up.

Do not re-derive the grep patterns or heuristics from either section — cite
them and apply them.

### Metrics leg

Cite these six named groups from `psana-daq-monitor` by section — do not
re-paste their PromQL:

1. Event rate ("A. Is the DAQ running? Is the event rate healthy?")
2. Deadtime / `DeadFrac` ("B. Is there excessive deadtime?")
3. Damage ("C. Is there damage?")
4. The `drp_num_*` counters ("D. Are there DRP errors?")
5. `MRQ_BufCt` ("I. Is online monitoring healthy? (MEB)")
6. EB fixup/timeout rates ("G. Is the event builder healthy?")

**Time window handling — the one leg that is fully symmetric between live
and past:**

- **Live**: use the monitor skill's own recent-window defaults (its
  "Default time ranges" table).
- **Past session**: derive the window from the session's own wall-clock
  bounds, not from "now":
  - `start` = the timestamp parsed from the session's log-file prefix
    (format `DD_HH:MM:SS`).
  - `end` = the mtime of the session's last-written log file.
  - Pass this derived `{start, end}` as the Grafana `timeRange` on every
    metrics query in this leg.

### ConfigDB leg — gated by the implicated-detector rule

Only query ConfigDB for detectors that are **implicated**. Don't dump every
device. A detector is implicated **iff** it is:

  (a) named in a state/control-log error, **or**
  (b) named in a `<C>`/`<E>` log line, **or**
  (c) the `detname` label on a breached `drp_num_*` or damage metric.

If no detector meets one of these three criteria, skip this leg entirely and
say so ("no detector implicated — skipped") rather than performing a lookup
anyway.

For any implicated detector, hand off using `psana-configdb`'s existing
**"arrived here from another skill"** handoff pattern (its "Related skills"
paragraph near the top of that file) — do not re-derive how to call it or
which endpoints to use; that skill owns its own endpoint list.

### Account for every leg

A leg is **applicable** unless one of these excludes it — and each exclusion
must be reported, never silently omitted:

- **live** sessions: branch-on-state prunes a leg for this particular branch
  (e.g. metrics contributes nothing for a DAQ that never reached `configured`)
- a leg's data source is unreachable per the "Prerequisites / preflight"
  section
- a documented per-leg gate excludes it (ConfigDB's implicated-detector rule
  above; `showPlatform` has no past-session equivalent)

**Live and past sessions have different completeness contracts, by design:**
live may legitimately prune legs via branch-on-state; a past session has no
live signal to branch on, so it runs every leg that is still applicable
(state-from-log, logs, metrics with a derived window, configdb if
implicated) regardless of what branch-on-state would have recommended live
— see "Branch-on-state ordering" above.

Every line in the final report's Raw Findings section must be present: a
finding, or `skipped — <reason>`. A leg that was pruned or gated must be
indistinguishable in the report from one that was reported as skipped —
never simply absent.

### Final step — emit this report

**This is the sweep's mandatory last action, not a reference template to
consult if convenient.** After every applicable leg has run (or been
explicitly skipped per "Account for every leg" above), stop and emit exactly
this structure as your response — do not answer in prose instead, and do
not omit it because the cause seems obvious. Fill in the placeholders; do
not change the headings or field names:

```
## Most Likely Cause
<one-line statement>
Confidence: <high | medium | low>
Evidence:
  - <finding> — <verified-live | verified-against-real-logs | inferred-from-code-only>
  - <finding> — <tier>

## Also Possible
1. <alternative explanation for the SAME symptom> — Confidence: <high | medium | low>
   Evidence:
     - <finding> — <tier>
2. ...

## Independent Issues
<separate faults found during the sweep that do NOT explain the primary
 symptom — write "none" if every finding relates to the one fault above>
1. <issue> — Confidence: <high | medium | low>
   Evidence:
     - <finding> — <tier>

## Raw Findings
- State: <finding, or "skipped — <reason>", or "reconstructed from control.log, not live" if past session>
- showPlatform: <finding, or "skipped — <reason>", or "not available for past sessions" if past session>
- Logs: <finding, or "skipped — <reason>">
- Metrics: <finding, or "skipped — <reason>">
- ConfigDB: <finding, or "no detector implicated — skipped">
```

**Two independent label systems — do not conflate them:**

- **`Confidence:`** (verdict-level, one per cause/alternative/issue) — how
  confident the sweep is that this *conclusion* is right, not how the
  supporting evidence was obtained:
  - **high** — direct evidence names the failing component and the
    mechanism is established
  - **medium** — evidence is consistent with this cause but doesn't
    isolate it from other explanations
  - **low** — a plausible inference; competing explanations have not been
    ruled out
- **Evidence tiers** (per evidence item, one per line under `Evidence:`) —
  how that specific piece of evidence was obtained. Use exactly these three,
  the same vocabulary as `psana-daq-control`'s "Provenance and confidence"
  section — do not invent a fourth tier here:
  - `verified-live`
  - `verified-against-real-logs`
  - `inferred-from-code-only`

**Also Possible vs. Independent Issues — the discrimination rule:** does
this finding offer an *alternative explanation for the same symptom* the
user reported (→ **Also Possible**), or is it a *separate fault* that would
still exist even if the primary cause were fixed (→ **Independent Issues**)?
A dying HSD card that presents as configure-timeout and, separately, as
elevated deadtime is one fault surfacing twice — report it once, not as two
Independent Issues. Rank Independent Issues by severity.

<!--
Note for future maintainers: the `Evidence:` field and the `## Most Likely
Cause` heading above are consumed verbatim by a planned future
history/checkpointing retrofit to this skill (item 07 in the ami-repo
planning docs). `Evidence:` is now a nested list of per-item findings each
carrying its own provenance tier, rather than a single citation string — the
retrofit should read the list, not assume a scalar. Do not casually rename
these field names or restructure this template without checking that
dependency.
-->

### Sweep provenance note

The sweep introduces **no new verified facts** about the DAQ of its own — it
is entirely composed of the four sibling skills' existing techniques,
cited and invoked rather than duplicated. The two genuinely novel pieces,
and what reviewers should scrutinize most closely, are: the **ranking
heuristic** — how the sweep decides what goes under "Most Likely Cause"
versus "Also Possible" when multiple legs report findings that could
compete — and the **Also-Possible-vs-Independent-Issues discrimination
rule** above. Both are **judgment calls**, not verified facts, and should be
treated and reviewed as such.

### What the sweep does not do

- No history/session search across prior runs, no GitHub/Slack integration,
  no occurrence/checkpoint bookkeeping — all out of scope here, deferred to a
  separate future skill.
- No "questions for reviewers" section in this file — reviewer questions for
  this skill live in a separate planning doc, not in the shipped skill.

---

## Release-source navigation (cross-cutting)

When any skill needs to read source ground truth — to look up a message, verify
a timeout, or check a guard — use this technique to locate the exact source tree
the running DAQ was built from.

**How to find the source:**

1. Read `# TESTRELDIR:` from the log file header (`psana-daq-logs` documents
   the header format). `TESTRELDIR` points at `<root>/install` (compiled
   output — `bin/` and `lib/` only). Source is at the **parent**:
   strip the trailing `/install`.
2. The source root contains e.g. `psdaq/psdaq/control/control.py` — confirmed
   readable.

**Identifying the release:** for conda releases the `lcls2_<MMDDYY>` component of
the `TESTRELDIR` path is the version marker. Do not try to parse a universal
pattern — the path may be a developer home directory (e.g.
`/sdf/home/w/weaver/lcls2/install`), not a conda release.

**`git` is unusable in these trees:** release trees are owned by `psrel:xs`; git
refuses with "dubious ownership" for any user. Plain file reads and greps work fine.

**`GIT_DESCRIBE` is absent** in all logs prior to the `daqlog_header.py` fix
shipping (Change C in this PR). After the fix, new sessions will carry it. Check
whether it is present; do not assume either way. The `lcls2_<MMDDYY>` path
component is the reliable version marker until then.

**Verified example:** `/sdf/group/lcls/ds/ana/sw/conda2/rel/xpp/lcls2_091826` —
strip `/install` suffix; `control.py` confirmed present and readable at the parent.

**Multiple releases coexist** — three different ones in one month at xpp alone.
Line numbers shift between releases by inconsistent amounts (e.g. a message at
`:2255` in `lcls2_091826` is at `:2220` in `lcls2_061226`; drift=35). A citation
valid in one release is wrong in another and yields a plausible-looking but
incorrect answer.

**If you are unsure which tree to read, ask the user.** Ask when: multiple
`TESTRELDIR` values are in play, the path does not match the conda convention, the
tree is unreadable, or the question spans a time range covering more than one
release. **Always state which tree you used**, so the human can correct you.

Other skills reference this section. The canonical source of this technique is here
in the router, whether reached via dispatch or the sweep; leaf skills point back to
it rather than duplicating it.

## Prerequisites / preflight

Before routing or sweeping, check what's actually usable in the current environment
and tell the user upfront if an angle is unavailable.

- **State/control (`psana-daq-control`)**: works with no external dependencies —
  talks directly to the `control` process over ZMQ. Reachable from `<hutch>-daq`
  hosts (where `ami-client` runs in production); the control port (`front_rep_port`
  = PORT_BASE + platform + 20, e.g. 30000 for platform 0) is firewalled from
  analysis nodes (e.g. `sdfiana024`, `sdfiana027`). Two distinct failure modes can
  block this, and either lands you in the same place: `daqstate`/`showPlatform`
  may simply be absent from `PATH` (no DAQ conda env active — verified-live: the
  release's `setup_env.sh`/`setup_env_daq.sh` does put them on `PATH`, but that
  alone does not fix reachability, since the ZMQ connect then fails separately on
  DNS/firewall grounds from an analysis node), or the binaries are present but the
  connect itself fails (firewalled, or the collect host doesn't resolve via DNS
  from an analysis node). Either way: see `psana-daq-control`'s own **"SSH
  escalation from analysis nodes"** section for the full remedy (env-sourcing,
  `.pcdsn`-suffixed ssh to the collect host, retry guidance) rather than
  attempting a narrower version of it here — or derive state from `control.log`
  if escalation is declined or unavailable.
- **Logs (`psana-daq-logs`)**: requires a readable DAQ log directory for the
  relevant hutch. The path convention and the list of hutches confirmed
  present/absent are documented in `psana-daq-logs/SKILL.md`:

  | Status | Hutches |
  |---|---|
  | Present with data (readable) | xpp, tmo, rix, mfx, ued |
  | Directory exists but empty (no year subdirs) | txi, det |
  | No directory | xcs, cxi, asc, tst |

  Check with `ls`/`test -d` before assuming it exists; if absent, say so rather
  than guessing an alternate path.
- **ConfigDB (`psana-configdb`)**: requires reachability to
  `pswww.slac.stanford.edu`. A simple check:

      curl -sf -o /dev/null -w '%{http_code}' \
        https://pswww.slac.stanford.edu/ws/configdb/ws/configDB/get_hutches/

  A `200` means the service is reachable and this angle is usable.

If one or more angles are unavailable, tell the user explicitly which diagnostic
angles you can and cannot pursue before proceeding with the ones that remain
(dispatch or sweep).

## AMI / DAQ ownership seam

This boundary is easy to get wrong in both directions. The AMI-spawned agent runs
alongside AMI, and **AMI is itself a shared-memory client of the MEB**:

- **`ami-performance-monitor`** owns AMI-side symptoms — graph latency, worker
  starvation, GUI lag, computation throughput.
- **This DAQ skill suite** owns the DAQ side — component failures, state
  transitions, deadtime, damage, configuration.
- **The MEB is the boundary.** If AMI isn't updating: check whether the MEB is
  delivering events (DAQ side, this suite) before blaming AMI's graph processing
  (AMI side, `ami-performance-monitor`).

## What this skill does not do

- For the **dispatch** path: intentionally has no PromQL, no log grep patterns, and
  no ConfigDB endpoint URLs of its own — those live in the leaf skills so they stay
  authoritative in one place. Load the relevant sibling skill before doing any real
  diagnostic work.
- For the **sweep** path: see "What the sweep does not do" above.
