---
name: psana-daq-snapshot
description: Autonomous end-to-end DAQ health sweep for a single session — live or a specific past one. Use for "what's wrong with the DAQ", "diagnose the current run", "what happened in last night's run", "give me a full assessment", "why did this run go bad" — asks one scoping question up front, then investigates state, logs, metrics, and (if implicated) configdb on its own, and returns one ranked report. Distinct from `psana-daq`, which is a dispatch router that hands off to a single angle and asks nothing else itself.
---

# Skill: psana-daq-snapshot

# LCLS-II DAQ Autonomous Session Snapshot

You are producing a **single ranked diagnostic report** for one DAQ session —
live or a specific past one — by sweeping every applicable diagnostic angle
yourself, with no further questions to the user after the initial scoping
question. This skill does not introduce new diagnostic techniques: it is
entirely composed of the four sibling skills' existing methods, called in
sequence and narrated as it goes.

## Read-only posture

This skill is read-only, no exceptions. Gather evidence and recommend; the
**human executes every remediation.** See `psana-daq-control/SKILL.md`'s
**"READ-ONLY POSTURE — NON-NEGOTIABLE"** section for the canonical
may-run/must-not-run table — do not restate it here.

---

## Step 0 — the one question, asked once

Ask, at the very start, before anything else:

> **"Live/current session, or a specific past session?"**

This is the **only** question this skill asks. Everything from this point
forward runs autonomously — no further prompts mid-sweep.

- **Live/current** → proceed to the sweep using the current/newest active
  session.
- **Past session** → hand off to `psana-daq-logs`' existing session-listing
  mechanism (its **"Session selection — present a list and ask"** section) to
  show candidate sessions and get the user's confirmed prefix. Do not
  re-derive that listing algorithm here — cite it and call it. Once you have
  the confirmed `DD_HH:MM:SS` prefix, that is the only further input needed;
  proceed to the sweep.

---

## The sweep

Narrate briefly before each leg (e.g. "Checking DAQ state...", "Reading logs
for session `<prefix>`...", "Querying metrics...", "Checking config for
implicated detectors...") — this is not a silent background sweep, the user
should see progress as each leg completes.

### Branch-on-state ordering

For a **live** session, follow `psana-daq`'s existing branch-on-state
workflow — see its **"Branch-on-state workflow for vague reports"** section —
to decide whether logs-then-configdb or metrics-then-logs-then-configdb is
the more informative order. Do not restate that decision tree here; cite it.
This citation assumes the live branch has already been established (state
vs. running-degraded) by the time you reach this point.

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

### Logs leg

Scope to the chosen session: live → current/newest active session; past →
the confirmed prefix from Step 0. Cite and reuse, rather than re-deriving:

- `psana-daq-logs`' **"Session selection — present a list and ask"** section
  (already used in Step 0 for the past-session case; for live, the
  current/newest session is used directly).
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

---

## Output template

Reproduce this structure for every report — fill in the placeholders, do not
change the headings or field names:

```
## Most Likely Cause
<one-line statement>
Confidence: <verified-live | verified-against-real-logs | inferred-from-code-only>
Evidence: <citations — the exact line(s)/finding(s) that support this>

## Also Possible
1. <alternative> — Confidence: <tier> — Evidence: <citations>
2. ...

## Raw Findings
- State: <finding, or "reconstructed from control.log, not live" if past session>
- showPlatform: <finding, or "not available for past sessions" if past session>
- Logs: <finding>
- Metrics: <finding>
- ConfigDB (implicated detectors only): <finding, or "no detector implicated — skipped">
```

Confidence granularity: **one tag per cause/alternative**, not per individual
evidence item. Use exactly these three tiers (the same vocabulary as
`psana-daq-control`'s "Confidence labelling" section) — do not invent a
fourth tier here:

- `verified-live`
- `verified-against-real-logs`
- `inferred-from-code-only`

<!--
Note for future maintainers: the `Evidence:` field and the `## Most Likely
Cause` heading above are consumed verbatim by a planned future
history/checkpointing retrofit to this skill. Do not casually rename these
field names or restructure this template without checking that dependency.
-->

---

## Provenance note

This skill introduces **no new verified facts** about the DAQ of its own — it
is entirely composed of the four sibling skills' existing techniques,
cited and invoked rather than duplicated. The one genuinely novel piece,
and the one reviewers should scrutinize most closely, is the **ranking
heuristic** — how this skill decides what goes under "Most Likely Cause"
versus "Also Possible" when multiple legs report findings. That heuristic is
a **judgment call**, not a verified fact, and should be treated and reviewed
as such.

---

## What this skill does not do

- No history/session search across prior runs, no GitHub/Slack integration,
  no occurrence/checkpoint bookkeeping — all out of scope here, deferred to a
  separate future skill.
- No restating of `psana-daq`'s branch-on-state tree, or of any sibling
  skill's command tables, PromQL, or heuristics — cite the owning skill's
  section instead of copying it.
- No "questions for reviewers" section in this file — reviewer questions for
  this skill live in a separate planning doc, not in the shipped skill.
