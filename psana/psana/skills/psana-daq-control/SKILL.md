---
name: psana-daq-control
description: Diagnose LCLS-II DAQ run-control state and failed state transitions. Use for "the DAQ won't start", "it's stuck in connected/configured", "alloc/connect/configure failed to change state", "X did not respond to <transition>", "X did not respond to <transition> phase 2", rollcall failures, "why isn't detector X being recorded", readout-group questions, and scan/step (beginstep/endstep) problems. Reconstructs state-machine transitions from control.log — needs no Grafana, no MCP, no ConfigDB reachability, and no reachable DAQ, so it works offline and when everything else is down.
---

# Skill: psana-daq-control — LCLS-II DAQ Run Control & State Machine Diagnosis

Diagnoses the DAQ *control layer*: the collection manager (`control`)'s
strict 8-state linear machine, and the per-component transition handshakes
underneath it. The question this skill answers:

> **Did the DAQ reach the state it should have, and if not, exactly where
> did it stop?**

`psana-daq-monitor` sees metrics from a DAQ that's already running.
`psana-daq-logs` reads text after the fact. `psana-configdb` answers
configuration questions. None of them model the state machine, so none can
localize a startup failure to a state boundary.

**Primary source: `control.log`'s timestamped transition history** — this
is the normal way this skill works, for a window ending now or any past
window alike. See "Reading transitions from control.log" below.

## READ-ONLY POSTURE — NON-NEGOTIABLE

You gather evidence and recommend; the **human** executes every
remediation.

| You MAY (read-only) | You MUST NOT (mutating) |
|---|---|
| Read `control.log` / `control_gui.log` | Edit any log file or `activedet.json` |
| Read `~<hutch>opr/.psdaq/*.activedet.json` | Run `daqstate`, `showPlatform`, or `selectPlatform` with ANY flags, including flags that look read-only |
| | Call any `DaqControl` method |

You may — and should — **recommend** a specific command, quoted exactly,
with a stated reason. The human runs it.

## Confidence and evidence tiers

Every conclusion gets a **Confidence** (`high`/`medium`/`low`); every
piece of evidence gets exactly one of four **evidence tiers**. The two
are independent axes. Full definitions and rules are in
`psana-daq/reference/report-template.md`, section "Two independent label
systems — do not conflate them."

Citations below derive from `lcls2_091826`. Grep anchors are given for
message strings (stable across releases); line numbers for structural
blocks shift between releases — verify in the running release's tree
(`psana-daq/reference/release-source.md`).

## Reading transitions from control.log

`control.log` and `control_gui.log` are Python-logged and timestamped
(unlike most C++ component logs). A session's transition history is
reconstructed directly from it:

- `<E>`/`<C>` lines name failed transitions and the specific component or
  precondition involved (Section 3 below).
- The last successful transition before a failure, read against Section
  1's state machine, tells you where the DAQ actually stopped.
- `# CMDLINE:` in the header shows the actual flags the control process
  was launched with — including its real `-T` phase-2 timeout; don't
  assume a default without checking it (see "Timeouts" below).

---

## Section 1 — The State Machine

**8 states, strictly linear** (`psdaq/psdaq/control/ControlDef.py:21,29`):

```
reset → unallocated → allocated → connected → configured → starting → paused → running
  0         1             2           3            4           5         6         7
```

**15 transitions**: `rollcall, alloc, dealloc, connect, disconnect,
configure, unconfigure, beginrun, endrun, beginstep, endstep, enable,
disable, slowupdate, reset`. Each has exactly one legal from-state and
to-state, guarded by a `condition_*` method (`control.py:841+`):

| Transition | From → To |
|---|---|
| `reset` | `*` → `reset` |
| `rollcall` | `reset`,`unallocated` → `unallocated` |
| `alloc` | `unallocated` → `allocated` |
| `dealloc` | `allocated` → `unallocated` |
| `connect` | `allocated` → `connected` |
| `disconnect` | `connected` → `allocated` |
| `configure` | `connected` → `configured` |
| `unconfigure` | `configured` → `connected` |
| `beginrun` | `configured` → `starting` |
| `endrun` | `starting` → `configured` |
| `beginstep` | `starting` → `paused` |
| `endstep` | `paused` → `starting` |
| `enable` | `paused` → `running` |
| `disable` | `running` → `paused` |
| `slowupdate` | `running` → *(none)* |

`slowupdate` is **internal** (destination `None`, no status report) — the
periodic SlowUpdate heartbeat, not a user-driven state change.

### Path arithmetic — compute before reading logs, treat as a candidate

`next_dict` (`control.py:281`) maps `(current, target) → next transition`.
Because the chain is linear, you can compute the remaining path by walking
`states` in `ControlDef.py`.

**Worked example.** `control.log` shows the last successful state as
`connected`, target `running`: index `connected`=3, `running`=7 (forward);
remaining order `configure → beginrun → beginstep → enable`;
`next_dict['connected']['running']` = `'configure'` — **the next
transition attempted was `configure`.**

This identifies **what would be attempted next, not what has failed** —
require the log's actual attempt-and-error line for that transition before
reporting it as the failed boundary. `lastTransition` (if read live) is
not a reliable universal record of the last *failed* attempt — it can
also reflect a successful one followed by a stall elsewhere, or a retry.

- Report the stall boundary with its corroborating line, not just the
  state: "stuck in `connected`, target `running`; `control.log` shows
  `configure failed to change state` at `<timestamp>`" is actionable;
  "it's in `connected`" alone is not.
- A regression (`running` → `paused` unprompted) means `disable` fired —
  check the log for whether an operator did it or `condition_disable`
  triggered on an error path.
- `reset` is reachable from any state — a jump to `reset` tells you
  nothing about intermediate failures.

---

## Section 2 — Two-Phase Transition Semantics

Every non-trivial transition runs in two phases:

| Message | Candidate meaning |
|---|---|
| `'%s did not respond to %s'` (phase 1, `control.py`: `"did not respond to %s' % (alias, transition)"`) | The component never acknowledged. **Candidates**: not running, crashed before responding, or a ZMQ-path problem — check its process and startup log via `psana-daq-logs` before concluding which. |
| `'%s did not respond to %s phase 2'` (`control.py`: `"did not respond to %s phase 2"`) | Phase 1 succeeded, then no phase-2 reply. **Candidates**: the process is alive but wedged doing the real work, or it died *after* phase 1 but before phase 2 (a different failure mode than "wedged"), or a phase-2-specific IPC/scope issue. Phase-2 scope is `drp`+`meb` only (`get_phase2_replies()`, `control.py:1254+`) — a TEB has no phase-2 reply to miss. |
| `'%s: %s' % (alias, err_msg)` (`control.py`: `"check_answers"`) | The component replied with its own error text (`err_info`) — read it literally, higher-quality evidence than a timeout message. |

Neither phase-1 nor phase-2 failure **proves** process-missing vs.
process-alive on its own — both are candidate predictions. Require the
corroborating evidence (the process's own log, or its rollcall lines in
`control.log` — see Section 3/4 below) before stating one as fact. This
split is **inferred-from-code-only**, not operationally confirmed as a
reliable discriminator — treat it as a strong hypothesis guiding where to
look next, not a verified rule.

### Timeouts

Read the control process's own `# CMDLINE:` header for its actual `-T`
(phase-2 timeout) rather than assuming a code default — real deployments
have been seen running with a non-default value. The code defaults
(`control.py:2614-2615`): phase-2 `-T` 12500ms, `--rollcall_timeout` 30s
(re-broadcasts every 1s), `alloc` phase-1 5000ms (hardcoded), `configure`
phase-1 60000ms (hardcoded). A phase-2 timeout close to the configured
value, rather than a large margin past it, may be an under-set timeout
rather than a wedged component — check the actual flag before concluding
either way.

---

## Section 3 — Error Interpretation Rules

Full message-by-message grep anchors (per-transition phase-1 templates,
rollcall messages, topology preconditions, DRP alias grammar,
active-detectors-file errors, XPM PV-put failures, state guards) are in
`reference/control-messages.md`. The durable interpretation rules:

- **`X failed to change state` is never the diagnosis** — it's the
  umbrella result of a `condition_*` guard returning False. Always read
  **upward** in the log for the line naming the actual component or
  precondition.
- **`N client did not respond to <t>` follows the per-alias lines** — the
  count says how many, the preceding lines say which. Read both.
- **`TEB/MEB didn't hear from:` is a two-line finding** — the header names
  a missing contributor, the **next** line(s) name it.
- **A rollcall warning is a leading indicator, not the failure** — rollcall
  misses don't stop the state machine (`retval = True` regardless). The
  same component often reappears as an `alloc` phase-1 non-response
  shortly after; report the pair together.
- **Simultaneous multi-alias misses (e.g. all of `hsd_0..3`) suggest a
  systemic group/launcher failure**, not independent per-card hardware
  faults — check whether siblings share the same timestamp before
  recommending anything card-specific.
- **`selectPlatform only permitted in unallocated state` is operator
  sequencing, not a DAQ fault** — someone ran it while past `unallocated`.
  Remediation: deallocate first, then retry (human does this).
- **Topology preconditions** (`at least one DRP is required`, `ami NOT
  supported in absence of MEB`) are **configuration/selection faults**
  most of the time, but check: a selected component that also failed to
  launch would produce the same symptom. Point at the activedet file
  (Section 4) and `control.log`'s rollcall lines — `rollcall: <alias>
  selected for data collection (readout group N)` (logged at INFO,
  `control.py:1928`; a variant without the group number is at
  `control.py:1930`) and `rollcall: <alias> NOT selected for data
  collection` (logged as a warning, `control.py:1901`) — as the first
  check, not a process log. These reflect selection as of rollcall time
  for that session. Don't rule the process log out if the rollcall lines
  look consistent with everything selected.
- **XPM PV-put failures** are control-plane reachability to the XPM, not a
  DRP or ConfigDB problem.
- **Active-detectors-file errors are file-content faults** — expect
  widespread downstream `alloc` errors from one file-level cause, not
  independent per-component failures.

---

## Section 4 — The `activedet.json` Artifact

The file that decides which components the DAQ requires and each one's
readout group — the direct answer to "why isn't detector X being
recorded?"

**Path**: `~<hutch>opr/.psdaq/x<XPM_MASTER>_p<PLATFORM>.activedet.json`,
overridable with `-r`. Read the `active detectors file: <path>` `<I>` line
from `control.log` rather than reconstructing the path. If the path is
`/dev/null`, `bypass_activedet` is set and **all components default to
active with readout group = platform** — the file is irrelevant to that
session.

**Structure** (verified-live against a real file):

```json
{
  "activedet": {
    "drp": {
      "epix100_0": { "active": 1, "det_info": { "readout": 0 } },
      "bld_0":     { "active": 0, "det_info": { "readout": 0 } }
    },
    "meb": {}, "teb": {}
  },
  "history": {
    "drp": { "alvium_0": { "det_info": { "readout": 0 } } },
    "tpr": {}
  }
}
```

- **`activedet`** splits into `active_set`/`inactive_set`
  (`get_active_and_inactive()`). A component in `active_set` is required —
  its absence at rollcall/`alloc` produces the respective warning/error. A
  component present but `active: 0` **does not by itself establish
  deliberate deselection** — it means it wasn't selected *this time*; cross-
  check the `rollcall: ... NOT selected for data collection` log line and,
  if available, the current config's intended component set, rather than
  treating `active: 0` alone as proof of intent.
- **`history`** records the **last-recorded** readout group for every
  DRP/TPR alias ever seen on this platform — **not a known-good value**.
  `condition_alloc` writes `history` entries before Configure or
  acquisition has succeeded, so a recorded value doesn't mean it ever
  actually worked; it only means it was assigned at some point. Use it as
  a comparison point, not as ground truth.

**Answering the two questions:**
1. **"Why isn't X being recorded?"** → look up `X` in `activedet.drp`.
   Missing or `active: 0` → likely deselected, but corroborate against the
   rollcall log line and the intended component set before calling it
   confirmed-deliberate.
2. **"What readout group should X be in?"** → `activedet.drp.X.det_info
   .readout` is current; `history.drp.X.det_info.readout` is the
   last-recorded value to compare against, not a known-good baseline.

Never edit this file — recommend the change and exact value; the human
applies it via the control GUI or `selectPlatform`.

---

## Section 5 — Scan / Step Content

Steps are more transitions on the same state machine:

```
configured ──beginrun──> starting ──beginstep──> paused ──enable──> running
                             ^                                          |
                             └──────────── endstep ◄──disable───────────┘
```

Drivers: `ConfigScan.py` (configuration scan), `BlueskyScan.py`
(Bluesky-integrated), `TimedRun.py` (fixed-duration). `step_value`
(`ControlDef.STEP_VALUE`) is the simulated-motor name reserved for step
value. `step_done` is an `Event`; the control process's `step_done_func`
thread watches the XPM's `StepDone` PV, honoring it only in `running`/
`paused` and logging `StepDone PV=... (ignore)` otherwise.

- **A scan stuck between steps is a state-machine stall** — apply Section
  1: `paused`→target `running` means `enable` failed; `starting`→target
  `paused` means `beginstep` failed.
- **A scan stuck at step 1 with the DAQ in `running`** is a candidate
  `step_done` problem, not a transition problem — check the control
  process's debug log for `StepDone PV=... (ignore)`.
- `beginstep`/`endstep` phase-1 failures use the same `did not respond to
  <transition>` template and two-phase rules as any other transition.
- `slowupdate` reports no status — won't appear as a step boundary.

---

## Section 6 — Handoffs

State the handoff explicitly; load the sibling skill by name before using
its techniques.

| Finding | Hand off to | What to ask |
|---|---|---|
| Phase-1 non-response, or missing / NOT selected in `control.log` rollcall lines | `psana-daq-logs` | Startup header + `<[EC]>` grep for that component, same session |
| `configure` phase-1 failure | `psana-configdb` | Was there a recent config change to the implicated device/alias? **A phase-1 failure alone does not establish that config retrieval/application failed** — check logs first. |
| Deadtime/damage/buffer symptoms once running | `psana-daq-monitor` | Event rates, `DeadFrac`, damage, buffer occupancy for the implicated detector |
| Phase-2 non-response | `psana-daq-logs` first, then `psana-daq-monitor` | What the process logged during the transition; if mid-run, whether metrics show it falling behind |
| `TEB/MEB didn't hear from: <alias>` | `psana-daq-logs` for `<alias>`, then `psana-daq-monitor` | Its process/log, then `EB_FxUpCt`/`EB_CbMsMk` |
| Vague report, no state info yet | `psana-daq` (router) | Let it pick the angle |

Log path conventions, filename grammar, and header format are owned by
`psana-daq-logs`. Metric names/thresholds by `psana-daq-monitor`. ConfigDB
endpoints by `psana-configdb`. Don't restate them here.

---

## Appendix — ZMQ Port Arithmetic

Full formula table and the XPM `FEH`/`NEH` offset quirk are in
`reference/control-messages.md`'s appendix. Durable summary: `back_*`
ports are control↔component channels (a component that never answers a
broadcast has a `back_pull`/`back_pub` problem); `front_*` are
control↔client (GUI and scripts); `fast_rep` serves the fast-reply thread;
`step_pub` is the step-done publisher.

## File-to-Role Reference

| File | Role |
|---|---|
| `psdaq/psdaq/control/control.py` | Collection manager: state machine, transition guards, error reporting |
| `psdaq/psdaq/control/ControlDef.py` | Constants: states, transitions, port bases |
| `psdaq/psdaq/control/DaqControl.py` | Python API to the live control process — never call its `set*`/`select*`/`store*` methods, and never call it directly from this skill at all |
| `psdaq/psdaq/control/daqstate.py` | CLI including state-changing flags (`--state`/`--transition`/`--config`/`--record`/`--bypass`) — never run |
| `psdaq/psdaq/control/showPlatform.py` | CLI querying the live control process — never run |
| `psdaq/psdaq/control/selectPlatform.py` | CLI: platform selection — never run, not just "mutating" |
| `psdaq/psdaq/control/ConfigScan.py` | Scan/step acquisition |
