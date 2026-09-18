---
name: psana-daq-control
description: Diagnose LCLS-II DAQ run-control state and failed state transitions. Use for "the DAQ won't start", "it's stuck in connected/configured", "alloc/connect/configure failed to change state", "X did not respond to <transition>", "X did not respond to <transition> phase 2", rollcall failures, "why isn't detector X being recorded", readout-group questions, and scan/step (beginstep/endstep) problems. Queries live DAQ state via daqstate/showPlatform/DaqControl and reads the activedet.json file — needs no Grafana, no MCP, and no ConfigDB reachability, so it works when everything else is down.
---

# Skill: psana-daq-control

# LCLS-II DAQ Run Control & State Machine Diagnosis

You are diagnosing the LCLS-II DAQ *control layer*: the collection manager
(`control`) that drives a strict 8-state linear machine, and the per-component
transition handshakes underneath it. The question this skill answers is the one
no other DAQ skill can:

> **Is the DAQ even running, and if not, exactly where did it stop?**

`psana-daq-monitor` sees metrics from a DAQ that is already running.
`psana-daq-logs` reads text after the fact. `psana-configdb` answers
configuration questions. None of them model the state machine, so none of them
can localize a startup failure to a state boundary.

## READ-ONLY POSTURE — NON-NEGOTIABLE

This skill runs against **live LCLS experiments**. You gather evidence and
recommend; the **HUMAN executes every remediation.**

| You MAY run (read-only) | You MUST NOT run (mutating) |
|---|---|
| `daqstate -P <hutch>` (prints current state) | `daqstate --state <target>` |
| `daqstate --state` (choice listing / status print) | `daqstate --transition <t>` |
| `daqstate --monitor` (passive status stream) | `daqstate --config` / `--record` / `--bypass` / `-B` |
| `showPlatform` (incl. `--json`) | `selectPlatform` (any invocation) |
| `DaqControl.getState()` / `getStatus()` / `getPlatform()` / `getInstrument()` / `getJsonConfig()` / `monitorStatus()` | `DaqControl.setState()` / `setTransition()` / `setConfig()` / `setRecord()` / `setBypass()` / `selectPlatform()` / `storeJsonConfig()` |
| Reading `~<hutch>opr/.psdaq/*.activedet.json` | Writing/editing any `activedet.json` |
| Reading log files (see `psana-daq-logs`) | Restarting or killing any DAQ process |

You may — and should — **recommend** a specific `setState`/`--transition`
command, quoted exactly, with a stated reason. Present it as a suggestion for
the human to run. Never run it yourself.

## Confidence labelling

Every conclusion you report must carry one of these, and this skill's own
claims are labelled the same way:

- **verified-live** — executed against the real service/filesystem
- **verified-against-real-logs** — grepped from actual production log files
- **inferred-from-code-only** — read from source, never operationally confirmed

Prefer a ranked "most likely / also possible" over one confident answer.

**Citation policy:** citations in this skill derive from `lcls2_091826`. Grep
anchors are provided for message strings — those are stable across releases. Line
numbers for structural blocks shift between releases; verify in the running release's
tree (`TESTRELDIR` in the log header, strip trailing `/install` for source root).
All `file:line` citations below are relative to the lcls2 checkout root.

---

## Section 1 — The State Machine

**8 states, strictly linear** (`psdaq/psdaq/control/ControlDef.py:21,29` — `states`
and `transitions` lists; as of `lcls2_091826`, numbers shift between releases):

```
reset → unallocated → allocated → connected → configured → starting → paused → running
  0         1             2           3            4           5         6         7
```

**15 transitions** (`ControlDef.py:21,29`):

```
rollcall, alloc, dealloc, connect, disconnect, configure, unconfigure,
beginrun, endrun, beginstep, endstep, enable, disable, slowupdate, reset
```

Transitions are registered on a `Machine` with a `condition_*` guard each
(`psdaq/psdaq/control/control.py:841+`, `add_transition` registration block; as of
`lcls2_091826`). Each transition has exactly one
legal from-state and one legal to-state, which is what makes the chain linear:

| Transition | From → To | Guard |
|---|---|---|
| `reset` | `*` → `reset` | `condition_reset` |
| `rollcall` | `reset`,`unallocated` → `unallocated` | `condition_rollcall` |
| `alloc` | `unallocated` → `allocated` | `condition_alloc` |
| `dealloc` | `allocated` → `unallocated` | `condition_dealloc` |
| `connect` | `allocated` → `connected` | `condition_connect` |
| `disconnect` | `connected` → `allocated` | `condition_disconnect` |
| `configure` | `connected` → `configured` | `condition_configure` |
| `unconfigure` | `configured` → `connected` | `condition_unconfigure` |
| `beginrun` | `configured` → `starting` | `condition_beginrun` |
| `endrun` | `starting` → `configured` | `condition_endrun` |
| `beginstep` | `starting` → `paused` | `condition_beginstep` |
| `endstep` | `paused` → `starting` | `condition_endstep` |
| `enable` | `paused` → `running` | `condition_enable` |
| `disable` | `running` → `paused` | `condition_disable` |
| `slowupdate` | `running` → *(none)* | `condition_slowupdate` |

(Registration block is `add_transition` calls in `control.py:841+`, as of `lcls2_091826`.)

`slowupdate` is an **internal** transition: its destination is `None` and it
deliberately does **not** report status afterward (grep: `"slowupdate is an internal
transition"` in `control.py`). Do not treat a `slowupdate` in the logs as a
user-driven state change; it is the periodic SlowUpdate heartbeat.

### Path arithmetic — do this before opening any log

The routing table `next_dict` (`control.py:281`, as of `lcls2_091826`; grep:
`"next_dict"` to locate in any other release) maps
`(current_state, target_state) → next transition to fire`. Because the chain is
linear, you can compute the whole remaining path yourself by walking the `states`
list in `ControlDef.py:21,29`.

**Worked example.** `daqstate` reports `state: connected`, the operator wants
`running`:

1. Index `connected` = 3, `running` = 7 → moving forward.
2. Remaining transitions, in order: `configure` (3→4), `beginrun` (4→5),
   `beginstep` (5→6), `enable` (6→7).
3. `next_dict['connected']['running']` = `'configure'` → the very next transition
   attempted was `configure`.
4. Therefore: **it died on `configure`.** Nothing about `beginrun`,
   `beginstep`, or `enable` has been attempted yet, and no log line from them
   can exist.

That single deduction eliminates most of the search space before any log is
opened. Going backward works identically: from `running` toward `unallocated`
the path is `disable → endstep → endrun → unconfigure → disconnect → dealloc`
(read off the reverse column of `next_dict`).

**Interpretation:**

- **Report the stall boundary, not the state.** "Stuck in `connected`, target
  `running`, next transition is `configure`, so `configure` is what failed" is
  actionable. "It's in `connected`" is not.
- The *failed* transition names the guard that returned False:
  `configure` → `condition_configure`. That guard's body is where the root cause lives.
- Only look for log evidence from transitions **at or before** the stall point.
  Searching for `enable` errors when the DAQ never left `connected` wastes time.
- A state that regresses (e.g. `running` → `paused` unprompted) means a
  `disable` fired. Check whether an operator did it or whether
  `condition_disable` was triggered by an error path.
- `reset` is reachable from any state, so a jump straight to `reset` is legal and
  tells you nothing about intermediate failures.

---

## Section 2 — Two-Phase Transition Semantics

**This is the single most important diagnostic distinction in this skill.**
Every non-trivial transition runs in two phases, and the two failure messages
mean genuinely different things.

| Message | Source | Meaning |
|---|---|---|
| `'%s did not respond to %s'` | `control.py` (grep: `"did not respond to %s' % (alias, transition)"`) | **Phase 1** — the component never acknowledged at all. Usually the process isn't running. |
| `'%s did not respond to %s phase 2'` | `control.py` (grep: `"did not respond to %s phase 2"`) | **Phase 1 succeeded**, then the component died doing the real work. The process IS running but is wedged. |
| `'%s: %s' % (alias, err_msg)` | `control.py` (grep: `"check_answers"`) | The component **replied with its own error text** (from its `err_info`). This carries the component's own diagnosis — read it literally. |

- **Phase 1** is `confirm_response(...)` inside `condition_common`, over
  `drp | teb | meb`.
- **Phase 2 waits on `drp` + `meb` only** — inside `get_phase2_replies()` in
  `control.py` (grep: `"get_phase2_replies"`): the scope is
  `self.filter_level('drp', ids) | self.filter_level('meb', ids)` — `control.py:1254+`
  as of `lcls2_091826`. A TEB is not a phase-2 participant, so "no TEB phase-2 reply"
  is not a thing.
- `check_answers()` is a *third* failure channel, independent of timeouts: the
  component replied in time but its reply body contained `err_info`. Non-zero error
  count here also fails the transition.

### Timeouts

| Timeout | Default | Where set | Note |
|---|---|---|---|
| Phase-2 timeout `-T` | **12500 ms** | `control.py:2614` (as of `lcls2_091826`) | An older 7500 ms default exists nearby with the note it "must be larger than the EB timeouts, currently at 12 s" |
| Rollcall timeout `--rollcall_timeout` | **30 s** | `control.py:2615` (as of `lcls2_091826`) | Loop re-broadcasts every 1 s |
| `alloc` phase-1 timeout | 5000 ms (hardcoded) | `control.py` (grep: `"timed out\""`) | Not configurable |
| `configure` phase-1 timeout | 60000 ms (hardcoded) | `control.py` (grep: `"condition_configure(): configure phase1 failed"`) | Not configurable |

**Real xpp production uses `-T 40000`, not the 12500 ms default.**
*(verified-against-real-logs: all 118 `CMDLINE:` header lines carrying a `-T`
flag across the September 2026 xpp `control.log` corpus show `-T 40000`.)*

**Interpretation:**

- **Phase-1 failure → hunt a missing or dead process.** The component never
  answered the broadcast at all. Most likely it isn't running: check whether
  its process exists, then read its log's startup header via
  `psana-daq-logs`. *(inferred-from-code-only)*
- **Phase-2 failure → the process is alive; look at what it was doing.** It
  acknowledged phase 1, so its ZMQ path works and it was scheduled. The failure
  is in the actual work (hardware access, configuration application,
  event-builder handshake). *(inferred-from-code-only)*
- **A phase-2 timeout on a system left at the `-T` default may simply be an
  under-set timeout, not a fault.** Check the control process's `CMDLINE:`
  header for its actual `-T` before concluding a component is wedged: 12500 ms
  is only marginally above the 12 s event-builder timeout referenced at
  (the older default is commented out nearby with that note), and production runs use 40000 ms. Say this explicitly
  to the user rather than reporting a fault. *(inferred-from-code-only)*
- **If you see the `'<alias>: <message>'` form (component replied with its own
  error text), quote it verbatim.** That text came from the component itself and
  is higher-quality evidence than any timeout message.
- Phase-2 messages only ever name a `drp` or `meb` alias — or `control` itself,
  which is registered as a `control`-level entry in `cmstate`. *(verified-against-real-logs:
  the only phase-2 non-response alias seen for `configure` in the corpus is `control`.)*

> **Provenance caveat, stated plainly:** the claim that the phase-1/phase-2
> split is *diagnostically important* — that it reliably separates "process
> missing" from "process wedged" — is **inferred-from-code-only.** It has not
> been operationally confirmed against a known-cause incident. This skill is
> organized around that claim, so it is structurally load-bearing. Treat it as
> a strong hypothesis, not a verified rule, and say so when you use it.

---

## Section 3 — Frequency-Ranked Error Catalog

**Provenance of the count column: verified-against-real-logs.**
**Provenance of the "Likely cause" and "Suggested next step" columns:
inferred-from-code-only.**

### Corpus — state this whenever you cite these numbers

```
/sdf/home/x/xppopr/daq/logs/2026/09/*control.log
```

- **One hutch only:** xpp.
- **One month only:** September 2026 (day prefixes present: 01–04, 08–10,
  15–18).
- **120 uncompressed `control.log` files**, 11,048 total lines. 21 additional
  `control.log.zst` rotations exist in the same directory and were **not**
  included in these counts.
- Counts are line counts, anchored to end-of-line where the message is a
  complete line, so a repeated failure across sessions counts once per
  occurrence.

**This does not necessarily generalize to other hutches.** The detector
population (`hsd_0..3`, `epix100_0`, `jungfrau1M_0`, `wav8_*`) is xpp-specific,
so both the ranking and the specific aliases will differ elsewhere. Whether the
*shape* of the distribution holds cross-hutch is an open question that has not
been checked. Do not present these numbers to a user as a general LCLS-II
baseline; present them as "what xpp did in one month."

### Table

| Error string | Observed count | Likely cause *(inferred)* | Suggested next step *(inferred)* |
|---|---|---|---|
| `alloc failed to change state` | 28 | Umbrella result of any `condition_alloc` failure. Grep anchor: `"failed to change state' % key"` in `control.py`. Always accompanied by a more specific line immediately above it. | Do not diagnose this line. Read the 1–5 lines above it in the same `control.log` — those name the actual component or precondition. |
| `1 client did not respond to alloc` | 19 | Exactly one component missing at `alloc`. The count is `len(retlist)`. | Read the line immediately *above* — it names the alias (grep: `"did not respond to alloc' % alias"` in `control.py`). Then check that process via `psana-daq-logs`. |
| `epix100_0 did not respond to alloc` | 11 | The `epix100_0` DRP process is not running or not reachable on the platform. | `showPlatform` to see whether it registered at all; if absent, read its log's startup header. |
| `hsd_2 did not respond to alloc` | 10 | HSD DRP process missing. | As above; but see the HSD note below — do not treat this as a single-card fault. |
| `hsd_3 did not respond to alloc` | 9 | HSD DRP process missing. | As above. |
| `hsd_1 did not respond to alloc` | 8 | HSD DRP process missing. | As above. |
| `hsd_0 did not respond to alloc` | 8 | HSD DRP process missing. | As above. |
| `4 client did not respond to alloc` | 8 | Four components missing at once — the four HSDs, in every observed instance. | Systemic, not per-detector. Check whether the whole HSD DRP process group failed to launch (one host, one job, one launcher). |
| `drp/epix100_0 did not respond to rollcall` | 8 | Warning-level. Required by the activedet file but never answered the 30 s rollcall broadcast. Grep: `"client + ' did not respond to rollcall'"` in `control.py`. | Rollcall still advances the state machine (grep: `"Despite rollcall transition warnings"` in `control.py`), so this is a *precursor*, not the failure. Expect a matching `alloc` failure next. |
| `did not respond to disable phase 2` (all aliases) | 8 | Phase-2 non-response during `disable`. Grep: `"did not respond to %s phase 2"` in `control.py`. Spread across 8 distinct aliases, 1 each. | Phase 1 succeeded — the process is alive. Usually seen during shutdown; check whether the run was being torn down. |
| `configure failed to change state` | 8 | Umbrella for any `condition_configure` failure. Grep: `"failed to change state' % key"` in `control.py`. | Read the lines above: distinguish `configure phase1 failed` (config problem) from `configure phase2 failed` (component wedged). |
| `teb0: TEB didn't hear from:` | 9 | TEB reported, via its own `err_info`, that contributors are missing. | The **following** log line(s) name the missing contributors, one per line. Check each named component. |
| `ami-meb0: MEB didn't hear from:` | 8 | Same, MEB side. | Same — read the following line(s) for the named contributor. |
| `drp/jungfrau1M_0 did not respond to rollcall` | 7 | Warning-level rollcall miss for the Jungfrau DRP. | As with `epix100_0` above. |
| `timing_0 did not respond to connect` | 6 | Timing DRP present at `alloc` but failed the `connect` handshake. Grep: `"did not respond to connect' % alias"` in `control.py`. | It answered `alloc`, so the process exists — look at its log for what happened during `connect`, not for a missing process. |
| `1 client did not respond to connect` | 6 | Companion count line. Every observed instance pairs with `timing_0`. | Read the line above for the alias. |
| `connect failed to change state` | 6 | Umbrella for `condition_connect` failure. Grep: `"failed to change state' % key"` in `control.py`. | Read the lines above. |
| `selectPlatform only permitted in unallocated state` | 6 | Someone ran `selectPlatform` (or the GUI's equivalent) while the DAQ was past `unallocated`. Grep: `"only permitted in unallocated state"` in `control.py`. | Not a DAQ fault — an operator-sequencing error. The DAQ must be deallocated first. Report as procedural, not as a failure. |
| `condition_configure(): configure phase1 failed` | 4 | Phase-1 `configure` failure. Grep: `"condition_configure(): configure phase1 failed"` in `control.py`. Configuration could not be applied/retrieved. | Check `psana-configdb` for a recent change to the implicated device or config alias. Read the per-alias `did not respond to configure` lines above it. |
| `control did not respond to configure phase 2` | 4 | Phase-2 `configure` non-response, attributed to `control` itself. Grep: `"did not respond to %s phase 2"` in `control.py`. | Every observed instance is immediately preceded by `teb0: TEB didn't hear from:` and/or `ami-meb0: MEB didn't hear from:`. Diagnose *those* instead. |
| `configure phase2 failed` | 4 | Umbrella for the phase-2 stage. | See above. |
| `disable failed to change state` | 2 | `condition_disable` failed. Grep: `"failed to change state' % key"` in `control.py`. | Usually shutdown-time; correlate with the `disable phase 2` lines. |
| `jungfrau1M_0 did not respond to alloc` | 2 | Jungfrau DRP process missing at `alloc`. | As with the other per-detector `alloc` misses. |
| `2 client did not respond to alloc` | 1 | Two components missing at `alloc`. | Read the two alias lines above. |
| `dealloc failed to change state` | 1 | `condition_dealloc` failed. Grep: `"failed to change state' % key"` in `control.py`. | Read the `did not respond to dealloc` line above (grep: `"did not respond to dealloc' % alias"` in `control.py`). |
| `wav8_ipm2_0 did not respond to alloc` | 1 | Singleton. | Per-detector `alloc` miss. |
| `timing_0 did not respond to alloc` | 1 | Singleton. | Per-detector `alloc` miss. |
| `teb0 did not respond to alloc` | 1 | Singleton — the TEB itself missing at `alloc`. | Without a TEB nothing downstream works (see Section 4 topology preconditions). |
| `bld_0 did not respond to alloc` | 1 | Singleton. | Per-detector `alloc` miss. |
| `ami-meb0 did not respond to alloc` | 1 | Singleton — the MEB missing at `alloc`. | Monitoring will be unavailable; AMI is unsupported without an MEB (grep: `"ami NOT supported in absence of MEB"` in `control.py`). |

Notably **absent from this corpus entirely (count 0):**
`duplicate alias responded to rollcall`, `at least one DRP is required`,
`at least one TEB is required`, `ami NOT supported in absence of MEB`. Those
messages exist in the source (Section 4) but were never triggered in this
sample. Do not tell a user they are common.

### The HSD pattern — read this before blaming a card

`hsd_0` (8), `hsd_1` (8), `hsd_2` (10), `hsd_3` (9) all appear at
**comparable frequency**, and `4 client did not respond to alloc` (8) fires in
lockstep with them. In the corpus, the four HSD `alloc` misses almost always
occur in the same log line group at the same timestamp.

**Interpretation:** this is a **systemic HSD-group failure signature — one
launcher, one host, or one process group failing to come up — not a single bad
card.** *(inferred-from-code-only; the frequency pattern itself is
verified-against-real-logs.)* If you see one `hsd_N` miss, immediately check
whether the other three are also in the log at the same timestamp before
recommending anything card-specific.

**Interpretation of the table as a whole:**

- **`alloc` dominates.** The `unallocated → allocated` boundary is where the
  DAQ most often fails to start in this corpus, and the cause is almost always
  a component that isn't there. Start there when the user says "the DAQ won't
  start."
- **`X failed to change state` is never the diagnosis.** It is the umbrella
  result of a guard returning False. Always read upward for
  the specific line.
- **The `N client did not respond to <t>` count line follows the per-alias
  lines**, so the count tells you *how many* and the preceding lines tell you
  *which*. Read both.
- **`TEB/MEB didn't hear from:` is a two-line finding.** The header line says a
  contributor is missing; the *next* line(s) name it. Do not report the header
  alone.
- **`selectPlatform only permitted in unallocated state` is not a fault.**
  Classify it as an operator-sequencing issue and say so.

---

## Section 4 — Other Error Message Templates

There are ~60 `report_error` / `report_warning` call sites in
`psdaq/psdaq/control/control.py` (60 by direct count, excluding the two
definitions and callback-argument passing). The grouping below *is* the diagnostic
value — a message's group tells you which layer failed.

### Stuck transition (umbrella)

- `control.py` (grep: `"failed to change state' % key"`) —
  `trigError = '%s failed to change state' % key`

Fires when `stateChange` was requested but `self.state == stateBefore`
afterwards, i.e. a `condition_*` guard returned False. Never the root cause.

### Per-transition phase-1 non-response

| Transition | Per-alias grep anchor | Count line grep anchor |
|---|---|---|
| `alloc` | `"did not respond to alloc' % alias"` in `control.py` | `"did not respond to alloc"` count context |
| `dealloc` | `"did not respond to dealloc' % alias"` in `control.py` | follows per-alias line |
| `connect` | `"did not respond to connect' % alias"` in `control.py` | follows per-alias line |
| `disconnect` | `"did not respond to disconnect' % alias"` in `control.py` | follows per-alias line |
| generic (`condition_common`, all others incl. `configure`) | `"did not respond to %s' % (alias, transition)"` in `control.py` | follows per-alias line |

### Rollcall

- `control.py` (grep: `"NOT selected for data collection"`) —
  `'rollcall: %s NOT selected for data collection'` (**warning**). Emitted for
  a newly-found component that is *not* in the activedet file, which then
  defaults to `active = 0`. This is the single highest-volume warning in the
  corpus (450 lines) and is **normal** for detectors intentionally left out.
- `control.py` (grep: `"duplicate alias responded to rollcall"`) —
  `'duplicate alias responded to rollcall: %s'`. Two processes claim the same
  alias (`check_for_dups`).
- `control.py` (grep: `"client + ' did not respond to rollcall'"`) —
  `client + ' did not respond to rollcall'` (**warning**). Required by the
  activedet file but silent for the whole timeout.
- Timeout: **30 s** default (`--rollcall_timeout`, `control.py:2615` as of
  `lcls2_091826`). The loop **re-broadcasts the rollcall message every 1 s**
  until the deadline.
- **Rollcall failures do not stop the state machine.** The guard sets `retval = True`
  despite missing clients, with the in-source comment "Despite rollcall transition
  warnings, allow state machine to advance." (grep: `"Despite rollcall transition
  warnings"` in `control.py`).

**Interpretation:** a rollcall warning is a **leading indicator**, not the
failure. The same component will usually reappear as an `alloc` phase-1
non-response a moment later. Report the pair together; the rollcall line is
often the earlier, cleaner timestamp for the same root cause.
*(inferred-from-code-only, though the corpus is consistent with it: every
alias with rollcall misses also has `alloc` misses.)*

### Topology preconditions — nothing downstream works until satisfied

These fail `alloc` before any other work can happen. Flag them as blocking.

- `control.py` (grep: `"at least one DRP is required"`) — `'at least one DRP is required'`
- `control.py` (grep: `"must use readout group"`) — `f'at least one DRP must use readout group {self.platform}'`
- `control.py` (grep: `"at least one TEB is required"`) — `'at least one TEB is required'`
- `control.py` (grep: `"ami NOT supported in absence of MEB"`) — `'ami NOT supported in absence of MEB'`
  (**warning** — the DAQ proceeds, but AMI monitoring will not work)

**Interpretation:** these are *configuration/selection* faults, not process
faults. No component is broken; the wrong set was selected. Point the user at
the activedet file (Section 5) and at `showPlatform`'s active-flag column, not
at any process's log. *(inferred-from-code-only.)* None of these appeared in
the corpus.

### DRP alias grammar

- `control.py` (grep: `"is missing _N suffix"`) — `f'drp id {unique_id} is missing _N suffix'`
- `control.py` (grep: `"has malformed _N suffix"`) — `f'drp id {unique_id} has malformed _N suffix'`

DRP aliases must end `_<N>`; the detector name is the alias with that suffix
stripped (`detector_name()` in `control.py`). A bad alias is a launch argument error
(`-u` on the DRP process) — check the offending process's `CMDLINE:` header via
`psana-daq-logs`. *(inferred-from-code-only.)*

### Active detectors file

- `control.py` (grep: `"Missing active detectors file"`) — `'Missing active detectors file %s'`
- `control.py` — `'active detectors file %s not found'`
- `control.py` (grep: `"active detectors file %s is empty"`) — `'active detectors file %s is empty'`
- `control.py` (grep: `"Missing \"activedet\" key"`) — `'Missing "activedet" key in active detectors file %s'`
- `control.py` — `'Failed to read configuration from active detectors file %s'`
- `control.py` (grep: `"is not a proper active detectors file"`) —
  `'/dev/null is not a proper active detectors file'` (**warning**)

All of these route to Section 5. They are file-content faults: the DAQ cannot
know which components to require.

### XPM PV puts

- `control.py` (grep: `"condition_alloc() failed putting"`) —
  `f'condition_alloc() failed putting {groups} to PV {self.pvListL0Groups}'`
- `control.py` (grep: `"timed out\""`) —
  `f"self.ctxt.put({pvName}, {val}) timed out"`

**Interpretation:** an EPICS/PVA write to the XPM failed. This is neither a DRP
problem nor a config-database problem — it is control-plane reachability to the
XPM. Suspect the XPM IOC or the PVA path, and note that `condition_alloc` will
report `alloc failed to change state` as a consequence.
*(inferred-from-code-only.)*

### State guards

- `control.py` (grep: `"deallocate first"`) —
  `'cannot change bypass_activedet setting in state \'%s\' -- deallocate first'`
  (permitted only in `reset` or `unallocated`)
- `control.py` (grep: `"only permitted in unallocated state"`) —
  `'selectPlatform only permitted in unallocated state'`

**Interpretation:** both are operator-sequencing errors, not faults. The
remediation is "deallocate first, then retry" — and the **human** does that.

---

## Section 5 — The `activedet.json` Artifact

Mentioned by no other DAQ skill. It is the file that decides **which components
the DAQ requires and which readout group each one belongs to** — so it is the
direct answer to "why isn't detector X being recorded?"

### Path

Built by `control.py` as:

```
~<hutch>opr/.psdaq/x<XPM_MASTER>_p<PLATFORM>.activedet.json
```

from `homedir = os.path.expanduser('~')` plus
`'%s/.psdaq/x%d_p%d.activedet.json' % (homedir, self.xpm_master, self.platform)`.
Overridable with the control process's `-r` flag. The active path in use is logged as
`active detectors file: <path>` — **read that line from `control.log` rather than
reconstructing the path**, since `-r` may override it.
*(verified-against-real-logs: this `<I>` line appears once per control session
in the corpus.)*

**`/dev/null` special case**: if the filename is `/dev/null`, `bypass_activedet` is
set True, a warning is logged, and **all components default to active with readout
group = platform**. If you see the bypass warning, the activedet file is irrelevant
to that session — say so instead of reading a file.

### Structure — **verified-live**

Files confirmed present: **xpp 10, rix 9, tmo 9** (also mfx 2, txi 10, ued 3).
A real file inspected directly, `/sdf/home/x/xppopr/.psdaq/x4_p0.activedet.json`
(the currently-in-use one for xpp platform 0, XPM master 4):

```
top-level keys:  ['activedet', 'history']
activedet    ->  ['drp', 'meb', 'teb']
history      ->  ['drp', 'tpr']
```

```json
{
  "activedet": {
    "drp": {
      "alvium_1_0": { "active": 1, "det_info": { "readout": 0 } },
      "bld_0":      { "active": 0, "det_info": { "readout": 0 } },
      "epix100_0":  { "active": 1, "det_info": { "readout": 0 } }
    },
    "meb": { },
    "teb": { }
  },
  "history": {
    "drp": { "alvium_0": { "det_info": { "readout": 0 } } },
    "tpr": { }
  }
}
```

(Shape is verbatim from the real file; alias set abbreviated. The `history.drp`
map in that file holds **22** entries — many more than `activedet.drp` — i.e.
every DRP alias ever seen on this platform.)

`/sdf/home/x/xppopr/.psdaq/p0.activedet.json` also exists with the same
top-level shape (`['activedet', 'history']`, `activedet` → `drp`/`meb`/`teb`,
`history` → `drp`) but is **not** the file the current control process reads —
it lacks the `x<XPM>_` prefix the code builds. Do not confuse the two; always
confirm against the `active detectors file:` log line.

### How the two keys are used

- **`activedet`** is read by `get_active_and_inactive()` in `control.py`, which splits
  it into `active_set` / `inactive_set`. A component in `active_set` is **required**
  — its absence at rollcall produces the `did not respond to rollcall` warning
  (grep: `"client + ' did not respond to rollcall'"` in `control.py`), and its
  absence at `alloc` produces the `did not respond to alloc` error (grep:
  `"did not respond to alloc' % alias"` in `control.py`). A component present on the
  platform but absent from `activedet` gets `active = 0` and the
  `NOT selected for data collection` warning.
  Readout group comes from `det_info.readout`.
- **`history`** is read nearby (grep: `"history"` context near `get_active_and_inactive`
  in `control.py`; `history['drp']` / `history['tpr']` are defaulted to empty dicts
  if absent). It records the last-known readout group for **every** DRP/TPR alias
  ever seen, and is consulted as a fallback when a newly-found detector has no
  `activedet` entry.

### Answering the two questions

1. **"Why isn't detector X being recorded?"** → look up `X` in
   `activedet.drp`. Missing entirely, or `"active": 0`, means it was
   deliberately deselected — the DAQ is behaving as configured. Cross-check
   against the `rollcall: drp/X NOT selected for data collection` warning
   (the `NOT selected for data collection` warning), which is the log-side confirmation.
2. **"What readout group should X be in?"** → `activedet.drp.X.det_info.readout`
   is the current value; `history.drp.X.det_info.readout` is the
   **prior-known-good** value. A mismatch, or a current value that differs from
   its peers, is the thing to report. This is the only place in the DAQ that
   preserves a previous-good group value to compare against.

**Interpretation:**

- Read this file **before** blaming a process. A "missing" detector that is
  simply `"active": 0` is not a failure at all, and this distinction is not
  visible from metrics or from a process log. *(inferred-from-code-only for the
  diagnostic ordering; the file structure is verified-live.)*
- The `activedet` errors in Section 4 (`Missing active detectors file`, `is
  empty`, `Missing "activedet" key`) all mean the DAQ cannot determine its
  required component set. Expect *widespread* downstream `alloc` errors from a
  single file-level fault — do not chase each component separately.
- `showPlatform`'s `*` (active) marker and readout-group line are the live view
  of what this file produced; comparing the two is how you confirm the file was
  actually applied. *(inferred-from-code-only.)*
- **Never edit this file.** Recommend the change and the exact value; the human
  applies it (normally via the control GUI or `selectPlatform`, both of which
  are off-limits to you).

---

## Section 6 — Live Query Tools

**The property that makes these valuable: they need no Grafana, no MCP server,
and no ConfigDB reachability.** They talk directly to the `control` process
over ZMQ, so they work when every other diagnostic angle is down. When
`psana-daq-monitor` reports "no metrics" and `psana-configdb` is unreachable,
these still answer "is it running, and where did it stop?"

### `daqstate` — `psdaq/psdaq/control/daqstate.py`

Verified flags (argument definitions in `daqstate.py`):

| Flag | Default | Meaning | Read-only? |
|---|---|---|---|
| `-p PLATFORM` | `0` (choices 0–7) | Platform | — |
| `-P INSTRUMENT` | **required** | Hutch name; verified against the live control process and exits on mismatch | — |
| `-C COLLECT_HOST` | `localhost` | Collection host | — |
| `-t TIMEOUT` | `10000` ms | Request timeout | — |
| `--phase1 JSON` | `None` | phase1Info; only meaningful with `--state` | — |
| `--state <s>` | — | **Sets** state (choices = `ControlDef.states`) | **NO — mutating** |
| `--transition <t>` | — | **Fires** a transition (choices = `ControlDef.transitions`) | **NO — mutating** |
| `--monitor` | — | Streams status/error/warning/progress/step events | **YES** |
| `--config ALIAS` | — | **Sets** config alias | **NO — mutating** |
| `--record {0,1}` | — | **Sets** recording flag | **NO — mutating** |
| `--bypass {0,1}` | — | **Sets** activedet bypass | **NO — mutating** |
| `-B` | — | Shortcut for `--config BEAM` | **NO — mutating** |

`--state`, `--transition`, `--monitor`, `--config`, `--record`, `--bypass`, and
`-B` are in one **mutually exclusive group** — you cannot combine them.

**With no flag at all**, `daqstate` prints current status via `getStatus()`:

```
daqstate -P xpp -p 0 -C <collect_host>
```
→ `last transition: <t>  state: <s>  configuration alias: <c>  recording: <r>
bypass_activedet: <b>  experiment_name: <e>  run_number: <n>
last_run_number: <m>`

**This bare invocation is your primary entry point.** It gives you the current
state (for Section 1 path arithmetic), the last transition attempted, and
whether `bypass_activedet` is on (which determines whether Section 5 applies).

`--monitor` is the read-only way to watch a transition attempt live; it prints
`error:` / `warning:` / `progress:` / `step_done:` / `data file:` events as the
control process publishes them.

### `showPlatform` — `psdaq/psdaq/control/showPlatform.py`

| Flag | Default | Meaning |
|---|---|---|
| `-p PLATFORM` | `0` (0–7) | Platform |
| `-C COLLECT_HOST` | `localhost` | Collection host |
| `-t TIMEOUT` | `2000` ms | Request timeout |
| `-v` | off | Also pretty-print the raw `getPlatform()` reply |
| `--json` | off | Print `getJsonConfig()` — activedet-format configuration |

Default output tabulates every registered process as
`alias  level/pid/host (* = active)`, and for **active DRPs additionally prints
its readout group** on a continuation line. Levels seen: `control`, `drp`, `teb`, `meb`,
`tpr`.

**Interpretation:**

- **A component absent from `showPlatform` is not registered at all** — that is
  the strongest possible confirmation of a phase-1 non-response cause. Present
  but without `*` means it registered and was *deselected*, which is a Section 5
  question, not a process question. *(inferred-from-code-only.)*
- The per-DRP readout group here is the **live, applied** value. Compare it
  against `activedet.json` to confirm the file took effect.
- `--json` gives you the activedet-format view without reading the file, which
  is useful when you cannot resolve the `~<hutch>opr` home directory.

### `selectPlatform` — `psdaq/psdaq/control/selectPlatform.py`

Flags: `-p` platform (0–7, default 0), `-C COLLECT_HOST` (default `localhost`),
`-t TIMEOUT` (default 2000), `-R READOUT_GROUP` (0–7, default = platform),
`-s SELECT` (repeatable), `--select-all`, `-u UNSELECT` (repeatable).

**Mutating — do not run it.** It is the source of the
`selectPlatform only permitted in unallocated state` error (grep:
`"only permitted in unallocated state"` in `control.py`), which is why that
message appears in Section 3. You may recommend an exact `selectPlatform` command
for the human to run, and you must note that the DAQ has to be in `unallocated` first.

### `DaqControl` Python API — `psdaq/psdaq/control/DaqControl.py`

| Method | Read or mutate |
|---|---|
| `getState` | read |
| `getPlatform` | read |
| `getJsonConfig` | read |
| `getInstrument` | read |
| `getStatus` | read |
| `monitorStatus` | read |
| `getBlock` | read |
| `storeJsonConfig` | **mutate** |
| `selectPlatform` | **mutate** |
| `setState` | **mutate** |
| `setConfig` | **mutate** |
| `setRecord` | **mutate** |
| `setBypass` | **mutate** |
| `setTransition` | **mutate** |

Constructor is keyword-only: `DaqControl(host=..., platform=..., timeout=...)`
(grep: `"def __init__"` in `DaqControl.py`).

**Restating the read-only posture concretely:** use only the `get*` and
`monitor*` methods, and only the flagless / `--monitor` forms of `daqstate`
plus `showPlatform`. If your diagnosis implies a state change, output the exact
command you would run and let the human run it. Never invoke `set*`,
`selectPlatform`, or `storeJsonConfig`.

---

## Section 7 — Scan / Step Content

Steps are **just more transitions on the same state machine**, which is why
they live in this skill rather than a separate one. A "step" is one
`beginstep → (running) → endstep` cycle within a run:

```
configured ──beginrun──> starting ──beginstep──> paused ──enable──> running
                             ^                                          |
                             └──────────── endstep ◄──disable───────────┘
```

(Transition registrations are in the `add_transition` block in `control.py:841+`, as of `lcls2_091826`.)

### Scan drivers

| Module | Role |
|---|---|
| `psdaq/psdaq/control/ConfigScan.py` | Configuration scan — drives steps and writes `STEPINFO` data |
| `psdaq/psdaq/control/BlueskyScan.py` | Bluesky-integrated scan; tracks `step_value`, writes `STEPINFO` |
| `psdaq/psdaq/control/TimedRun.py` | Fixed-duration run, with `set_connected_state()` / `set_running_state()` helpers |

### Step plumbing

- **`step_value` / `STEP_VALUE`** — `ControlDef.STEP_VALUE = 'step_value'`
  (`ControlDef.py:45`, as of `lcls2_091826`), documented in-source as "name of
  simulated motor reserved for step value". `BlueskyScan` detects a motor named
  `STEP_VALUE` and overrides its own counter from that motor's position, otherwise
  increments internally and resets to 1 at scan end. The value is recorded into the
  step data as `step_value` and into phase1 as `step`.
- **`step_done` flag** — an `Event` in both the control process and the scan drivers.
  The control process runs a dedicated `step_done_func` thread that monitors the XPM's
  `StepDone` PV (grep: `"StepDone"` in `control.py`). The callback only honors the PV
  in state `running` or `paused` and *ignores it otherwise*, logging
  `StepDone PV=... in state ... (ignore)` — note the in-source comment
  "There is a race between self.state=running and stepdone". On acceptance it
  publishes `step_msg(1)` over ZMQ; scan drivers wait on their own `step_done` event.
- **`step_pub_port`** = `PORT_BASE + platform + 50` (`ControlDef.py:126-127`,
  as of `lcls2_091826`). `daqstate --monitor` surfaces the same events as
  `step_done: <n>`.
- **`CHUNKINFO = 252`, `STEPINFO = 253`** (`ControlDef.py:40-41`, as of
  `lcls2_091826`; both commented `# psdaq/drp/drp.hh`). These are `namesid`
  values for the step/chunk metadata blocks written by the scan drivers.
- `phase1Info['beginstep']['step_values']` is defaulted to `{}` if absent
  (grep: `"step_values"` in `ConfigScan.py`).

**Interpretation:**

- **A scan stuck between steps is a state-machine stall like any other.** Run
  the bare `daqstate` and apply Section 1: `paused` with target `running` means
  `enable` failed; `starting` with target `paused` means `beginstep` failed.
- **A scan that never advances past step 1 with the DAQ sitting in `running` is
  a `step_done` problem, not a transition problem.** The `StepDone` PV either
  isn't firing or arrived in the wrong state and was ignored.
  Check the control process's debug log for `StepDone PV=... (ignore)`.
  *(inferred-from-code-only.)*
- `beginstep` / `endstep` phase-1 failures are logged via the same `did not
  respond to <transition>` template as all other transitions, following the same
  two-phase rules as Section 2 and yielding the same `'%s failed to change state'`
  umbrella.
- Because `slowupdate` is internal and reports no status (grep: `"slowupdate is an
  internal transition"` in `control.py`), it will not appear as a step boundary.
  Do not read it as one.

---

## Section 8 — Handoffs

State the handoff explicitly and invoke the sibling skill by name via the
`skill` tool before using its techniques — do not guess at log paths, PromQL,
or ConfigDB URLs from this skill's summary.

| Finding | Hand off to | What to ask it |
|---|---|---|
| Component down / not responding (**phase 1**) — `X did not respond to <t>`, or absent from `showPlatform` | `psana-daq-logs` | Read `X`'s log startup header (`# CMDLINE:`, `# HOST:`, `# SLURM_JOB_ID:`) and grep `<[EC]>` in the same session prefix |
| `configure` **phase-1** failure — `condition_configure(): configure phase1 failed` | `psana-configdb` | Was there a recent configuration change to the implicated device or config alias? |
| Deadtime / damage / buffer symptoms after the DAQ *is* running | `psana-daq-monitor` | Event rates, `DeadFrac`, `DRP_Damage`, buffer occupancy for the implicated detector |
| Phase-2 failure — `X did not respond to <t> phase 2` | `psana-daq-logs` **first**, then `psana-daq-monitor` | The process is alive, so read what it logged during the transition; if it was mid-run, check whether metrics show it falling behind |
| `TEB/MEB didn't hear from: <alias>` | `psana-daq-logs` for `<alias>`, then `psana-daq-monitor` | `<alias>` is a missing event-builder contributor; check its process, then `EB_FxUpCt` / `EB_CbMsMk` |
| Vague report with no state information yet | `psana-daq` (the router) | Let it pick the angle; come back here if the answer is "it won't start" |

**Do not duplicate sibling content.** Log path conventions, filename grammar,
the `zstdcat` requirement for `.log.zst`, and the header-block format are owned
by `psana-daq-logs`. Metric names and thresholds are owned by
`psana-daq-monitor`. ConfigDB endpoints are owned by `psana-configdb`.

Out of scope entirely, per the router's scope table: psana2 analysis API
(`ask-lcls2`), timing-sequence authoring and rate arithmetic (`xpm-seq`),
generic Slurm/EPICS questions (`ask-slurm-s3df` / `ask-epics`), AMI-side
performance (`ami-performance-monitor`).

---

## Appendix — ZMQ Port Arithmetic

Useful when a component is reachable-but-silent and someone needs to check the
right socket. All from `psdaq/psdaq/control/ControlDef.py`, with `p` = platform
(0–7). Structural constants — `PORT_BASE = 29980` (`ControlDef.py:42`, as of
`lcls2_091826`) and `CHUNKINFO`/`STEPINFO` (`ControlDef.py:40-41`) are kept as
line citations because the *values* are the facts:

| Socket | Formula | Note |
|---|---|---|
| `PORT_BASE` | **29980** | `ControlDef.py:42` as of `lcls2_091826` |
| `back_pull_port` | `PORT_BASE + p` | `ControlDef.py` (grep: `"back_pull_port"`) |
| `back_pub_port` | `PORT_BASE + p + 10` | same |
| `front_rep_port` | `PORT_BASE + p + 20` | same |
| `front_pub_port` | `PORT_BASE + p + 30` | same |
| `fast_rep_port` | `PORT_BASE + p + 40` | same |
| `step_pub_port` | `PORT_BASE + p + 50` | `ControlDef.py:126-127` as of `lcls2_091826` |
| `scan_pull_port` | `PORT_BASE + p + 60` | same |
| `xpm_pull_port` | `PORT_BASE + xpm + 70` | same |

`xpm_pull_port(xpm_name)` does **not** take a platform. It extracts an integer
from the XPM name and **adds 16 when the name's second colon-field is `FEH`**,
because NEH and FEH share a host. So `...:NEH:3` → 29980 + 3 + 70 = 30053,
while `...:FEH:3` → 29980 + 19 + 70 = 30069. (Grep: `"FEH"` in `ControlDef.py`
to locate the logic in any release.)

For platform 0 the concrete set is: 29980 (back_pull), 29990 (back_pub),
30000 (front_rep), 30010 (front_pub), 30020 (fast_rep), 30030 (step_pub),
30040 (scan_pull).

**Interpretation:** `back_*` are the control↔component channels (a component
that never answers a broadcast has a `back_pull`/`back_pub` problem);
`front_*` are the control↔client channels (`daqstate`/GUI); `fast_rep` serves
the fast-reply thread; `step_pub` is the step-done publisher.
*(inferred-from-code-only.)* Note `scan_pull_port` is defined but has no in-tree
caller — do not assume a listener exists on it.

---

## File-to-Role Reference

| File | Role |
|---|---|
| `psdaq/psdaq/control/control.py` | Collection manager: state machine, transition guards, error reporting (~2600 lines) |
| `psdaq/psdaq/control/ControlDef.py` | Constants: states, transitions, CHUNKINFO, STEPINFO, PORT_BASE |
| `psdaq/psdaq/control/DaqControl.py` | Python API: getState, getStatus, setState, setTransition, etc. |
| `psdaq/psdaq/control/daqstate.py` | CLI: read-only state/status queries |
| `psdaq/psdaq/control/showPlatform.py` | CLI: tabulate registered processes and readout groups |
| `psdaq/psdaq/control/selectPlatform.py` | CLI: platform selection (unallocated state only) |
| `psdaq/psdaq/control/ConfigScan.py` | Scan/step acquisition |
| `psdaq/psdaq/control/BlueskyScan.py` | Bluesky scan integration |
