# Control Error Message Templates and ZMQ Port Arithmetic

Full enumeration of `control.py`'s `report_error`/`report_warning` call
sites (grouped by layer), plus the ZMQ port-arithmetic appendix. `SKILL.md`
keeps only the durable per-group interpretation rules; this file holds the
message-by-message grep anchors and templates those rules are built on.

No frequency/count table is kept here — a hand-counted snapshot inside a
static reference file decays the moment the log corpus it was derived from
keeps growing (this bit an earlier version of this skill twice). If
frequency context is ever needed for a specific hutch/month, re-derive it
directly against that corpus rather than trusting a number from this file.

---

## Other Error Message Templates

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

**Interpretation:** these are usually *configuration/selection* faults, not
process faults — the wrong set was selected rather than a component being
broken. Point the user at the activedet file (Section 4 in `SKILL.md`) and
`control.log`'s rollcall lines — `rollcall: <alias> selected for data
collection (readout group N)` (logged at INFO, `control.py:1928`; a
variant without the group number is at `control.py:1930`) and `rollcall:
<alias> NOT selected for data collection` (logged as a warning,
`control.py:1901`) — as the first check. But a selected component that
*also* failed to launch produces the same symptom — don't rule that out
just because the topology precondition looks like the simpler
explanation. *(inferred-from-code-only.)*

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

All of these route to Section 4 (in `SKILL.md`). They are file-content
faults: the DAQ cannot know which components to require.

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
`front_*` are the control↔client channels (GUI and scripts); `fast_rep` serves
the fast-reply thread; `step_pub` is the step-done publisher.
*(inferred-from-code-only.)* Note `scan_pull_port` is defined but has no in-tree
caller — do not assume a listener exists on it.
