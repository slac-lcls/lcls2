---
name: psana-daq
description: Entry point for diagnosing LCLS-II DAQ problems end-to-end. Use when the user reports a general DAQ issue without a specific angle yet — e.g. "the DAQ is broken", "we're losing data", "high deadtime", "damage events", "DAQ won't start" — and you need to figure out which diagnostic angle (state/control, metrics, logs, or configuration) to pursue. Routes to psana-daq-control (run-control state machine and transition failures), psana-daq-monitor (Grafana/Prometheus metrics), psana-daq-logs (live raw DAQ log files), and psana-configdb (read-only detector/system configuration lookups).
---

# Skill: psana-daq

# LCLS-II DAQ Diagnosis Router

You are the entry point for diagnosing LCLS-II DAQ (Data Acquisition) problems.
This skill does not itself query anything — it decides which of four sibling
skills to load, in what order, based on what the user has told you so far.

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
| vague report, e.g. "the DAQ is broken" / "we're losing data" with no other detail | use the branch-on-state workflow below |

## Branch-on-state workflow for vague reports

Check DAQ **state first, always** — it tells you whether the DAQ reached
running at all, which determines every subsequent step.

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

This order is a recommendation, not a requirement. If the user's question
already targets one angle specifically (e.g. "what's the deadtime right
now?"), load that skill directly.

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
in the router; leaf skills point back to it rather than duplicating it.

## Prerequisites / preflight

Before routing, check what's actually usable in the current environment and tell
the user upfront if an angle is unavailable.

- **State/control (`psana-daq-control`)**: works with no external dependencies —
  talks directly to the `control` process over ZMQ. Always available if the DAQ
  host is reachable.
- **Logs (`psana-daq-logs`)**: requires a readable DAQ log directory for the
  relevant hutch. The path convention and the list of hutches confirmed
  present/absent are documented in `psana-daq-logs/SKILL.md`:

  | Status | Hutches |
  |---|---|
  | Present (readable) | xpp, tmo, rix, mfx, txi, ued, det |
  | Absent | xcs, cxi, asc, tst |

  Check with `ls`/`test -d` before assuming it exists; if absent, say so rather
  than guessing an alternate path.
- **ConfigDB (`psana-configdb`)**: requires reachability to
  `pswww.slac.stanford.edu`. A simple check:

      curl -sf -o /dev/null -w '%{http_code}' \
        https://pswww.slac.stanford.edu/ws/configdb/ws/configDB/get_hutches/

  A `200` means the service is reachable and this angle is usable.

If one or more angles are unavailable, tell the user explicitly which diagnostic
angles you can and cannot pursue before proceeding with the ones that remain.

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

This skill intentionally has no PromQL, no log grep patterns, and no
ConfigDB endpoint URLs — those live in the leaf skills so they stay
authoritative in one place. Load the relevant sibling skill before doing any
real diagnostic work.
