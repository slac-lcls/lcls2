---
name: psana-daq
description: Entry point for diagnosing LCLS-II DAQ problems end-to-end. Use when the user reports a general DAQ issue without a specific angle yet — e.g. "the DAQ is broken", "we're losing data", "high deadtime", "damage events", "DAQ won't start" — and you need to figure out which diagnostic angle (state/control, metrics, logs, or configuration) to pursue. Routes to psana-daq-control (run-control state machine and transition failures), psana-daq-monitor (Grafana/Prometheus metrics), psana-daq-logs (live raw DAQ log files), and psana-configdb (read-only detector/system configuration lookups).
---

# Skill: psana-daq

# LCLS-II DAQ Diagnosis Router

You are the entry point for diagnosing LCLS-II DAQ (Data Acquisition) problems.
Route to relevant leaf skills; for a full session/window report load
[psana-daq-snapshot](../psana-daq-snapshot/SKILL.md). Reuse established scope
and evidence, and load only the angles needed for the current question.

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

Reuse the supplied hutch, live/historical mode, time window and zone,
launch/run identities, release and available evidence. For a hutch-wide window,
include relevant launches without dividing the report by platform/partition.
Discover configuration only when needed for a particular query. Historical
work starts from retained logs/run evidence; today's config cannot establish
which configuration was launched then.

### Config discovery

A conventional configuration location is `~<hutch>opr/daq/scripts/*.py`;
prefer a supplied path or launch evidence. Where applicable, list candidates:

    ls ~<hutch>opr/daq/scripts/*.py

Two tiers exist:

- **Leaf endstation configs** (e.g. `3rix.py`, `crix.py`) — contain `from <base> import*` and
  a `config.select([...])` call. These are what operators launch.
- **Base configs** (e.g. `rix.py`) — define `procmgr_config =` and hold the identity
  assignments at the top. These are what you read for `platform`/`collect_host`/`hutch`.

Use a supplied config identity or corroborate it with retained launch commands
and headers. If ambiguity matters for a live query, ask; a historical log report
can proceed with identity marked unknown. A stale launcher artifact or file
mtime alone does not identify the running configuration.

### Identity extraction

Read configuration Python **textually, never import it**: it is executable code
and can have side effects. Follow relevant base-config imports textually, and
extract `platform`, `collect_host`, `hutch`, and optional `station` only where
unambiguous. If `station` is absent, the control CLI defaults it to platform;
verify against that release's `control.py`. Record syntax errors and unresolved
computed values rather than executing the config to obtain them.

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

### Handing off to leaf skills

Once `hutch` is established here, **state it explicitly when invoking a leaf
skill** — e.g. "Loading psana-daq-monitor for hutch=rix" — rather than relying
on the leaf skill to re-derive or re-ask for it. `hutch` and `instrument` are
the same identifier: Prometheus's `instrument` label is set directly from a
component's `--hutch` argument (`psdaq/psdaq/cas/epics_exporter.py:29-51`,
`self._hutch = hutch` fed into `g.add_metric([self._hutch, self._id], ...)`
under the `instrument` label), so no translation is needed between this
router's `hutch` and `psana-daq-monitor`'s `instrument` label.

Pass all established context and coverage limits to the leaf, not only hutch.
A dependency unavailable in a partial installation should be reported as such;
use retained evidence and available skills without guessing missing procedures.

---

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

For an unscoped **live** problem, check state through `psana-daq-control` when
reachable. If startup/transition failure is indicated, read the same launch's
control and participant logs next. If running but degraded, available metrics
can locate affected components/windows, then correlate logs and configuration.
If state is unavailable, use retained logs and say what cannot be checked.

For **historical** work, reconstruct state from logs/transition evidence and
use the supplied window. Never branch on today's state. Metrics are optional:
control timestamps, run metadata or supplied scope can establish a window.
Some older C++ logs have no timestamps; a metrics window cannot timestamp those
lines. Keep their association approximate and based on launch/transition
context. Check the deployed log format before attempting time filtering.

For a narrow question, load the relevant skill directly; do not repeat the
whole sweep or service preflight.

## Release-source navigation (cross-cutting)

When any skill needs to read source ground truth — to look up a message, verify
a timeout, or check a guard — use this technique to locate the exact source tree
the running DAQ was built from.

1. Read `TESTRELDIR`, `GIT_DESCRIBE` (if present), command and environment
   metadata from the relevant process headers. `TESTRELDIR` commonly ends in
   `/install`; check its parent for source rather than assuming it exists.
2. Record the release/build identity for each relevant launch. A directory name
   or today's checkout alone does not establish the deployed revision; a dirty
   build can differ from its commit. Header fields depend on launcher version.
3. Prefer the matching readable source; use message/function anchors because
   line numbers drift. If git refuses ownership checks, plain source reads may
   still work; do not change trust settings as part of diagnosis.
4. With multiple releases, map evidence to each source tree. If matching source
   is unavailable, label the mismatch and limit claims to what was inspected.
   Ask only when resolving it is necessary to answer the question.

Leaf skills use this section as the shared release/source lookup procedure.

## Prerequisites / preflight

Check only services needed by the requested investigation; supplied historical
evidence does not require live-service preflight.

- **State/control:** requires the matching DAQ environment and a reachable
  control process over ZMQ, or retained logs for reconstruction. It does not
  require Grafana or ConfigDB. Use the control skill's read-only instructions.
- **Logs:** check the supplied or launcher-configured root using the logs
  skill's **Path convention**. No static hutch-availability list is authoritative.
- **Metrics:** requires available Grafana tools and retained data for the
  requested interval. Missing series are not zero event rates.
- **ConfigDB:** requires successful HTTP and JSON application responses for
  the needed read endpoint. Reachability alone does not prove historical
  content/key retrieval is supported.

Report unavailable/partial legs and continue those supported by evidence.

## AMI / DAQ ownership seam

This boundary is easy to get wrong in both directions. The AMI-spawned agent runs
alongside AMI, and **AMI is itself a shared-memory client of the MEB**:

- **`ami-performance-monitor`** (optional external skill, if installed) owns AMI-side symptoms — graph latency, worker
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
