---
name: psana-daq-monitor
description: Query Grafana/Prometheus DAQ metrics for LCLS-II over a hutch, platform, and time window (DRP event rates, deadtime, damage, DRP errors, buffers, file writing, event builder, timing/hardware links, MEB monitoring). Use when the user's question already targets Grafana/Prometheus metrics specifically, or after psana-daq has narrowed a general DAQ issue down to a metrics investigation. Every query covers the full handed-off window — range queries, plus whole-window increase() totals; there is no separate live-value mode.
---

# Skill: psana-daq-monitor — LCLS-II DAQ Performance Monitor

Queries Prometheus (via Grafana MCP tools) for DAQ metrics from the L2SI
DAQ SDF dashboard, scoped to a hutch, platform, and `[start, end]` UTC
window. Read-only — observes and diagnoses, never manipulates the DAQ.

**Related skills:** for a broader or not-yet-diagnosed issue, load
`psana-daq` first to establish `hutch`/`platform`/window and decide which
angle to pursue. When metrics alone don't explain a finding, load
`psana-daq-logs` or `psana-daq-control` by name.

**Prerequisite:** `grafana_*` MCP tools (Prometheus datasource access) in
the current session. Call `grafana_list_datasources` (or attempt Step 1
below) first — if it fails or no `grafana_*` tools are available, stop and
tell the user Grafana MCP access is required and not currently configured,
rather than retrying calls that will all fail.

**Reference material** (cite, don't restate):
- `reference/metric-groups.md` — the A–I diagnosis hierarchy, full PromQL,
  interpretation per group, and the per-metric-type summarization rules
  (rates vs. fractions vs. counters vs. occupancy vs. status vs. latency).
- `reference/metric-catalog.md` — metric tables (type, labels, datasource),
  dashboard template variables, the architecture/data-flow diagram, system
  constants (`EB_TMO_MS`, buffer sizes, etc.), and the source file map.
- `reference/dashboard-panels.md` — panel IDs, deeplink examples, and the
  finding → recommendation table.

## Known resources (skip discovery)

| Resource | Name | UID |
|---|---|---|
| Prometheus (main) | prom_psdm | `afjrx1j1815vkf` |
| Prometheus (1s scrape) | prom_psdm_1s | `afotnjy2b05q8b` |
| Dashboard | L2SI DAQ SDF | `wihghwb` |
| Dashboard | L2SI DAQ SDF (single source) | `wijxd23` |

**Verify these UIDs before trusting them** — this exact table has rotted
once already: an earlier version of this skill pointed at `prom_drp`/
`prom_drp5s` (`000000002`/`000000008`), which are now unreachable
(`psmetric03.pcdsn:9090` refuses connections). The dashboards' own `prom`
template variable (`wihghwb`'s `templating` definition) is the live
source of truth for which datasource is actually current — re-check it
with `grafana_get_dashboard_property(uid="wihghwb",
jsonPath="$.variables")` if a query against the UIDs above fails with
"datasource not found" or a connection error, rather than assuming the
table above is still right. Don't propagate a hardcoded UID anywhere
else without the same caveat — this is exactly the kind of fact that
decays silently (see the standing principle against corpus/snapshot
constants in `README.md`).

## Establish instrument, partition, and window

If `psana-daq` already established `hutch`, `platform`, and the window,
use them directly — don't re-ask or re-derive. If invoked standalone, ask
for all three; don't guess, even if only one hutch currently shows active
metrics.

**`hutch` ≡ Prometheus's `instrument` label, but the source differs by
process family:**
- DAQ processes (`drp`, `teb`, `monReqServer`) set it directly from their
  own `-P {hutch}` flag (`drp.cc:51`, `teb.cc:1404`,
  `monReqServer.cc:1121`; e.g. the cnf's `std_opts = f'-P {hutch} -C
  {collect_host} -M {prom_dir}'` in `psdaq/psdaq/cnf/rix.cnf:35`).
- `epics_exporter.py`-bridged metrics (XPM `L0InpRate`/`DeadFrac`/etc.,
  `Us:`/`Cu:` link metrics) get it from the exporter's `-H` flag or the
  first field of a `-D hutch/id/PV` spec (`epics_exporter.py:98,103`) — a
  **static XPM→hutch table hardcoded in the base cnf file** (e.g.
  `psdaq/psdaq/cnf/neh-base.py:80`, `-D rix/xpm-3/..., rix/xpm-1/...,
  rix/xpm-5/...`), not read from the running DAQ.
  *(inferred-from-code-only, not verified live)* — this means a DAQ
  running as hutch `tst` on an XPM the cnf table assigns to `rix` would
  show `instrument="tst"` on its DRP metrics but `instrument="rix"` on
  its `DeadFrac`/`L0InpRate`/etc. Don't assume these always agree.
- None of these processes take a `--hutch` flag. AMI (`ami-node --hutch
  {hutch}`, sometimes `{hutch}_{instance}`, see
  `psdaq/psdaq/cnf/srcf-rix.cnf:148`) and the scan scripts
  (`psdaq/psdaq/cas/config_scan_base.py:43`) do — unrelated to any
  Prometheus label here.

**`platform` ≡ Prometheus's `partition` label — but only for DAQ-process
metrics.** This label means different things on different metric
families; see the table below before adding a `partition=` filter to any
query:

| Metric family | What `partition` means | How to filter |
|---|---|---|
| DAQ-process metrics (`drp_*`, `DRP_*`, `TEB_*`, `MEB_*`, `MRQ_*`, `EB_*`, `*Ctb*`) | the process's `-p` argument = platform (`control.py:189-190` sets `params['partition'] = self.platform`; `DrpBase.cc:813,1266`, `PGPDetector.cc:471`, `EbAppBase.cc:132`, `teb.cc:280`, `monReqServer.cc:451`) | `partition="<platform>"` — normal filtering; omitting it on a `sum(rate(...))` silently merges partitions |
| XPM per-group metrics (`L0InpRate`, `L0AccRate`, `DeadFrac`, `NumL0Acc`, `NumL0Inp`, `RunTime`, `Run`) | the XPM readout-group index 0–7 (`epics_exporter.py` loops `for i in range(8)` over `PART:%d` PVs, lines 56–62) — equals the platform only for the common readout group | `<platform>` selects only the common group; other groups need `partition="<rog>"`; all 8 groups are **always** exported, whether or not a DAQ is running on them |
| XPM link metrics (`Us:RxLinkUp`, `Cu:RxLinkUp`, `Us:RxDspErrs`, `Cu:RxDspErrs`) | **no `partition` label at all** — registered via `-G` → `registerGlobalPV()`, labels `['instrument','id']` only (`epics_exporter.py:47`) | never add `partition=` to these; a query that does returns zero series (verified live) |
| `drp_deadtime` | label is the platform (same labeling as other DRP metrics), but the *value* is read from an XPM PV path `<pv_base>:XPM:<xpm_id>:PART:<det_info.readout>:DeadFLnk` (`DrpBase.cc:1637-1648`) — the detector's own readout group, which can differ from the platform | `partition="<platform>"` for the label, but don't assume the value tracks the platform's own readout group |

Verified live: `DeadFrac{instrument="rix"}` returns `partition` values
0..7, all with `id="xpm-3"`; `Us:RxLinkUp{instrument="rix"}` returns 3
series (`id=xpm-1,xpm-3,xpm-5`) with labels
`__name__,id,instance,instrument,job` — no `partition` — and adding
`partition="2"` to that query returns zero series.

**Scope — do not over-apply any of this:** these are all specifically
Prometheus `partition`-label meanings. None of them generalize to Slurm
partitions (unrelated concept, same word) or to `det_info.readout` in
`activedet.json` (see `psana-daq-control/SKILL.md` Section 4) — that's a
DAQ-config concept, not a Prometheus label, even though it's closely
related to the XPM readout-group meaning above.

If `platform` wasn't handed off and the instrument has only one active
partition, proceeding without the filter is low-risk — but say so
explicitly rather than silently omitting it.

## The one query shape: range, always

Every query in this skill is:

```
grafana_query_prometheus(
    datasourceUid="afjrx1j1815vkf",
    expr='<metric>{instrument="<instrument>",partition="<platform>"}',
    queryType="range",
    startTime="<window.start>", endTime="<window.end>",
    stepSeconds=<see below>
)
```

**Never use `queryType="instant"` for a plain selector or a short-range
expression meant to answer a window question** — `startTime` is silently
ignored for an instant query (verified live via the Grafana MCP tools);
it evaluates only at `endTime`, so a deadtime spike, transient damage
burst, buffer exhaustion that recovered, or an EB timeout cluster
anywhere else in the window is invisible no matter what `endTime` you
pick. This holds whether the window ends five minutes ago or last month
— there is no separate "live" instant-query path to choose instead.

**One exception:** a whole-window `increase()` evaluated with
`queryType="instant", endTime="<window.end>"` *does* correctly answer a
window question, because its range vector (`[<window-duration>]`) already
spans the entire window before `endTime` — the single returned value
already covers it. See `reference/metric-groups.md`'s "Summarizing a
range result by metric type" section for the worked examples. This is the
only case where instant is correct; a plain selector or a short-range
expression (e.g. `irate(metric[5s])`) under `queryType="instant"` is not.

`stepSeconds`: pick so the returned point count stays manageable for the
window's actual length — 15s for a window under ~30 minutes, 30–60s for a
multi-hour window. Don't default to 15s for an hours-long window; that can
return thousands of points. Any `rate()`/`increase()`/`irate()` range
vector used inside a *range* query must be `>= stepSeconds`, or samples
between consecutive points are skipped.

**`timeRange={"from":…,"to":…}` is a panel-rendering parameter** — used
only by `get_panel_image` and `generate_deeplink`. Passing it to
`query_prometheus` instead of `startTime`/`endTime` is rejected with a
parse error, not silently ignored. `startTime`/`endTime` are the only time
parameters `query_prometheus` accepts.

## Step 1: confirm metrics are flowing, load the dashboard

Fire in parallel (one message, multiple tool calls) — independent Grafana
queries always go together, not sequentially:

```
# Confirm metrics present for the window
grafana_query_prometheus(
    datasourceUid="afjrx1j1815vkf",
    expr='drp_event_rate{instrument="<instrument>",partition="<platform>"}',
    queryType="range",
    startTime="<window.start>", endTime="<window.end>", stepSeconds=30
)

# Load dashboard panel queries for context
grafana_get_dashboard_panel_queries(uid="wihghwb")
```

- **Series present** → proceed with `{instrument="<value>",
  partition="<platform>"}` on every subsequent query.
- **No series returned** → don't jump to "DAQ not running." See the
  decision table below.

## Empty-result interpretation

An empty result for a range query — no series at all, not a measured zero
— under-determines the cause. Don't collapse it to a flat "DAQ not
running":

| Possible cause | What would corroborate it |
|---|---|
| Outside Prometheus retention | An oldest-available-sample timestamp or a retention setting you've actually checked — not merely the absence of a result |
| Scrape gap (brief exporter restart/outage) | A nonzero result immediately before and after the empty stretch, and the gap is short relative to the window |
| Genuinely not flowing (DAQ down, detector disconnected, exporter renamed) | Corroborating evidence from `psana-daq-control` (no session covering the window) or `psana-daq-logs` (no process writing logs in the window) |
| Wrong label for this metric (e.g. `partition=` on a metric that doesn't carry that label, like `Us:RxLinkUp`) | Check the metric's actual label set with `grafana_list_prometheus_label_names` before concluding the series doesn't exist |
| Indeterminate | None of the above confirmed — report "no samples returned for the requested window," not a specific cause |

A **measured** zero (a real sample whose value is 0) is different from an
**empty** result (no series returned) — a measured zero is itself a
finding ("not flowing right now"); an empty result is the ambiguous case
this table covers. Don't conflate the two in a report.

## Standing principle: zero is not the same as unhealthy or down

Zero event rate means "no events measured in this window" — not "DAQ not
running." It could mean no triggers were selected for that readout group.
Zero damage rate means "no damage measured" — not "healthy," though in
practice it usually is. Apply this reading before escalating any zero-
valued metric to a stronger claim.

One specific instance of this: `MRQ_BufCt=0` (free MEB buffers) is the
textbook signature of monitoring data loss, but a `0` sample alone can
also reflect fast buffer turnover between scrapes rather than genuine
exhaustion — corroborate against `MEB_EvtCt`/`TEB_EvtCt` rate parity and
`MEB_PrcCt` before calling it data loss (see `reference/metric-groups.md`
section I). This is a judgment call from one incident, not a verified
rule — a MEB expert hasn't signed off on it; flag it as such when applied.

## Diagnosis hierarchy

Work top-to-bottom — event rate (A) is the primary liveness signal. Each
row's PromQL and interpretation rules are in `reference/metric-groups.md`.

| Group | Question it answers | Reference section |
|---|---|---|
| A. Event rate | Is the DAQ running? Is the event rate healthy? | `metric-groups.md` §A |
| B. Deadtime | Is excessive deadtime rejecting triggers? | `metric-groups.md` §B |
| C. Damage | Are events coming through damaged? | `metric-groups.md` §C |
| D. DRP errors | Are DRP-level errors discarding events? | `metric-groups.md` §D |
| E. Buffers | Are buffers/queues healthy, or backing up? | `metric-groups.md` §E |
| F. File writing | Is file writing keeping up? | `metric-groups.md` §F |
| G. Event builder | Is the TEB/MEB event builder healthy? | `metric-groups.md` §G |
| H. Timing/hardware links | Are XPM/HSD timing and hardware links up? | `metric-groups.md` §H |
| I. Online monitoring (MEB) | Is monitoring (AMI/psplot/psana-live) receiving events? | `metric-groups.md` §I |

**Filter `partition` by what it means for that specific metric family**
(see the table above) — never put `partition=` on `Us:`/`Cu:` link
metrics (no such label exists; the query returns empty); still never
omit it from a DAQ-process `sum(rate(...))` aggregate, where omitting it
silently merges partitions instead of erroring.

For recommendations keyed off a specific finding, see
`reference/dashboard-panels.md`'s finding → recommendation table.
