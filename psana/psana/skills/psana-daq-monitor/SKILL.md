---
name: psana-daq-monitor
description: Query Grafana/Prometheus DAQ metrics for LCLS-II over a hutch, platform, and time window (DRP event rates, deadtime, damage, DRP errors, buffers, file writing, event builder, timing/hardware links, MEB monitoring). Use when the user's question already targets Grafana/Prometheus metrics specifically, or after psana-daq has narrowed a general DAQ issue down to a metrics investigation. Every query is a range query over the handed-off window — there is no separate live-value mode.
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

**`hutch` ≡ Prometheus's `instrument` label** — set directly from a
component's `--hutch` argument (`psdaq/psdaq/cas/epics_exporter.py:29-51`).

**`platform` ≡ Prometheus's `partition` label** —
`psdaq/psdaq/control/control.py:189-190` sets `params['partition'] =
self.platform` directly, and the DRP/EB/MEB exporters label their series
with it (`psdaq/drp/DrpBase.cc:813`, `psdaq/drp/PGPDetector.cc:471`,
`psdaq/psdaq/eb/src/EbAppBase.cc:132`, `psdaq/psdaq/monreq/
monReqServer.cc:450-454`). Add `partition="<platform>"` to every query
whose catalog entry in `reference/metric-catalog.md` lists `partition` as
a label — without it, multiple partitions running for the same instrument
get silently merged (`sum(rate(TEB_EvtCt{instrument="xpp"}[1m]))` sums
across partitions) or mixed.

**Scope of this equivalence — do not over-apply it:** this is specifically
the DAQ-process Prometheus `partition` label. It does **not** generalize
to Slurm partitions (unrelated concept, same word) or readout groups
(`det_info.readout` in `activedet.json`, see `psana-daq-control/
SKILL.md` Section 4, commonly *defaults* to platform but can be assigned
differently).

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

**Never use `queryType="instant"` for a window-scoped query** —
`startTime` is silently ignored for an instant query (verified live via
the Grafana MCP tools); it evaluates only at `endTime`, so a deadtime
spike, transient damage burst, buffer exhaustion that recovered, or an EB
timeout cluster anywhere else in the window is invisible no matter what
`endTime` you pick. This holds whether the window ends five minutes ago
or last month — there is no separate "live" instant-query path to choose
instead.

`stepSeconds`: pick so the returned point count stays manageable for the
window's actual length — 15s for a window under ~30 minutes, 30–60s for a
multi-hour window. Don't default to 15s for an hours-long window; that can
return thousands of points.

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

**Partition-filter every query that has the label** (see above) — the
common failure is forgetting it on a `sum(rate(...))` aggregate, which
silently merges partitions instead of erroring.

For recommendations keyed off a specific finding, see
`reference/dashboard-panels.md`'s finding → recommendation table.
