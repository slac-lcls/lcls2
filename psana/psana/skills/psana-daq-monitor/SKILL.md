---
name: psana-daq-monitor
description: Query Grafana/Prometheus DAQ metrics dashboards for LCLS-II (DRP event rates, deadtime, damage, errors, buffers, file writing, event builder, MEB monitoring). Load when the user's question already targets Grafana/Prometheus metrics specifically, or after psana-daq has narrowed a general DAQ issue down to a metrics investigation.
---

# Skill: psana-daq-monitor

# LCLS-II DAQ Performance Monitor

You are diagnosing performance of the LCLS-II DAQ (Data Acquisition) system using
Grafana MCP tools. You query Prometheus for DAQ metrics from the L2SI DAQ SDF
dashboard. You do NOT manipulate the DAQ — you only observe and diagnose.

**Related skills:** for a broader or not-yet-diagnosed DAQ issue, load
`psana-daq` first to decide which diagnostic angle to pursue. When metrics
alone don't explain a finding, load `psana-daq-logs` (raw DAQ log
inspection) or `psana-configdb` (read-only configuration lookups) via the
`skill` tool.

**Prerequisite:** This skill requires a Grafana MCP server providing `grafana_*`
tools (Prometheus datasource access) to be configured in the active opencode
session. Before proceeding, call `grafana_list_datasources` (or attempt the
Step 0 query below) — if it fails or no Grafana MCP tools are available, stop
and tell the user that Grafana MCP access is required for this skill and is
not currently configured, rather than attempting further `grafana_*` calls
that will all fail.

---

## Architecture Overview

```
Accelerator Timing (929 kHz)
        |
       XPM  (timing distribution + L0 trigger decisions)
      / | \
    DTI DTI DTI  (downstream timing interfaces)
    / \   |   \
  HSD Epix Jungfrau ...  (detector front-ends with timing receivers)
    |   |      |
   PGP PGP   PGP  (data links via DMA)
    |   |      |
  DRP  DRP   DRP  (Data Readout Processors — one per detector)
    |         |
    +----+----+
         |
        TEB  (Trigger Event Builder — runs trigger, decides persist/monitor)
       /   \
     DRP    MEB  (results back to DRPs for file writing; MEB for monitoring)
      |       |
   XTC2     Shared Memory → AMI / psplot / psana-live
   files
```

- **XPM**: FPGA timing master. Distributes pulse IDs and timestamps at ~929 kHz,
  generates L0 triggers, enforces flow control via inhibits. Publishes `L0InpRate`,
  `L0AccRate`, `DeadFrac`, `RunTime`, `NumL0Acc` as EPICS PVs bridged to Prometheus
  via `epics_exporter.py`.

- **DRP**: Data Readout Processor — one per detector. Three thread types: Reader
  (PGP DMA), Worker (format XTC, compute trigger primitives), Collector (batch
  assembly). Sends trigger inputs to TEB via RDMA, receives results, writes
  XTC2 + SMD files locally.

- **TEB**: Trigger Event Builder. Assembles trigger inputs from all DRPs by pulse
  ID. Runs trigger algorithm to set `persist`/`monitor` flags in a `ResultDgram`.
  Returns results to DRPs via RDMA. Event timeout: 12 seconds (`EB_TMO_MS`).
  Up to 4 TEBs (`MAX_TEBS`).

- **MEB**: Monitoring Event Builder. Assembles full events for online monitoring.
  Distributes via POSIX shared memory to clients (AMI, psplot, psana-live). Buffer
  pool circulates: MEB → TEB (MRQ) → DRP (RDMA write) → MEB → shmem → client → MEB.
  Up to 4 MEBs (`MAX_MEBS`).

- **File Writing**: Happens inside each DRP process (not a separate file writer).
  Each DRP writes its own XTC2 data file + SMD index file via `BufferedFileWriterMT`
  (multi-threaded, supports `O_DIRECT`). Files are chunked at 500 GB
  (`DefaultChunkThresh`).

---

## Discovery (always start here)

Known UIDs — use these directly, skip discovery tool calls:

| Resource | Name | UID |
|----------|------|-----|
| Prometheus (1s scrape) | prom_drp | `000000002` |
| Prometheus (5s scrape) | prom_drp5s | `000000008` |
| Dashboard | L2SI DAQ SDF | `wihghwb` |
| Dashboard | L2SI DAQ SDF (single source) | `wijxd23` |

If a query fails with "datasource not found", fall back to
`grafana_list_datasources` to re-discover UIDs.

### Step 0: Establish which hutch/instrument (ALWAYS DO THIS FIRST)

**`hutch` and `instrument` are the same identifier** — Prometheus's `instrument`
label is set directly from a component's `--hutch` argument
(`psdaq/psdaq/cas/epics_exporter.py`). If the `psana-daq` router (or earlier
in this conversation) already established `hutch`, use it as `instrument`
directly — do not re-ask or re-derive it.

If not already known, **ask the user** which hutch/instrument to investigate
— do not guess or auto-select, even if only one instrument currently shows
active metrics. Known instruments: asc, mfx, rix, tmo, tst, txi, ued, xpp.

Once you have it (from either source), proceed to Step 1, which confirms
metrics are actually flowing for it before loading the dashboard.

**If `psana-daq` also handed off a scope** (live, or a past date/time
range), use it instead of the "Default time ranges" table below: a live
scope uses the defaults as normal, but a **past** scope means every query in
this skill should use explicit `startTime`/`endTime` covering that session's
window rather than a `now`-relative default. Derive the window the same way
`psana-daq`'s autonomous sweep does for a past session — start from the
session's log-prefix timestamp, end from the last-written log file's mtime
— rather than re-deriving it independently here.

If you need to see which instruments currently have *any* active DAQ metrics
(e.g. the user isn't sure and there's no router context to fall back on),
use this as a discovery aid — but treat its result as informational, not a
substitute for asking:

    grafana_list_prometheus_label_values(
        datasourceUid="000000002",
        labelName="instrument",
        matches=[{"filters": [
            {"name": "__name__", "value": "drp_event_rate", "type": "="}
        ]}]
    )

### Step 1: Confirm metrics are flowing and load the dashboard

Fire these in parallel (single message, multiple tool calls):

    # Confirm metrics flowing for the stated instrument
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='drp_event_rate{instrument="<instrument>"}',
        queryType="instant",
        endTime="now"
    )

    # Load dashboard panel queries for context
    grafana_get_dashboard_panel_queries(uid="wihghwb")

- **Metrics present** → proceed with `{instrument="<value>"}` on all
  subsequent queries.
- **No results** → tell the user plainly: DAQ is not running for that
  instrument, or metrics are not flowing; check connectivity before
  proceeding further.

> **Parallel query rule:** Always fire independent Grafana queries in a single
> message. The event rate, deadtime, damage, and error queries are all
> independent — send them together, not sequentially.

> **Default time ranges:**
>
> | Query type | Window | Parameters |
> |---|---|---|
> | Instant (current state) | now only | `endTime="now"`, no `startTime` |
> | Range (recent trend) | 5 min | `startTime="now-5m"`, `endTime="now"`, `stepSeconds=15` |
> | Range (wider view) | 30 min | `startTime="now-30m"`, `endTime="now"`, `stepSeconds=30` |
>
> The dashboard defaults to a 5-minute window with auto-refresh.

---

## Dashboard Variables

The L2SI DAQ SDF dashboard uses these template variables. When building queries,
substitute these as label selectors:

| Variable | Label | Values | Usage |
|---|---|---|---|
| `$instrument` | `instrument` | asc, mfx, rix, tmo, tst, txi, ued, xpp | Filter all DRP/EB metrics |
| `$partition` | `partition` | 0–7 | Filter TEB/MEB and per-partition views |
| `$group` | `partition` | 0–7 | Filter readout group stats (single group) |
| `$detname` | `detname` | e.g. hsd, epixquad1kfps, jungfrau1M, ... | Filter per-detector views |
| `$prom` | — | datasource selector | Selects which Prometheus instance to query |

Partitions are logical DAQ instances (0–7). Multiple experiments can run
simultaneously on different partitions. The "common readout group" has the
same number as the partition — all detectors in the partition belong to it.

---

## Diagnosis Hierarchy

Work top-to-bottom. Start with event rate — it is the primary liveness signal.

### A. Is the DAQ running? Is the event rate healthy?

The most basic health check: are events flowing through the system?

**Metric check (fire in parallel):**

    # DRP event rate per detector
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='drp_event_rate{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # L0 input rate (trigger rate into readout group)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='L0InpRate{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # L0 accept rate (triggers accepted)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='L0AccRate{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # TEB event rate
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='sum(rate(TEB_EvtCt{instrument="<instrument>"}[1m]))',
        queryType="instant", endTime="now"
    )

**Interpretation:**
- `drp_event_rate` > 0 across all expected detectors = DAQ is running
- `L0InpRate` ≈ `L0AccRate` = minimal deadtime, triggers accepted efficiently
- `L0InpRate` >> `L0AccRate` = significant deadtime (see section B)
- TEB rate ≈ L0AccRate = event builder keeping up
- Zero event rate = DAQ not running or detector not connected
- Mismatched rates across detectors = possible per-detector issues

Note: `L0InpRate` and `L0AccRate` are EPICS PVs from the XPM, bridged to
Prometheus via `epics_exporter.py`. `L0InpRate` = L0 triggers input / time.
`L0AccRate` = L0 triggers accepted / time.

**Trend check (if rates look anomalous):**

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='drp_event_rate{instrument="<instrument>"}',
        queryType="range",
        startTime="now-5m", endTime="now", stepSeconds=15
    )

---

### B. Is there excessive deadtime?

Deadtime is the fraction of L0 triggers rejected because the readout system
is busy. Computed by the XPM as `Δ(numl0Inh) / Δ(numl0)` per measurement
interval. The DRP-side `drp_deadtime` reads the XPM's per-link `DeadFLnk`
EPICS PV — it is NOT computed by the DRP itself.

**Metric check:**

    # Dead fraction per readout group (0–1 scale)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='DeadFrac{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Per-detector deadtime percentage
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='100.*drp_deadtime{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

**Interpretation** (operational guidelines — no code-defined alarm thresholds):

| DeadFrac | Meaning |
|---|---|
| < 1% | Healthy — system keeping up with triggers |
| 1–5% | Marginal — monitor closely |
| 5–20% | Concerning — investigate backpressure source |
| > 20% | Critical — significant data loss |

**Follow-up:** If deadtime is high, check:
1. Event builder health (section G) — TEB/MEB backpressure
2. Buffer depth (section E) — DRP buffer exhaustion
3. File writing (section F) — recording backpressure

---

### C. Is there damage?

Damaged events have missing or corrupted data from one or more detectors.
The damage field is a **bitmask** — multiple types can be set simultaneously.

**Damage types** (from `xtcdata/xtc/Damage.hh`):

| Bit | Name | Meaning |
|---|---|---|
| 0 | Truncated | Data was truncated |
| 1 | OutOfOrder | Data arrived out of order |
| 2 | OutOfSynch | Data out of synchronization |
| 3 | Corrupted | Data corrupted |
| 4 | DroppedContribution | A contributor didn't send data (set by TEB `fixup()`) |
| 5 | MissingData | Expected data is missing (set by detectors: EpixHR, Jungfrau, etc.) |
| 6 | TimedOut | Operation timed out (set by PvaDetector, UdpEncoder) |
| 12 | UserDefined | User-defined damage |

**Metric check:**

    # Total damage count per detector (monotonically increasing gauge)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='DRP_Damage{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Damage rate (events/sec)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='irate(DRP_Damage{instrument="<instrument>"}[5s])',
        queryType="instant", endTime="now"
    )

**Interpretation:**
- Zero damage rate = healthy
- Nonzero but low (< 1/s) = occasional glitches, usually benign
- Sustained damage rate = systematic issue — check hardware links (H)
  and DRP errors (D)
- Damage on a single detector = detector-specific problem
- Damage on all detectors = timing or event builder issue

**Trend check:**

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='irate(DRP_Damage{instrument="<instrument>"}[5s])',
        queryType="range",
        startTime="now-5m", endTime="now", stepSeconds=15
    )

---

### D. Are there DRP errors?

Six types of discarded error events, dropped early in the readout chain
(never reach recording or monitoring). All are registered as `MetricType::Gauge`
in prometheus-cpp despite being monotonically increasing — use `rate()` to
get the rate.

**Error definitions** (from `psdaq/drp/DrpBase.cc`):

| Metric | Code-derived meaning | Event discarded? |
|---|---|---|
| `drp_num_dma_errors` | DMA transfer error from PGP kernel driver (`dmaErrors[]` nonzero). Hardware-level: CRC failure, link error, FIFO overflow. | **Yes** |
| `drp_num_pgp_jump` | Gap in 24-bit `evtCounter` (expected `m_lastComplete+1`, got different). Events lost between hardware and DRP. PGP link drops, firmware buffer overflow. | **Yes** (skipped events) |
| `drp_num_no_common_rog` | Timing header's `readoutGroups()` bitmask lacks the partition bit. Would break TEB/MEB pulse ID ordering. | **Yes** |
| `drp_num_missing_rogs` | **SlowUpdate transitions only.** Missing one or more expected readout groups (from `rogMask`). Would break psana. | **Yes** |
| `drp_num_th_error` | Bit 7 of timing header control field set by XPM/TPR firmware. Timing distribution error (CRC, clock domain crossing). | **No** — logged but processing continues |
| `drp_num_no_tr_dgram` | Transition buffer pool (128 slots, `TEB_TR_BUFFERS`) exhausted. Can happen during shutdown or transition storms. | **Yes** |

**Metric check (fire all in parallel):**

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='rate(drp_num_dma_errors{instrument="<instrument>"}[1m])',
        queryType="instant", endTime="now"
    )

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='rate(drp_num_pgp_jump{instrument="<instrument>"}[1m])',
        queryType="instant", endTime="now"
    )

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='rate(drp_num_no_common_rog{instrument="<instrument>"}[1m])',
        queryType="instant", endTime="now"
    )

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='rate(drp_num_missing_rogs{instrument="<instrument>"}[1m])',
        queryType="instant", endTime="now"
    )

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='rate(drp_num_th_error{instrument="<instrument>"}[1m])',
        queryType="instant", endTime="now"
    )

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='rate(drp_num_no_tr_dgram{instrument="<instrument>"}[1m])',
        queryType="instant", endTime="now"
    )

**Total error count (single stat per detector):**

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='sum(drp_num_dma_errors{instrument="<instrument>",detname="<detname>"})+sum(drp_num_no_common_rog{instrument="<instrument>",detname="<detname>"})+sum(drp_num_missing_rogs{instrument="<instrument>",detname="<detname>"})+sum(drp_num_th_error{instrument="<instrument>",detname="<detname>"})+sum(drp_num_pgp_jump{instrument="<instrument>",detname="<detname>"})+sum(drp_num_no_tr_dgram{instrument="<instrument>",detname="<detname>"})',
        queryType="instant", endTime="now"
    )

**Action:** Any sustained error rate > 0 needs investigation. Cross-reference
with hardware link status (section H). Note that `drp_num_th_error` is the
only error that does NOT discard the event.

---

### E. Are buffers and queues healthy?

Buffer exhaustion causes backpressure → deadtime → data loss. The chain is:
DMA fills → pebble fills → deadtime rises. Trace back to the earliest
bottleneck (usually file writing or event builder).

**Metric check:**

    # FileWriter free buffer depth (should stay > 0)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='DRP_RecordDepth{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Pebble buffers in use vs max
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='drp_pebble_in_use{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='drp_pebble_in_use_max{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # DMA buffers in use vs max
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='drp_dma_in_use{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='drp_dma_in_use_max{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

**Interpretation:**
- `DRP_RecordDepth` → 0 = file writer buffer exhaustion, recording will stall
- `drp_pebble_in_use` approaching `drp_pebble_in_use_max` = DRP backpressure
- `drp_dma_in_use` approaching `drp_dma_in_use_max` = DMA ring buffer filling up
- Growing queue occupancy over time = system falling behind

**Action:** Buffer exhaustion chains: DMA fills → pebble fills → deadtime rises.
Trace back to the earliest bottleneck (usually file writing or event builder).

---

### F. Is file writing keeping up?

File writing happens inside each DRP process (not a separate file writer).
Each DRP writes its own XTC2 data file + SMD index file via
`BufferedFileWriterMT` (multi-threaded, supports `O_DIRECT`).

**Metric check:**

    # File writing state (1 = writing, 0 = not writing)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='DRP_fileWriting{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # FileWriter blocked waiting for free buffer (0/1)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='DRP_bufFreeBlk{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Record rate (bytes/sec being written)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='rate(DRP_RecordSize{instrument="<instrument>"}[1m])',
        queryType="instant", endTime="now"
    )

    # Total recorded data size
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='DRP_RecordSize{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

**Interpretation:**
- `DRP_fileWriting` = 0 when expected to record = run not configured for
  recording, or writing stalled
- `DRP_bufFreeBlk` = 1 = file writer blocked waiting for a free buffer from
  `BufferedFileWriterMT` — backpressure will propagate upstream
- Declining `DRP_RecordDepth` trend (section E) = file system I/O bottleneck

**Action:** Check disk I/O, NFS mount health, available disk space. Files are
chunked at 500 GB (`DefaultChunkThresh`). Consider reducing data rate or
adding compression.

---

### G. Is the event builder healthy?

The TEB assembles trigger inputs from all DRPs by pulse ID. Events older
than 12 seconds (`EB_TMO_MS = 12000`) are timed out. "Fixups" (`EB_FxUpCt`)
are incomplete events flushed when a newer complete event arrives — missing
contributors get `DroppedContribution` damage (bit 4). The `eb` label
distinguishes TEB vs MEB instances.

**Metric check:**

    # TEB event and write rates
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='sum(rate(TEB_EvtCt{instrument="<instrument>"}[1m]))',
        queryType="instant", endTime="now"
    )

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='sum(rate(TEB_WrtCt{instrument="<instrument>"}[1m]))',
        queryType="instant", endTime="now"
    )

    # MEB event rate
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='sum(rate(MEB_EvtCt{instrument="<instrument>"}[1m]))',
        queryType="instant", endTime="now"
    )

    # Event builder fixup count (flushed incomplete events)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='sum(rate(EB_FxUpCt{instrument="<instrument>",eb="TEB"}[1m]))',
        queryType="instant", endTime="now"
    )

    # Event builder timeout count (12-second timeout exceeded)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='sum(rate(EB_ToEvCt{instrument="<instrument>",eb="TEB"}[1m]))',
        queryType="instant", endTime="now"
    )

    # Age of oldest incomplete event
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='EB_EvAge{instrument="<instrument>",eb="TEB"}',
        queryType="instant", endTime="now"
    )

**Interpretation:**
- `TEB_EvtCt` rate ≈ L0AccRate = TEB keeping up
- `TEB_WrtCt` rate > 0 = events being recorded
- `EB_FxUpCt` rate > 0 = incomplete events flushed (missing contributor).
  The `EB_FxUpSc` histogram shows which source IDs triggered fixups.
  `EB_CbMsMk` is the most recent missing contributors bitmask.
- `EB_ToEvCt` rate > 0 = events exceeding 12-second timeout (detector too
  slow or disconnected)
- MEB rate = 0 when monitoring expected = monitoring MEB not receiving events
- `EB_EvAge` growing = event builder falling behind

Note: Fixup/timeout logging in the code goes quiet after 50 cumulative
events to avoid log flooding.

**Action:** High fixup or timeout rates indicate a slow detector. Check
per-detector event rates and latency. The `EB_FxUpSc` histogram identifies
the offending source ID.

---

### H. Are timing and hardware links healthy?

**XPM Link Status (fire in parallel):**

    # Ultrafast (Us) link status
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='Us:RxLinkUp{}',
        queryType="instant", endTime="now"
    )

    # Copper (Cu) link status
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='Cu:RxLinkUp{}',
        queryType="instant", endTime="now"
    )

    # XPM dispatch error rates
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='rate(Us:RxDspErrs{}[1m])',
        queryType="instant", endTime="now"
    )

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='rate(Cu:RxDspErrs{}[1m])',
        queryType="instant", endTime="now"
    )

**HSD Hardware Status (uses prom_drp5s datasource):**

    # PGP local and remote link ready (bitmask)
    grafana_query_prometheus(
        datasourceUid="000000008",
        expr='loclinkrdy{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    grafana_query_prometheus(
        datasourceUid="000000008",
        expr='remlinkrdy{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # JESD link alignment
    grafana_query_prometheus(
        datasourceUid="000000008",
        expr='RxDataNAlign{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Feature extraction out-of-range rate
    grafana_query_prometheus(
        datasourceUid="000000008",
        expr='rate(fexoor{instrument="<instrument>"}[1m])',
        queryType="instant", endTime="now"
    )

**Interpretation:**
- `RxLinkUp` = 0 for any link = timing link down, detector won't receive
  triggers
- Nonzero `RxDspErrs` rate = dispatch errors on timing link, may cause
  trigger issues
- `loclinkrdy` or `remlinkrdy` = 0 = PGP link down between DRP and HSD
- `RxDataNAlign` ≠ 0 = JESD deserialization issue on HSD
- Growing `fexoor` rate = HSD feature extraction seeing out-of-range values

**Action:** Link down → check cables, power cycle HSD, re-initialize timing.
JESD issues → HSD firmware/hardware problem. FEXOOR → adjust HSD thresholds.

---

### I. Is online monitoring healthy? (MEB)

The MEB is the gateway to all online monitoring (AMI, psplot, psana-live).
It has a circular buffer pool that flows through three measured phases:

```
    MEB_AppTm          MEB_TrgTm              MEB_PrcTm
   (idle time)     (request→build)        (build→client done)
       |                  |                       |
  +---------+     +---------------+      +----------------+
  | Buffer  |     | MRQ→TEB→DRP→  |      | XtcMonitorSvr  |
  | in free |────>| RDMA→MEB      |─────>| →shmem→client   |────> back to free list
  |  list   |     | event build   |      | processing      |
  +---------+     +---------------+      +----------------+
```

`MEB_TrgTm + MEB_PrcTm + MEB_AppTm ≈ total buffer cycle time`

**Critical failure mode:** When `MRQ_BufCt` (free buffers) drops to 0, the
MEB can't send MRQ requests to the TEB. The TEB's buffer list for this MEB
drains, and it silently sets `monitor(0)` on events (`TEB_nMonCt` increments).
**No error is raised — monitoring events are just dropped.** Everything looks
fine to the DAQ, but AMI/psplot/psana-live see event gaps.

**Metric check (fire in parallel):**

    # Free monitoring buffers (0 = monitoring data loss!)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='MRQ_BufCt{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Total monitoring buffer pool size (constant)
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='MRQ_BufCtMax{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Buffers in-flight to shmem clients
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='MEB_PrcCt{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # MEB event rate
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='rate(MEB_EvtCt{instrument="<instrument>"}[1m])',
        queryType="instant", endTime="now"
    )

    # Max buffer processing time (ns) — slow shmem client indicator
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='MEB_PrcTmM{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Max request-to-build time (ns) — DRP/TEB bottleneck indicator
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='MEB_TrgTmM{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Min buffer idle time (ns) — near zero = buffer-starved
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='MEB_AppTmm{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Event latency (μs) — wall-clock lag
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='MEB_EvtLat{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

    # Split events — serious error
    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='MEB_SpltCt{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

**Interpretation:**
- `MRQ_BufCt` = 0 = **monitoring data loss** — MEB can't request new events.
  **Caveat, not yet confirmed by a DAQ expert (added 2026-09-23, from a live
  sweep):** a `0` sample alone can also reflect fast buffer turnover between
  Prometheus scrapes rather than genuine exhaustion — one real 24h xpp sweep
  saw `MRQ_BufCt=0` on 30 of 33 samples over 8h while `rate(MEB_EvtCt)`
  tracked `rate(TEB_EvtCt)` almost exactly (120.04 vs 120.03) and
  `MEB_PrcCt=0` (no buffers held by shmem clients) — i.e. monitoring was
  fully keeping up. Before concluding data loss from `MRQ_BufCt=0` alone,
  corroborate with `MEB_EvtCt`/`TEB_EvtCt` rate parity and `MEB_PrcCt`; if
  both indicate healthy throughput, the `0` samples may be an artifact of
  scrape timing rather than the failure mode described above. This
  corroboration rule is a judgment call from one incident, not a verified
  fact — treat `MRQ_BufCt=0` as before if these companion metrics also look
  unhealthy or are unavailable.
- `MRQ_BufCt` / `MRQ_BufCtMax` ratio < 25% = buffer pressure, investigate
  client speed
- `MEB_PrcCt` high = many buffers held by shmem clients — slow AMI/psplot
- `MEB_PrcTmM` high (max processing time, nanoseconds) = shmem client is slow
- `MEB_TrgTmM` high (max request-to-build time, nanoseconds) = DRP/TEB
  bottleneck on monitoring path
- `MEB_AppTmm` near zero (min idle time) = buffers immediately reused, MEB
  is buffer-starved
- `MEB_EvtLat` high = wall-clock lag between event timestamp and MEB
  processing
- `MEB_SpltCt` > 0 = **serious error** — late contributions after fixup
  created duplicate pulse IDs
- Compare `rate(MEB_EvtCt)` vs `rate(TEB_EvtCt)`: large gap = events not
  reaching monitoring

**DRP-side MEB backpressure check:**

    grafana_query_prometheus(
        datasourceUid="000000002",
        expr='MCtbO_TxPdg{instrument="<instrument>"}',
        queryType="instant", endTime="now"
    )

High `MCtbO_TxPdg` = DRP's posts to MEB are backing up (MEB overwhelmed).

**Action:** Slow monitoring client → simplify AMI graph, reduce plot
complexity. Buffer exhaustion → increase MEB buffer count (`-n` flag).
Split events → investigate timing/trigger issues. If monitoring doesn't
matter, `TEB_nMonCt` incrementing is benign.

---

## Metrics Reference

### DRP Metrics (prom_drp — `000000002`)

| Metric | Type | Key Labels | Description |
|--------|------|------------|-------------|
| `drp_event_rate` | Rate | instrument, detname, partition | Events/sec per detector (computed from counter) |
| `drp_deadtime` | Float | instrument, detname, partition | Dead fraction per detector (0–1), read from XPM PV |
| `drp_event_age` | Gauge | instrument, detname | Age of current event |
| `drp_dma_in_use` | Gauge | instrument, detname | DMA buffers currently in use |
| `drp_dma_in_use_max` | Constant | instrument, detname | DMA buffer pool size (set at configure) |
| `drp_dma_size` | Gauge | instrument, detname | DMA buffer size |
| `drp_pebble_in_use` | Gauge | instrument, detname | Pebble buffers in use |
| `drp_pebble_in_use_max` | Constant | instrument, detname | Pebble buffer pool size (set at configure) |
| `drp_pgp_byte_rate` | Rate | instrument, detname | PGP data rate (bytes/sec) |
| `drp_num_pgp_bufs` | Gauge | instrument, detname | PGP buffers available |
| `drp_num_pgp_in_hw` | Gauge | instrument, detname | PGP buffers in hardware |
| `drp_num_pgp_in_rx` | Gauge | instrument, detname | PGP buffers in receive |
| `drp_num_pgp_in_user` | Gauge | instrument, detname | PGP buffers in user space |
| `drp_th_latency` | Gauge | instrument, detname | Timing header latency |
| `drp_pv_latency` | Gauge | instrument, detname | PV (process variable) latency |
| `drp_port_rcv_rate` | Rate | instrument | InfiniBand receive rate |
| `drp_port_xmit_rate` | Rate | instrument | InfiniBand transmit rate |
| `drp_num_dma_errors` | Gauge* | instrument, detname, partition | DMA transfer errors (monotonically increasing) |
| `drp_num_pgp_jump` | Gauge* | instrument, detname, partition | PGP sequence discontinuities |
| `drp_num_no_common_rog` | Gauge* | instrument, detname, partition | Events without common readout group |
| `drp_num_missing_rogs` | Gauge* | instrument, detname, partition | SlowUpdates missing readout groups |
| `drp_num_th_error` | Gauge* | instrument, detname, partition | Timing header error bit set |
| `drp_num_no_tr_dgram` | Gauge* | instrument, detname, partition | Transition buffer pool exhausted |
| `DRP_Damage` | Gauge | instrument, detname, partition | Damaged event count (monotonically increasing) |
| `DRP_DamageType` | Histogram | instrument, detname, partition | Damage type distribution (16 bins) |
| `DRP_RecordSize` | Counter | instrument, detname, partition | Cumulative bytes recorded |
| `DRP_RecordDepth` | Gauge | instrument, detname, partition | File writer free buffer slots |
| `DRP_RecordDepthMax` | Constant | instrument, detname, partition | File writer max buffer slots |
| `DRP_fileWriting` | Gauge | instrument, detname, partition | File writing active (0/1) |
| `DRP_bufFreeBlk` | Gauge | instrument, detname, partition | FileWriter blocked on free buffer (0/1) |
| `DRP_bufPendBlk` | Gauge | instrument, detname, partition | FileWriter blocked on pending buffer (0/1) |
| `DRP_evtSize` | Gauge | instrument, detname | Event size |
| `DRP_evtLatency` | Gauge | instrument, detname | Event latency |

*Gauge\** = registered as Gauge in prometheus-cpp but monotonically increasing
— use `rate()` to get the rate.

### Event Builder Metrics (prom_drp — `000000002`)

| Metric | Type | Key Labels | Description |
|--------|------|------------|-------------|
| `TEB_EvtCt` | Counter | instrument, partition | TEB total event count |
| `TEB_WrtCt` | Counter | instrument, partition | TEB write (record) count |
| `TEB_TrCt` | Counter | instrument, partition | TEB transition count |
| `TEB_SpltCt` | Counter | instrument, partition | TEB split event count |
| `TEB_TxPdg` | Gauge | instrument, partition | TEB transmit pending |
| `TEB_EvtLat` | Gauge | instrument, partition | TEB event latency |
| `TEB_trg_dt` | Gauge | instrument, partition | Trigger decision time |
| `MEB_EvtCt` | Counter | instrument, partition | MEB total event count |
| `MEB_TrCt` | Counter | instrument, partition | MEB transition count |
| `MEB_EvtLat` | Gauge | instrument, partition | MEB event latency (μs) |
| `MEB_SpltCt` | Counter | instrument, partition | MEB split events (serious error) |
| `MEB_ReqCt` | Counter | instrument, partition | MRQ messages sent to TEBs |
| `MEB_PrcCt` | Gauge | instrument, partition | Buffers in-flight to shmem clients |
| `MEB_PrcTm` | Gauge | instrument, partition | Buffer processing time sample (ns) |
| `MEB_PrcTmm` | Gauge | instrument, partition | Buffer processing time minimum (ns) |
| `MEB_PrcTmM` | Gauge | instrument, partition | Buffer processing time maximum (ns) |
| `MEB_PrcTmA` | Gauge | instrument, partition | Buffer processing time average (ns) |
| `MEB_TrgTm` | Gauge | instrument, partition | MRQ request-to-build time sample (ns) |
| `MEB_TrgTmm` | Gauge | instrument, partition | MRQ request-to-build time minimum (ns) |
| `MEB_TrgTmM` | Gauge | instrument, partition | MRQ request-to-build time maximum (ns) |
| `MEB_TrgTmA` | Gauge | instrument, partition | MRQ request-to-build time average (ns) |
| `MEB_AppTm` | Gauge | instrument, partition | Buffer idle time sample (ns) |
| `MEB_AppTmm` | Gauge | instrument, partition | Buffer idle time minimum (ns) |
| `MEB_AppTmM` | Gauge | instrument, partition | Buffer idle time maximum (ns) |
| `MEB_AppTmA` | Gauge | instrument, partition | Buffer idle time average (ns) |
| `MEB_RogCt0`–`7` | Counter | instrument, partition | Events per readout group (0–7) |
| `MRQ_BufCt` | Gauge | instrument, partition | Free monitoring buffers (**0 = data loss**) |
| `MRQ_BufCtMax` | Constant | instrument, partition | Total monitoring buffer pool size |
| `MRQ_TxPdg` | Gauge | instrument, partition | MRQ transmit pending |
| `EB_FxUpCt` | Counter | instrument, partition, eb | Flushed incomplete events (fixups) |
| `EB_ToEvCt` | Counter | instrument, partition, eb | Timed-out events (12s default) |
| `EB_EvAge` | Gauge | instrument, partition, eb | Age of oldest incomplete event |
| `EB_CbMsMk` | Gauge | instrument, partition, eb | Missing contributors bitmask |
| `EB_RxPdg` | Gauge | instrument, partition, eb | Receive pending |
| `EB_TxPdg` | Gauge | instrument, partition, eb | Transmit pending |
| `EB_FxUpSc` | Histogram | instrument, partition, eb | Fixup source distribution |
| `EB_CtrbSc` | Histogram | instrument, partition, eb | Contributor source distribution |

### DRP-side EB Contributor Metrics (prom_drp — `000000002`)

| Metric | Type | Key Labels | Description |
|--------|------|------------|-------------|
| `TCtbO_EvtCt` | Counter | instrument, detname | Events posted to TEB |
| `TCtbO_TxPdg` | Gauge | instrument, detname | TEB post transmit pending |
| `TCtbO_InFlt` | Gauge | instrument, detname | Events in flight to TEB |
| `TCtbO_Lat` | Gauge | instrument, detname | TEB contribution latency |
| `TCtbI_EvtCt` | Counter | instrument, detname | Results received from TEB |
| `TCtbI_MisCt` | Counter | instrument, detname | Results missing an input event |
| `MCtbO_EvCt` | Counter | instrument, detname | Events posted to MEB |
| `MCtbO_TxPdg` | Gauge | instrument, detname | MEB post transmit pending |

### Readout Group & Timing Metrics

| Metric | Datasource | Type | Description |
|--------|------------|------|-------------|
| `L0InpRate` | prom_drp | Gauge | L0 trigger input rate (from XPM EPICS PV) |
| `L0AccRate` | prom_drp | Gauge | L0 trigger accept rate |
| `DeadFrac` | prom_drp | Gauge | Dead fraction (0–1): Δ(numl0Inh) / Δ(numl0) |
| `RunTime` | prom_drp | Gauge | Current run elapsed time (seconds) |
| `NumL0Acc` | prom_drp | Gauge | Total L0 accepts in current run |
| `Us:RxLinkUp` | prom_drp | Gauge | Ultrafast XPM receive link status |
| `Cu:RxLinkUp` | prom_drp | Gauge | Copper XPM receive link status |
| `Us:RxDspErrs` | prom_drp | Gauge* | Ultrafast XPM dispatch errors |
| `Cu:RxDspErrs` | prom_drp | Gauge* | Copper XPM dispatch errors |
| `loclinkrdy` | prom_drp5s | Gauge | HSD PGP local link ready (bitmask) |
| `remlinkrdy` | prom_drp5s | Gauge | HSD PGP remote link ready (bitmask) |
| `RxDataNAlign` | prom_drp5s | Gauge | HSD JESD link alignment status |
| `fexoor` | prom_drp5s | Gauge* | HSD feature extraction out-of-range |

---

## System Constants

| Constant | Value | Source | Significance |
|---|---|---|---|
| `EB_TMO_MS` | 12,000 ms | `eb.hh:53` | Event builder timeout |
| `TEB_TR_BUFFERS` | 128 | `eb.hh:54` | Transition buffer pool — exhaustion → `drp_num_no_tr_dgram` |
| `MEB_TR_BUFFERS` | 24 | `eb.hh:56` | MEB transition buffer pool |
| `MAX_ENTRIES` | 64 | `eb.hh:59` | Events per batch |
| `TICK_RATE` | ~928,571 Hz | `eb.hh:48` | Timing system tick rate (13/14 MHz) |
| `MAX_LATENCY` | 16,777,216 ticks | `eb.hh:62` | Max buffering capacity (~18 seconds) |
| `EvtCtrMask` | 0xffffff | `DrpBase.cc:39` | 24-bit event counter wrap mask |
| `BATCH_TIMEOUT` | 1 ms | `TebContributor.cc:42` | Batch flush timeout |
| `DefaultChunkThresh` | 500 GB | `DrpBase.hh` | File chunk rotation threshold |
| `MAX_TEBS` | 4 | `eb.hh` | Maximum TEB instances |
| `MAX_MEBS` | 4 | `eb.hh` | Maximum MEB instances |

---

## Tuning & Action Recommendations

| Finding | Recommendation |
|---------|---------------|
| Zero event rate | DAQ not running — check run control, detector power, timing links |
| High deadtime (>5%) | Trace backpressure: check buffers (E), file writing (F), then event builder (G) |
| Sustained damage rate | Check hardware links (H), correlate with specific detectors via `detname` label |
| DMA errors | Hardware issue — check PGP cables, HSD card, firmware version |
| PGP jumps | Link instability — check `loclinkrdy`/`remlinkrdy`, reseat cables |
| No common ROG | Timing config — verify readout group assignments match partition |
| Missing ROGs (SlowUpdate) | Check all detectors' RoGs are triggered for SlowUpdates |
| Timing header errors | XPM/TPR timing distribution issue — check fiber integrity, clock sources |
| Transition buffer exhaustion | Usually during shutdown; if during running, transitions arriving too fast |
| Buffer exhaustion (pebble/DMA near max) | Downstream bottleneck — check file writer and event builder |
| FileWriter stalled (`DRP_bufFreeBlk`=1) | Disk I/O bottleneck — check NFS, disk space, reduce data rate |
| High EB fixup rate | Slow detector — identify via `EB_FxUpSc` histogram and per-detector event rate |
| High EB timeout rate | Detector disconnected or severely slow — check timing links (12s timeout) |
| XPM link down | Timing distribution failure — check XPM, cables, power |
| JESD misalignment | HSD ADC issue — may need HSD re-initialization or firmware update |
| High FEXOOR rate | HSD threshold misconfiguration — adjust feature extraction parameters |
| Uneven detector rates | Per-detector issue — compare `drp_event_rate` across detnames |
| MEB buffer exhaustion (`MRQ_BufCt`=0) | Monitoring clients too slow — simplify AMI graph, reduce plot complexity |
| High `MEB_PrcTmM` | Slow shmem client — reduce monitoring data volume or client complexity |
| MEB split events | Serious error — late contributions after fixup. Investigate timing/trigger. |
| Monitoring gap (MEB rate << TEB rate) | Insufficient MEB buffers or slow clients. Increase `-n` flag or reduce rate. |

---

## Dashboard & Visualization

    # Render any panel as an image
    grafana_get_panel_image(
        dashboardUid="wihghwb",
        panelId=<id>,
        timeRange={"from": "now-5m", "to": "now"},
        variables={"var-instrument": "<instrument>", "var-partition": "<partition>"}
    )

**Key panel IDs:**

| Panel ID | Title | Shows |
|---|---|---|
| 16 | DRP and Readout Groups | Overview (deadtime, L0 rates, TEB/MEB counts) |
| 50 | Rates | `drp_event_rate` per detector |
| 52 | Damage | `DRP_Damage` per detector |
| 109 | FileWriter Free Buffers | `DRP_RecordDepth` |
| 134 | Damage rate | `irate(DRP_Damage[5s])` |
| 136 | File Writer State | `DRP_fileWriting`, `DRP_bufFreeBlk` |
| 182 | Record rate | `rate(DRP_RecordSize)` |
| 252 | Discarded error event rates | All 6 `drp_num_*` error rates |
| 388 | Detector Deadtime | `100*drp_deadtime` |
| 478 | Rate of XPM RxDspErrs | `rate(Us:RxDspErrs)`, `rate(Cu:RxDspErrs)` |
| 113 | XPM LinkUp | `Us:RxLinkUp`, `Cu:RxLinkUp` |
| 479 | HSD PGP Link Status | `loclinkrdy`, `remlinkrdy` |
| 480 | HSD JESD Link Status | `RxDataNAlign` |
| 483 | FEXOOR Rate | `rate(fexoor)` |

**Generate shareable links:**

    grafana_generate_deeplink(
        resourceType="dashboard",
        dashboardUid="wihghwb",
        timeRange={"from": "now-5m", "to": "now"}
    )

    grafana_generate_deeplink(
        resourceType="panel",
        dashboardUid="wihghwb",
        panelId=50,
        timeRange={"from": "now-5m", "to": "now"}
    )

---

## Source Code Reference

| File | Role |
|------|------|
| `psdaq/drp/DrpBase.cc` | Core PGP reader, all 6 DRP error counters, damage tracking |
| `psdaq/drp/PGPDetector.cc` | DRP metric registration for PGP-based detectors |
| `psdaq/drp/TebReceiver.cc` | TEB result processing, file writing, MEB posting |
| `psdaq/psdaq/eb/src/teb.cc` | TEB main: trigger decisions, MRQ buffer management |
| `psdaq/psdaq/eb/src/EbAppBase.cc` | Event builder base: EB metrics, fixup/timeout logic |
| `psdaq/psdaq/eb/src/EventBuilder.cc` | Epoch/event sorting, timeout detection |
| `psdaq/psdaq/monreq/monReqServer.cc` | MEB main: buffer management, all MEB metrics, shmem dispatch |
| `psdaq/psdaq/eb/src/MebContributor.cc` | DRP-side MEB posting logic |
| `psdaq/psdaq/eb/src/TebContributor.cc` | DRP-side TEB posting logic |
| `psdaq/psdaq/xpm/PVPStats.cc` | XPM statistics (L0InpRate, DeadFrac) — C++ |
| `psdaq/psdaq/pyxpm/pvstats.py` | XPM statistics — Python |
| `psdaq/psdaq/cas/epics_exporter.py` | EPICS PV → Prometheus bridge |
| `psdaq/psdaq/service/MetricExporter.cc` | Prometheus exposition (HTTP on ports 9200–9299) |
| `xtcdata/xtcdata/xtc/Damage.hh` | Damage bitmask enum |
| `psdaq/psdaq/eb/src/eb.hh` | System constants (EB_TMO_MS, buffer sizes, tick rate) |
