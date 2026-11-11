# Diagnosis Hierarchy — Metric Groups A–I

The ordered diagnostic workflow for `psana-daq-monitor`. `SKILL.md`
establishes `<instrument>`, `<platform>`, and the `[start, end]` window,
and the query shape rules (`queryType="range"` for anything time-
sensitive, plus the one whole-window `increase()` exception) — read it
first. Every PromQL example below omits `queryType`/`startTime`/
`endTime`/`stepSeconds` for brevity; fill them in per `SKILL.md`'s "The
one query shape" section.

Work top-to-bottom. Start with event rate (A) — it is the primary
liveness signal.

## Summarizing a range result by metric type

A metric's *type* determines what its range matters for and how to
summarize it agent-side — don't treat every result the same way:

| Metric type | Examples | How to summarize |
|---|---|---|
| Rates | `drp_event_rate`, `L0InpRate`, `L0AccRate` | min/max/trend across the series; a flat zero stretch within an otherwise-nonzero series is a real finding (an interval where the DAQ was idle or stalled), not noise to average away |
| Fractions / gauges | `DeadFrac`, `drp_deadtime` | peak value and the duration of any interval sustained above the operational threshold (see §B) — a brief spike reads differently from a sustained plateau |
| Monotonic counters | `DRP_Damage`, `EB_FxUpCt`, `EB_ToEvCt`, `drp_num_*` | `increase(metric[window])` for a total, or a `rate()` range query to see *when* increases happened. **Account for process restarts**: a mid-window counter reset shows as a spurious drop in a raw range query; `increase()` handles a single reset correctly, but a visible sawtooth across many points is itself a finding (repeated restarts), not an artifact to discard |
| Occupancy | `MRQ_BufCt`, `DRP_RecordDepth` | the *minimum* value reached and how long the series stayed at/near zero — a brief zero-crossing reads differently from an extended one (see the `MRQ_BufCt=0` corroboration caveat in §I) |
| Status | `Us:RxLinkUp`, `loclinkrdy`, `remlinkrdy` | whether the value was **ever** zero in the window, and for how long — a status metric's history is a boolean-over-time question, not a trend |
| Latency / age | `EB_EvAge`, `MEB_EvtLat` | peak value and whether elevation was sustained vs. a brief transient — a single-sample spike is weaker evidence than an elevated plateau across several consecutive samples |

For a monotonic counter's total over the whole window in one number, use:

```
grafana_query_prometheus(
    datasourceUid="afjrx1j1815vkf",
    expr='increase(<counter>{instrument="<instrument>",partition="<platform>"}[<window-duration>])',
    queryType="instant", endTime="<window.end>"
)
```

— but prefer the range-`rate()` form for anything timing-sensitive (e.g.
"did damage cluster near the start or the end"), since the instant
`increase()` form loses *when* within the window something happened and
requires computing the duration string to exactly match the window.

---

## A. Is the DAQ running? Is the event rate healthy?

**Metric check (fire in parallel):**

    # DRP event rate per detector
    grafana_query_prometheus(
        datasourceUid="afjrx1j1815vkf",
        expr='drp_event_rate{instrument="<instrument>",partition="<platform>"}'
    )

    # L0 input rate (trigger rate into readout group)
    grafana_query_prometheus(
        datasourceUid="afjrx1j1815vkf",
        expr='L0InpRate{instrument="<instrument>",partition="<platform>"}'
    )

    # L0 accept rate (triggers accepted)
    grafana_query_prometheus(
        datasourceUid="afjrx1j1815vkf",
        expr='L0AccRate{instrument="<instrument>",partition="<platform>"}'
    )

    # TEB event rate
    grafana_query_prometheus(
        datasourceUid="afjrx1j1815vkf",
        expr='sum(rate(TEB_EvtCt{instrument="<instrument>",partition="<platform>"}[1m]))'
    )

`L0InpRate`/`L0AccRate` are EPICS PVs from the XPM, bridged to Prometheus
via `epics_exporter.py`. `L0InpRate` = L0 triggers input / time.
`L0AccRate` = L0 triggers accepted / time.

On `L0InpRate`/`L0AccRate`, `partition` is the XPM readout group, not
necessarily the platform — `<platform>` selects only the common group;
other groups need `partition="<rog>"`. A series existing for a group
doesn't by itself mean a DAQ is running there, since all 8 groups are
always exported (see `SKILL.md`'s "Establish instrument, partition, and
window" table).

**Interpretation:**
- `drp_event_rate` > 0 across all expected detectors = DAQ running in that
  window
- `L0InpRate` ≈ `L0AccRate` = minimal deadtime, triggers accepted
  efficiently; `L0InpRate` >> `L0AccRate` = significant deadtime (§B)
- TEB rate ≈ `L0AccRate` = event builder keeping up
- A **measured** zero event rate means no events in this window — not
  necessarily "DAQ not running" (could be no triggers selected for that
  readout group). An **empty** result (no series at all) is the separate,
  more ambiguous case — see `SKILL.md`'s empty-result decision table.
- Mismatched rates across detectors = possible per-detector issue

---

## B. Is there excessive deadtime?

Deadtime is the fraction of L0 triggers rejected because the readout
system is busy. Computed by the XPM as `Δ(numl0Inh) / Δ(numl0)` per
measurement interval. The DRP-side `drp_deadtime` reads the XPM's
per-link `DeadFLnk` EPICS PV — it is NOT computed by the DRP itself, and
it reads the **detector's own readout group** PV (`det_info.readout`),
which is not necessarily the platform.

**Metric check:**

    # Dead fraction per readout group (0–1 scale)
    grafana_query_prometheus(
        datasourceUid="afjrx1j1815vkf",
        expr='DeadFrac{instrument="<instrument>",partition="<platform>"}'
    )

    # Per-detector deadtime percentage
    grafana_query_prometheus(
        datasourceUid="afjrx1j1815vkf",
        expr='100.*drp_deadtime{instrument="<instrument>",partition="<platform>"}'
    )

On `DeadFrac`, `partition` is the XPM readout group, not necessarily the
platform — `<platform>` selects only the common group; other groups need
`partition="<rog>"`. A series existing for a group doesn't by itself mean
a DAQ is running there, since all 8 groups are always exported (see
`SKILL.md`'s "Establish instrument, partition, and window" table).

Report the **peak** value plus the duration of any interval sustained
above the thresholds below over the window — a brief spike and a
sustained plateau are different findings.

**Interpretation** (operational guidelines — no code-defined alarm
thresholds):

| DeadFrac | Meaning |
|---|---|
| < 1% | Healthy — system keeping up with triggers |
| 1–5% | Marginal — monitor closely |
| 5–20% | Concerning — investigate backpressure source |
| > 20% | Critical — significant data loss |

**Follow-up:** if deadtime is high, check event builder health (§G),
buffer depth (§E), and file writing (§F) in that order.

---

## C. Is there damage?

Damaged events have missing or corrupted data from one or more
detectors. The damage field is a **bitmask** — multiple types can be set
simultaneously.

**Damage types** (from `xtcdata/xtcdata/xtc/Damage.hh`):

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

**Metric check** (`DRP_Damage` is a monotonic counter — use `increase()`
for a total, or `irate()`/range-`rate()` to see timing):

    # Total damage count per detector over the window
    grafana_query_prometheus(
        datasourceUid="afjrx1j1815vkf",
        expr='increase(DRP_Damage{instrument="<instrument>",partition="<platform>"}[<window-duration>])',
        queryType="instant", endTime="<window.end>"
    )

    # Damage per step interval, to see when it happened
    grafana_query_prometheus(
        datasourceUid="afjrx1j1815vkf",
        expr='increase(DRP_Damage{instrument="<instrument>",partition="<platform>"}[<stepSeconds>s])'
    )

The range vector's duration must equal `stepSeconds` so consecutive
points tile the window with no gaps — unlike the previous
`irate(...[5s])` form, which only reflected the last few seconds before
each point and left most of the interval between points invisible.

**Interpretation:**
- Zero damage over the window means no damage measured — not an
  independent confirmation of health, though in practice it usually
  tracks that way
- Nonzero but low (< 1/s) = occasional glitches, usually benign
- Sustained damage rate = systematic issue — check hardware links (§H)
  and DRP errors (§D)
- Damage on a single detector = detector-specific problem; damage on all
  detectors = timing or event builder issue

---

## D. Are there DRP errors?

Six types of discarded error events, dropped early in the readout chain
(never reach recording or monitoring). All are registered as
`MetricType::Gauge` in prometheus-cpp despite being monotonically
increasing — use `rate()`/`increase()`.

**Error definitions** (from `psdaq/drp/DrpBase.cc`):

| Metric | Code-derived meaning | Event discarded? |
|---|---|---|
| `drp_num_dma_errors` | DMA transfer error from PGP kernel driver (`dmaErrors[]` nonzero). Hardware-level: CRC failure, link error, FIFO overflow. | **Yes** |
| `drp_num_pgp_jump` | Gap in 24-bit `evtCounter` (expected `m_lastComplete+1`, got different). Events lost between hardware and DRP. PGP link drops, firmware buffer overflow. | **Yes** (skipped events) |
| `drp_num_no_common_rog` | Timing header's `readoutGroups()` bitmask lacks the partition bit. Would break TEB/MEB pulse ID ordering. | **Yes** |
| `drp_num_missing_rogs` | **SlowUpdate transitions only.** Missing one or more expected readout groups (from `rogMask`). Would break psana. | **Yes** |
| `drp_num_th_error` | Bit 7 of timing header control field set by XPM/TPR firmware. Timing distribution error (CRC, clock domain crossing). | **No** — logged but processing continues |
| `drp_num_no_tr_dgram` | Transition buffer pool (128 slots, `TEB_TR_BUFFERS`) exhausted. Can happen during shutdown or transition storms. | **Yes** |

**Metric check (fire all six in parallel):**

    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='rate(drp_num_dma_errors{instrument="<instrument>",partition="<platform>"}[1m])')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='rate(drp_num_pgp_jump{instrument="<instrument>",partition="<platform>"}[1m])')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='rate(drp_num_no_common_rog{instrument="<instrument>",partition="<platform>"}[1m])')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='rate(drp_num_missing_rogs{instrument="<instrument>",partition="<platform>"}[1m])')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='rate(drp_num_th_error{instrument="<instrument>",partition="<platform>"}[1m])')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='rate(drp_num_no_tr_dgram{instrument="<instrument>",partition="<platform>"}[1m])')

**Action:** any sustained error rate > 0 over the window needs
investigation. Cross-reference with hardware link status (§H). Note that
`drp_num_th_error` is the only one of the six that does NOT discard the
event.

---

## E. Are buffers and queues healthy?

Buffer exhaustion causes backpressure → deadtime → data loss. The chain
is: DMA fills → pebble fills → deadtime rises. Trace back to the earliest
bottleneck (usually file writing or event builder).

**Metric check:**

    # FileWriter free buffer depth (should stay > 0)
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='DRP_RecordDepth{instrument="<instrument>",partition="<platform>"}')

    # Pebble buffers in use vs max
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='drp_pebble_in_use{instrument="<instrument>"}')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='drp_pebble_in_use_max{instrument="<instrument>"}')

    # DMA buffers in use vs max
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='drp_dma_in_use{instrument="<instrument>"}')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='drp_dma_in_use_max{instrument="<instrument>"}')

(`drp_pebble_in_use`/`drp_dma_in_use` and their `_max` counterparts carry
no `partition` label per `reference/metric-catalog.md` — `instrument`
alone is the correct scope for these.)

These are occupancy gauges — report the **minimum** reached over the
window and the duration at/near zero/exhaustion.

**Interpretation:**
- `DRP_RecordDepth` → 0 = file writer buffer exhaustion, recording will
  stall
- `drp_pebble_in_use` approaching `drp_pebble_in_use_max` = DRP
  backpressure
- `drp_dma_in_use` approaching `drp_dma_in_use_max` = DMA ring buffer
  filling up
- Growing queue occupancy over the window = system falling behind

**Action:** trace the exhaustion chain back to the earliest bottleneck
(usually file writing (§F) or event builder (§G)).

---

## F. Is file writing keeping up?

File writing happens inside each DRP process (not a separate file
writer). Each DRP writes its own XTC2 data file + SMD index file via
`BufferedFileWriterMT` (multi-threaded, supports `O_DIRECT`).

**Metric check:**

    # File writing state (1 = writing, 0 = not writing)
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='DRP_fileWriting{instrument="<instrument>",partition="<platform>"}')

    # FileWriter blocked waiting for free buffer (0/1)
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='DRP_bufFreeBlk{instrument="<instrument>",partition="<platform>"}')

    # Record rate (bytes/sec being written)
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='rate(DRP_RecordSize{instrument="<instrument>",partition="<platform>"}[1m])')

**Interpretation:**
- `DRP_fileWriting` = 0 when expected to record = run not configured for
  recording, or writing stalled
- `DRP_bufFreeBlk` = 1 = file writer blocked waiting for a free buffer
  from `BufferedFileWriterMT` — backpressure will propagate upstream
- Declining `DRP_RecordDepth` trend (§E) = file system I/O bottleneck

**Action:** check disk I/O, NFS mount health, available disk space.
Files are chunked at 500 GB (`DefaultChunkThresh`). Consider reducing
data rate or adding compression.

---

## G. Is the event builder healthy?

The TEB assembles trigger inputs from all DRPs by pulse ID. Events older
than 12 seconds (`EB_TMO_MS = 12000`) are timed out. "Fixups"
(`EB_FxUpCt`) are incomplete events flushed when a newer complete event
arrives — missing contributors get `DroppedContribution` damage (bit 4).
The `eb` label distinguishes TEB vs MEB instances.

**Metric check:**

    # TEB event and write rates
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='sum(rate(TEB_EvtCt{instrument="<instrument>",partition="<platform>"}[1m]))')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='sum(rate(TEB_WrtCt{instrument="<instrument>",partition="<platform>"}[1m]))')

    # MEB event rate
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='sum(rate(MEB_EvtCt{instrument="<instrument>",partition="<platform>"}[1m]))')

    # Event builder fixup and timeout counts
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='sum(rate(EB_FxUpCt{instrument="<instrument>",partition="<platform>",eb="TEB"}[1m]))')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='sum(rate(EB_ToEvCt{instrument="<instrument>",partition="<platform>",eb="TEB"}[1m]))')

    # Age of oldest incomplete event
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='EB_EvAge{instrument="<instrument>",partition="<platform>",eb="TEB"}')

**Interpretation:**
- `TEB_EvtCt` rate ≈ `L0AccRate` = TEB keeping up; `TEB_WrtCt` rate > 0 =
  events being recorded
- `EB_FxUpCt` rate > 0 = incomplete events flushed (missing contributor).
  The `EB_FxUpSc` histogram shows which source IDs triggered fixups;
  `EB_CbMsMk` is the most recent missing-contributors bitmask
- `EB_ToEvCt` rate > 0 = events exceeding the 12-second timeout (detector
  too slow or disconnected)
- MEB rate = 0 when monitoring expected = monitoring MEB not receiving
  events
- `EB_EvAge` growing = event builder falling behind

Note: fixup/timeout logging in the code goes quiet after 50 cumulative
events to avoid log flooding.

**Action:** high fixup or timeout rates indicate a slow detector. Check
per-detector event rates and latency; the `EB_FxUpSc` histogram
identifies the offending source ID.

---

## H. Are timing and hardware links healthy?

**XPM link metrics have no `partition` label** — `Us:*`/`Cu:*` series come
from the XPM's own EPICS PV bridge, registered via the exporter's `-G`
flag → `registerGlobalPV()` (`psdaq/psdaq/cas/epics_exporter.py:47`,
labels `['instrument','id']`), and **one instrument can have several XPM
exporters** registered (`psdaq/psdaq/cnf/neh-base.py` registers multiple
`-D <hutch>/xpm-N/...` exporters per hutch). Querying `Us:RxLinkUp{}` with
no selector at all mixes every exported XPM across every hutch into one
result — never do this. Scope by `instrument`, and where the target
XPM's identity is uncertain, discover the actual `id` values first rather
than summing or averaging across them silently:

    # Discover which XPM `id` values exist for this instrument
    grafana_list_prometheus_label_values(
        datasourceUid="afjrx1j1815vkf",
        labelName="id",
        matches=[{"filters": [
            {"name": "__name__", "value": "Us:RxLinkUp", "type": "="},
            {"name": "instrument", "value": "<instrument>", "type": "="}
        ]}]
    )

**XPM link status (fire in parallel, scoped):**

    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='Us:RxLinkUp{instrument="<instrument>"}')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='Cu:RxLinkUp{instrument="<instrument>"}')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='rate(Us:RxDspErrs{instrument="<instrument>"}[1m])')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='rate(Cu:RxDspErrs{instrument="<instrument>"}[1m])')

If more than one `id` is returned for the instrument, **inspect each
`id`'s result individually** rather than aggregating with `sum()`/`avg()`
by default — a healthy XPM's result can hide a second, unhealthy XPM's
`RxLinkUp=0` if you sum them.

**HSD hardware status (uses prom_psdm_1s datasource):**

    grafana_query_prometheus(datasourceUid="afotnjy2b05q8b",
        expr='loclinkrdy{instrument="<instrument>"}')
    grafana_query_prometheus(datasourceUid="afotnjy2b05q8b",
        expr='remlinkrdy{instrument="<instrument>"}')
    grafana_query_prometheus(datasourceUid="afotnjy2b05q8b",
        expr='RxDataNAlign{instrument="<instrument>"}')
    grafana_query_prometheus(datasourceUid="afotnjy2b05q8b",
        expr='rate(fexoor{instrument="<instrument>"}[1m])')

These are status metrics — report whether the value was **ever** zero in
the window, and for how long.

**Interpretation:**
- `RxLinkUp` = 0 for any link = timing link down, detector won't receive
  triggers
- Nonzero `RxDspErrs` rate = dispatch errors on timing link, may cause
  trigger issues
- `loclinkrdy` or `remlinkrdy` = 0 = PGP link down between DRP and HSD
- `RxDataNAlign` ≠ 0 = JESD deserialization issue on HSD
- Growing `fexoor` rate = HSD feature extraction seeing out-of-range
  values

**Next checks (read-only; this skill performs no hardware operations):**
link down → visually check cable seating and report it; a power cycle or
timing re-initialization is an operator-run intervention, not a step this
diagnosis performs. JESD issues → report as an HSD firmware/hardware
lead. FEXOOR → report the out-of-range rate; threshold changes are an
operator decision.

---

## I. Is online monitoring healthy? (MEB)

The MEB is the gateway to all online monitoring (AMI, psplot,
psana-live). It has a circular buffer pool that flows through three
measured phases:

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

**Critical failure mode:** when `MRQ_BufCt` (free buffers) drops to 0,
the MEB can't send MRQ requests to the TEB. The TEB's buffer list for
this MEB drains, and it silently sets `monitor(0)` on events
(`TEB_nMonCt` increments). **No error is raised — monitoring events are
just dropped.** Everything looks fine to the DAQ, but AMI/psplot/
psana-live see event gaps.

**Metric check (fire in parallel):**

    # Free monitoring buffers (0 = possible monitoring data loss, see caveat)
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='MRQ_BufCt{instrument="<instrument>",partition="<platform>"}')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='MRQ_BufCtMax{instrument="<instrument>",partition="<platform>"}')

    # Buffers in-flight to shmem clients
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='MEB_PrcCt{instrument="<instrument>",partition="<platform>"}')

    # MEB event rate
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='rate(MEB_EvtCt{instrument="<instrument>",partition="<platform>"}[1m])')

    # Max processing / request-to-build time (ns), min idle time (ns)
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='MEB_PrcTmM{instrument="<instrument>",partition="<platform>"}')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='MEB_TrgTmM{instrument="<instrument>",partition="<platform>"}')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='MEB_AppTmm{instrument="<instrument>",partition="<platform>"}')

    # Event latency (μs), split events (serious error)
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='MEB_EvtLat{instrument="<instrument>",partition="<platform>"}')
    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='MEB_SpltCt{instrument="<instrument>",partition="<platform>"}')

**Interpretation:**
- `MRQ_BufCt` = 0 = the textbook signature of monitoring data loss — MEB
  can't request new events. **Corroboration caveat** (one incident, not
  DAQ-expert-confirmed): a `0` sample can also reflect fast buffer
  turnover between scrapes rather than genuine exhaustion — a real 24h
  xpp sweep saw `MRQ_BufCt=0` on 30 of 33 samples over 8h while
  `rate(MEB_EvtCt)` tracked `rate(TEB_EvtCt)` almost exactly (120.04 vs
  120.03) and `MEB_PrcCt=0` (no buffers held by shmem clients) — i.e.
  monitoring was fully keeping up. Corroborate with `MEB_EvtCt`/
  `TEB_EvtCt` rate parity and `MEB_PrcCt` before calling `MRQ_BufCt=0`
  data loss; if either companion metric looks unhealthy or is
  unavailable, treat `MRQ_BufCt=0` as data loss as before.
- `MRQ_BufCt` / `MRQ_BufCtMax` ratio < 25% = buffer pressure, investigate
  client speed
- `MEB_PrcCt` high = many buffers held by shmem clients — slow AMI/psplot
- `MEB_PrcTmM` high (max processing time, ns) = shmem client is slow
- `MEB_TrgTmM` high (max request-to-build time, ns) = DRP/TEB bottleneck
  on monitoring path
- `MEB_AppTmm` near zero (min idle time) = buffers immediately reused,
  MEB is buffer-starved
- `MEB_EvtLat` high = wall-clock lag between event timestamp and MEB
  processing
- `MEB_SpltCt` > 0 = **serious error** — late contributions after fixup
  created duplicate pulse IDs
- Compare `rate(MEB_EvtCt)` vs `rate(TEB_EvtCt)`: large gap = events not
  reaching monitoring

`MRQ_BufCt` is an occupancy gauge (minimum reached, duration near zero —
apply the corroboration caveat above using the same window for
`MEB_EvtCt`/`TEB_EvtCt`/`MEB_PrcCt`); `MEB_EvtLat` is a latency gauge
(peak, sustained elevation).

**DRP-side MEB backpressure check:**

    grafana_query_prometheus(datasourceUid="afjrx1j1815vkf",
        expr='MCtbO_TxPdg{instrument="<instrument>"}')

High `MCtbO_TxPdg` = DRP's posts to MEB are backing up (MEB overwhelmed).

**Action:** slow monitoring client → simplify AMI graph, reduce plot
complexity. Buffer exhaustion → increase MEB buffer count (`-n` flag).
Split events → investigate timing/trigger issues. If monitoring doesn't
matter, `TEB_nMonCt` incrementing is benign.
