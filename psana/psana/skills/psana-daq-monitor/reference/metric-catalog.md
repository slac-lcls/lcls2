# Metric Catalog, Architecture, and System Constants

Full metric tables, the data-flow architecture, system constants, and the
source file map for `psana-daq-monitor`. `SKILL.md`'s diagnosis groups
(A–I, in `reference/metric-groups.md`) cite specific rows here rather than
repeating type/label information — look a metric up here when you need its
exact type, labels, or datasource, or the underlying mechanism that
explains why a metric means what `metric-groups.md` says it means.

---

## Dashboard Variables

The L2SI DAQ SDF dashboard uses these template variables. When building
queries, substitute these as label selectors:

| Variable | Label | Values | Usage |
|---|---|---|---|
| `$instrument` | `instrument` | asc, mfx, rix, tmo, tst, txi, ued, xpp | Filter all DRP/EB metrics |
| `$partition` | `partition` | 0–7 | Filter TEB/MEB and per-partition views; **equals the DAQ-process `platform`** — see `SKILL.md`'s "Establish instrument, partition, and window" section for the exact scope of this equivalence and what it does NOT extend to |
| `$group` | `partition` | 0–7 | Filter readout group stats (single group) |
| `$detname` | `detname` | e.g. hsd, epixquad1kfps, jungfrau1M, ... | Filter per-detector views |
| `$prom` | — | datasource selector | Selects which Prometheus instance to query |

Partitions are logical DAQ instances (0–7). Multiple experiments can run
simultaneously on different partitions. The "common readout group" has the
same number as the partition — all detectors in the partition belong to it.

---

## DRP Metrics (prom_psdm — `afjrx1j1815vkf`)

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

## Event Builder Metrics (prom_psdm — `afjrx1j1815vkf`)

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
| `MRQ_BufCt` | Gauge | instrument, partition | Free monitoring buffers (**0 = possible data loss**, see `metric-groups.md` §I caveat) |
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

## DRP-side EB Contributor Metrics (prom_psdm — `afjrx1j1815vkf`)

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

## Readout Group & Timing Metrics

| Metric | Datasource | Type | Description |
|--------|------------|------|-------------|
| `L0InpRate` | prom_psdm | Gauge | L0 trigger input rate (from XPM EPICS PV) |
| `L0AccRate` | prom_psdm | Gauge | L0 trigger accept rate |
| `DeadFrac` | prom_psdm | Gauge | Dead fraction (0–1): Δ(numl0Inh) / Δ(numl0) |
| `RunTime` | prom_psdm | Gauge | Current run elapsed time (seconds) |
| `NumL0Acc` | prom_psdm | Gauge | Total L0 accepts in current run |
| `Us:RxLinkUp` | prom_psdm | Gauge | Ultrafast XPM receive link status — labeled `instrument`, `partition`, `id` (XPM instance); **always scope by these**, see `metric-groups.md` §H |
| `Cu:RxLinkUp` | prom_psdm | Gauge | Copper XPM receive link status — same labeling caveat |
| `Us:RxDspErrs` | prom_psdm | Gauge* | Ultrafast XPM dispatch errors — same labeling caveat |
| `Cu:RxDspErrs` | prom_psdm | Gauge* | Copper XPM dispatch errors — same labeling caveat |
| `loclinkrdy` | prom_psdm_1s | Gauge | HSD PGP local link ready (bitmask) |
| `remlinkrdy` | prom_psdm_1s | Gauge | HSD PGP remote link ready (bitmask) |
| `RxDataNAlign` | prom_psdm_1s | Gauge | HSD JESD link alignment status |
| `fexoor` | prom_psdm_1s | Gauge* | HSD feature extraction out-of-range |

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

- **XPM**: FPGA timing master. Distributes pulse IDs and timestamps at
  ~929 kHz, generates L0 triggers, enforces flow control via inhibits.
  Publishes `L0InpRate`, `L0AccRate`, `DeadFrac`, `RunTime`, `NumL0Acc` as
  EPICS PVs bridged to Prometheus via `epics_exporter.py`.

- **DRP**: Data Readout Processor — one per detector. Three thread types:
  Reader (PGP DMA), Worker (format XTC, compute trigger primitives),
  Collector (batch assembly). Sends trigger inputs to TEB via RDMA,
  receives results, writes XTC2 + SMD files locally.

- **TEB**: Trigger Event Builder. Assembles trigger inputs from all DRPs
  by pulse ID. Runs trigger algorithm to set `persist`/`monitor` flags in
  a `ResultDgram`. Returns results to DRPs via RDMA. Event timeout: 12
  seconds (`EB_TMO_MS`). Up to 4 TEBs (`MAX_TEBS`).

- **MEB**: Monitoring Event Builder. Assembles full events for online
  monitoring. Distributes via POSIX shared memory to clients (AMI,
  psplot, psana-live). Buffer pool circulates: MEB → TEB (MRQ) → DRP
  (RDMA write) → MEB → shmem → client → MEB. Up to 4 MEBs (`MAX_MEBS`).

- **File Writing**: Happens inside each DRP process (not a separate file
  writer). Each DRP writes its own XTC2 data file + SMD index file via
  `BufferedFileWriterMT` (multi-threaded, supports `O_DIRECT`). Files are
  chunked at 500 GB (`DefaultChunkThresh`).

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
