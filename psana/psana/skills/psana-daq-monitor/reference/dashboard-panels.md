# Dashboard, Visualization, and Tuning Recommendations

Panel IDs, deeplink examples, and the finding→recommendation table for
`psana-daq-monitor`. `SKILL.md`'s diagnosis groups (A–I) cite this file for
"which panel shows this" and "what to recommend" rather than repeating it.

---

## Dashboard & Visualization

    # Render any panel as an image
    grafana_get_panel_image(
        dashboardUid="wihghwb",
        panelId=<id>,
        timeRange={"from": "<window.start>", "to": "<window.end>"},
        variables={"var-instrument": "<instrument>", "var-partition": "<platform>"}
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
        timeRange={"from": "<window.start>", "to": "<window.end>"}
    )

    grafana_generate_deeplink(
        resourceType="panel",
        dashboardUid="wihghwb",
        panelId=50,
        timeRange={"from": "<window.start>", "to": "<window.end>"}
    )

Note `timeRange` is accepted here and by `get_panel_image` — these are
rendering calls, not `query_prometheus`. See `SKILL.md`'s "The one query
shape" section for why `timeRange` must not be passed to
`query_prometheus` itself.

---

## Tuning & Action Recommendations

| Finding | Recommendation |
|---------|---------------|
| Measured zero event rate (not an empty/no-series result) | Check run control, detector power, timing links — see `SKILL.md`'s standing principle before escalating to "DAQ not running" |
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
| MEB buffer exhaustion (`MRQ_BufCt`=0) | Monitoring clients too slow — simplify AMI graph, reduce plot complexity (corroborate first, see `metric-groups.md` §I) |
| High `MEB_PrcTmM` | Slow shmem client — reduce monitoring data volume or client complexity |
| MEB split events | Serious error — late contributions after fixup. Investigate timing/trigger. |
| Monitoring gap (MEB rate << TEB rate) | Insufficient MEB buffers or slow clients. Increase `-n` flag or reduce rate. |
