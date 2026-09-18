# Device Config Diagnostics: ConfigDB Fields vs. psana Runtime Consumers

ConfigDB records what configuration was **requested**. psana's own
`_configs`/`_seg_configs()` attributes — populated from XTC Config
transitions baked into the actual data file/shared-memory stream — record
what configuration **actually reached the data** for a given run. These can
diverge (e.g. a config change between runs with no new Configure
transition). This file documents, for each device type, what psana can and
cannot see about a ConfigDB misconfiguration — i.e. which fields have a
traceable runtime consumer in psana, and which are invisible to psana
entirely.

---

## trigger_0 / teb

| ConfigDB field | What it controls | psana consumer (file:line) or "NOT CONSUMED BY PSANA" | Symptom if misconfigured |
|---|---|---|---|
| `buildAll`, `buildDets` | Which detectors' contributions the TEB event-builds | NOT CONSUMED BY PSANA | Invisible to psana; verify via ConfigDB/DAQ logs only |
| `prescale`, `persistValue`, `monitorValue` | Bit-mask rules for per-event persist/monitor/prescale decisions | `psana/psana/detector/ts.py:161-178` (`triginfo_triginfo_0_0_1.prescale()/persist()/monitor()`) — decodes the per-event *result* of these rules (bit-unpacking the triginfo dgram), not the config fields by name | Unexpected persist/monitor bit patterns per event relative to expected trigger logic |

## hsd_0 (HSD digitizer)

| ConfigDB field | What it controls | psana consumer (file:line) or "NOT CONSUMED BY PSANA" | Symptom if misconfigured |
|---|---|---|---|
| `user.fex.gate_ns` | FEX sample-window length | `psana/psana/hsd/hsd.pyx:361-369` (`hsd_raw_2_0_0._load_config`), `hsd.pyx:381-388` (`hsd_raw_3_0_0._load_config`) — converts to `_padLength` via `int(gate_ns*0.160*13/14)*40` | Wrongly-sized/truncated `padded()` waveform array; triggers "Skipping hsd FEX peak out of range" warning at hsd.pyx:181-184 if inconsistent with actual FEX peak data |
| `user.fex.ymin`/`ymax` (legacy `fex.ymin[0]`/`ymax[0]` in `hsd_hsd_1_2_3`, hsd.pyx:103-111) | FEX baseline range | `psana/psana/hsd/hsd.pyx:361-369` — averaged into `_padValue` | Wrong baseline/pad value in reconstructed waveform |
| `user.fex.corr.baseline` | FEX baseline correction | `hsd.pyx:381-388` (`hsd_raw_3_0_0`) — sourced as `_padValue` | Same as above (wrong baseline/pad value) |
| `user.raw.start_ns` | Raw ADC capture window start time | NOT CONSUMED BY PSANA (only gate_ns/ymin/ymax-derived values are read) | Invisible to psana; would only manifest as a raw ADC capture window shifted in time relative to expectations, with no explicit warning. Verify via ConfigDB only |

## timing_0 / ts (per-readout-group eventcode/inhibit)

| ConfigDB field | What it controls | psana consumer (file:line) or "NOT CONSUMED BY PSANA" | Symptom if misconfigured |
|---|---|---|---|
| per-readout-group `eventcode` config | Which event codes are asserted per readout group | `psana/psana/detector/ts.py:106-108` (`ts_ts_0_0_1.eventcodes()`) — decodes the per-event 288-bit `sequenceValues` bitfield (the runtime effect of the config, not the config transition's eventcode field by name) | Missing/extra eventcodes relative to what the experiment's config specifies |
| per-readout-group `inhibit` config | Inhibit rules per readout group | `psana/psana/detector/ts.py:132-133` (`ts_raw_2_1_0.inhibitCounts()`) — reads per-event `inhibitCounts` array (runtime effect) | Unexpected inhibit activity relative to configured inhibit rules |

## jungfrau

| ConfigDB field | What it controls | psana consumer (file:line) or "NOT CONSUMED BY PSANA" | Symptom if misconfigured |
|---|---|---|---|
| `gainMode`, `gain0` | Gain mode selection | `psana/psana/detector/jungfrau.py:45-53` (`jungfrau_raw_0_1_0._seg_configs_user()`), used downstream in `psana/psana/detector/UtilsJungfrau.py` `calib_jungfrau`/`calib_jungfrau_versions` (~line 609, ~851) for gain/pedestal constant selection | Wrong ADU-to-energy conversion, visible as systematically wrong calibrated pixel values |
| `hotPixelThresh` | Hot pixel masking threshold | `psana/psana/detector/jungfrau.py:96-104` (`jungfrau_raw_0_2_0.hot_pixel_thresh()`) reads per-event value | Hot pixels not masked as expected, or over-aggressive masking |

## epix10ka / epixhr2x2 / epixuhr / epixuhr3x2 / epixm320 (epix_base family)

| ConfigDB field | What it controls | psana consumer (file:line) or "NOT CONSUMED BY PSANA" | Symptom if misconfigured |
|---|---|---|---|
| `trbit`, `asicPixelConfig` (or merged `cbitsConfig`) | Per-ASIC/per-pixel gain-mode selection bits | `psana/psana/detector/epix_base.py:76-85` (abstract `_cbits_config_segment`), overridden in `epix10ka.py:47-49`/`104-113` (`UtilsEpix10ka.cbits_config_epix10ka`), `epixhr2x2.py:45-48` (`cbits_config_epixhr2x2`) | Wrong pedestal-subtraction per pixel/ASIC — classic banding/step artifact at ASIC boundaries in `calib()`/`image()` output |
| `gainAsic`, `gainCSVAsic` (epixuhr) | Per-ASIC gain selection | `psana/psana/detector/epixuhr.py:140-148` (`_seg_gainAsics()`/`_seg_gainCSVAsics()`) | Same gain-range mismatch pattern as above |

**Related runtime warning:** `epix_base.py:84` — "epix_base._cbits_config_segment - MUST BE REIMPLEMENTED - return None" — fires if a detector subclass fails to implement gain-bit extraction, silently disabling gain-aware calibration.

## epix100

| ConfigDB field | What it controls | psana consumer (file:line) or "NOT CONSUMED BY PSANA" | Symptom if misconfigured |
|---|---|---|---|
| (no live gain-mode field read) | — | Falls back to hardcoded `GAIN_DEFAULT`/`GAIN_FACTOR_DEFAULT` if calib DB gain is missing — `psana/psana/detector/epix100.py:38-61` (`epix100_raw_2_0_1._gain()`) | — |

**Related runtime warning:** `epix100.py:54` — "gain is missing in calib constants, try to set default" — this is a calib-DB/config mismatch signal worth checking when epix100 values look suspiciously default-ish.

## opal

| ConfigDB field | What it controls | psana consumer (file:line) or "NOT CONSUMED BY PSANA" | Symptom if misconfigured |
|---|---|---|---|
| (any opal config field) | — | NOT CONSUMED BY PSANA — confirmed via grep across `opal.py`/`opal_base.py`; `calib()` (`psana/psana/detector/opal_base.py:44-67`) relies entirely on `_pedestals()`/`_gain()` from the separate calib-constants DB | If an opal misconfiguration is suspected, it must be diagnosed via ConfigDB/DAQ logs only |

## piranha4

| ConfigDB field | What it controls | psana consumer (file:line) or "NOT CONSUMED BY PSANA" | Symptom if misconfigured |
|---|---|---|---|
| (any piranha4 config field) | — | NOT CONSUMED BY PSANA — no config-field reads found anywhere in `piranha4.py`; all classes are thin pass-throughs | Diagnose via ConfigDB/DAQ logs only |

## wave8

| ConfigDB field | What it controls | psana consumer (file:line) or "NOT CONSUMED BY PSANA" | Symptom if misconfigured |
|---|---|---|---|
| `read_only.ChanEnable` | Channel-enable mask | `psana/psana/detector/wave8.py:130-166` (`wave8v1wf_raw_1_0_0.__init__`) | If inconsistent with per-channel descriptors, `_all_wf_same_length` goes False and `raw_all()` returns None (wave8.py:157-166, 203-206) — a silently missing waveform |
| `read_only.DelayN`, `read_only.NumberOfSamplesN` (per channel) | Per-channel unpacking offsets/sample counts | Same `__init__`, used to compute `_offsets`/`_nsamples` for unpacking the packed multi-channel waveform | Same silently-missing-waveform behavior as above if misconfigured/inconsistent |

---

## General runtime signals (not config-specific)

- **Segment completeness silent-drop:** `psana/psana/detector/detector_impl.py:121-136` (`DetectorImpl._segments()`) — if the segment indices present in an event don't exactly match the expected sorted list, the whole detector silently returns `None` for that event rather than raising a warning. Key diagnostic gotcha: a detector "disappearing" from an event is often a segment-count mismatch, not damage.
- **Damage bitmask decode:** `psana/psana/detector/damage.py` (`Damage` class, `_load_damage_info` at damage.py:98-137) — decodes Truncated/OutOfOrder/OutOfSynch/Corrupted/DroppedContribution/MissingData/TimedOut/UserDefined bits per segment per event — general-purpose data-quality signal, reflects DAQ/data-path health rather than ConfigDB misconfiguration specifically, but useful to check alongside config diagnosis.
