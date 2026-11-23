# Changelog

All notable changes to this project will be documented in this file.
See [docs/CHANGELOG_GUIDE.md](docs/CHANGELOG_GUIDE.md) for how to add entries.

## [Unreleased]

### Added

- Added `psdaq/AGENTS.md` and rules in `psdaq/docs/` for AI assistants editing the GPU DRP or reading DAQ measurements. @rclaus2 [#300](https://github.com/slac-lcls/lcls2/pull/300)
- Added sm_120 (RTX PRO 6000 Blackwell) to the CUDA architectures built. @rclaus2 [#300](https://github.com/slac-lcls/lcls2/pull/300)
- Added the GPU DRP metrics `DRP_tigSleeps` and `DRP_colBlocks`, counting when the trigger-input receiver sleeps and when the Collector blocks. @rclaus2 [#300](https://github.com/slac-lcls/lcls2/pull/300)
- Added ePixUHR calibration and a CALIB mode for taking darks to the GPU DRP, with the Reader/Reducer performance work that runs five GPU DRPs on one node at 33 kHz, and per-stage metrics. @rclaus2 [#269](https://github.com/slac-lcls/lcls2/pull/269)
- Added reading a `drp_pva` detector's serial number from `<prefix>SerialNumber_RBV`, for calibration. @chrisvam [#288](https://github.com/slac-lcls/lcls2/pull/288)
- Added the event builder metrics `TCtbO_TxOut<i>`, `MCtbO_TxOut<i>` and `TEB_TxOut<i>`, the writes outstanding on each DRP-to-TEB, DRP-to-MEB and TEB-to-DRP link, and `*_TxErCt`, their failed writes and timeouts. @rclaus2 [#313](https://github.com/slac-lcls/lcls2/pull/313)
- Add the `parkOnJump` kwarg to help with debugging jumping L1s when you need to keep the devGui running.

### Changed
- @msdubrovin [#327](https://github.com/slac-lcls/lcls2/pull/327) epix_deploy_constants - exclude epixm320 from the panel loop
- @msdubrovin [#324](https://github.com/slac-lcls/lcls2/pull/324) epix_deploy_constants - add loop over segments for constants merging
- @msdubrovin [#293](https://github.com/slac-lcls/lcls2/pull/293) extend command det_dark_proc with --ctdepl, rename alvium.py to epics2d.py, merge it with axis.py
- Changed the GPU DRP's trigger-input receiver to back off when there are no triggers instead of keeping a core at 100%, and its Collector to time out partial TEB batches as the CPU DRP does. @rclaus2 [#300](https://github.com/slac-lcls/lcls2/pull/300)
- Changed the TEB, MEB and DRP to detect a reused or overwritten RDMA buffer slot when it happens and report the source, slot and pulse IDs; `-k eb_slot_check=0` turns this off. @rclaus2 [#289](https://github.com/slac-lcls/lcls2/pull/289)
- Changed control's BeginRun phase 1 timeout from 6 s to 12 s, since fetching calibration constants can take several seconds. @rclaus2 [#269](https://github.com/slac-lcls/lcls2/pull/269)
- Changed the DAQ and psana tests to run on every push to a PR. @mavaylon1 [#267](https://github.com/slac-lcls/lcls2/pull/267)
- Changed an event builder write that stays blocked to fail after 30 s instead of 300 s, and to abort, naming the peer, where carrying on would lose a transition buffer. @rclaus2 [#313](https://github.com/slac-lcls/lcls2/pull/313)

### Deprecated

### Removed

- Removed the stale `psdaq/drpGpu/FINDINGS.md`, `RULES.md` and `TODO.md`. @rclaus2 [#300](https://github.com/slac-lcls/lcls2/pull/300)

### Fixed

- Fixed a DRP with few DMA buffers deadlocking at high trigger rates: the pebble now defaults to more than `MAX_TEBS` TEB batches (512 buffers), and a smaller or non-power-of-2 `pebbleBufCount` is refused at startup. @rclaus2 [#350](https://github.com/slac-lcls/lcls2/pull/350)
- Fixed the GPU DRP not finding a multi-panel ePixUHR detector's calibration constants from one panel's serial number, and taking panel 0's constants for every panel. A failed fetch now tells the operator "failed to fetch calibration constants", with the details in the log. @rclaus2 [#348](https://github.com/slac-lcls/lcls2/pull/348)
- Fixed BeginRun taking minutes with many scalar PVA detectors: psana no longer looks up calibration constants for detector types that never use them (pv, BLD, timing, epics, wave8, hsd, ...), so they also no longer get another detector's constants from the detector DB. @chrisvam [#333](https://github.com/slac-lcls/lcls2/pull/333)
- Fixed epixuhr `raw()`/`calib()` padding to `max(segment number)+1` panels when only some panels are read out. They now return one panel per read-out segment, matching the full-detector constants and geometry found by partial-name matching, so `image()` works. @chrisvam [#338](https://github.com/slac-lcls/lcls2/pull/338)
- Fix bug that causes non-recorded shmem data with run number zero to not use the latest calibration constants @chrisvam [#322](https://github.com/slac-lcls/lcls2/pull/322)
- Fixed missing h5 data overflow error created by move to numpy2. @chrisvam [#303](https://github.com/slac-lcls/lcls2/pull/303)
- Fixed calibration prefetch replacing usable cached constants with empty fetch results, and prevented unreadable caches or caches with incompatible detector IDs from being used to prepare detector caches. @chrisvam @monarin [#323](https://github.com/slac-lcls/lcls2/pull/323)
- Fixed the 1 ms waits in the TEB batch timeout, `EbLfServer::pend()` and `SPSCQueue::pop()` ending early because they were timed on a clock that steps every ~1 ms. TEB, MEB and DRP threads blocked up to ~1000 times a second at full rate, and TEB batches were flushed early. @rclaus2 [#300](https://github.com/slac-lcls/lcls2/pull/300)
- Fixed a GPU DRP hanging on its first event when its trigger primitive has no GPU `event()`, or when it is not in `buildDets`. The first now fails Configure with a message. @rclaus2 [#300](https://github.com/slac-lcls/lcls2/pull/300)
- Fixed `MebContributor::post()` copying an L1Accept into the MEB buffer before checking its size. @rclaus2 [#300](https://github.com/slac-lcls/lcls2/pull/300)
- Fixed `NoOpReducer` reading GPU device 0's properties instead of the DRP's own GPU's. @rclaus2 [#300](https://github.com/slac-lcls/lcls2/pull/300)
- Fixed psana SMD callbacks restarting their step and event counters at chunk boundaries, which could skip selected steps or repeat event-selection windows in serial and MPI processing. @monarin [#287](https://github.com/slac-lcls/lcls2/pull/287)
- Fixed parallel psana hanging silently when the MPI layout leaves an EventBuilder with no BigData ranks. `Communicators` now rejects such layouts up front with the numbers spelled out. @amandashack [#249](https://github.com/slac-lcls/lcls2/pull/249)
- Fixed a multi-lane DRP stalling silently when one lane stops delivering, as in #273. The reader now logs, once, which lanes arrived and which are missing for the stuck event, and counts it in the `drp_num_lane_stalls` metric. A partial event left by the stall no longer survives a Reset or an evtCounter restart, where it used to combine with the next event into a duplicate transition and abort the DRP; such events are counted in `drp_num_stale_events`. @rclaus2 [#282](https://github.com/slac-lcls/lcls2/pull/282)
- Fixed `XtcMonitorServer` and `ShmemClient` not saying why `mq_open` failed. They now name the exceeded system limit (e.g. `fs.mqueue.msg_max`) or other cause and how to fix it, and errors no longer appear ahead of earlier output in logs. @rclaus2 [#276](https://github.com/slac-lcls/lcls2/pull/276)
- Fixed `epix100_config`, `epixquad_config`, and `epixquad1kfps_config` so the PCIe firmware package is imported against its own copy of surf instead of the FEB package's copy. @ruck314 [#252](https://github.com/slac-lcls/lcls2/pull/252)
- Fixed crash of epix100_monitor_ioc process on startup due to a missing import @ddamiani [#295](https://github.com/slac-lcls/lcls2/pull/295)
- Fixed large event builder writes stalling indefinitely on the libfabric `tcp` provider, which hung the MEB on Configure at UED (#277): every write now completes, and writes on a link that goes quiet keep moving. @rclaus2 [#313](https://github.com/slac-lcls/lcls2/pull/313)
- Fixed a CPU DRP occasionally leaving an event out of every batch at high rate, so it was never processed or freed and the TEB later aborted with "missing src" and "Pulse ID did not advance". New metrics `drp_num_batch_gaps` and `drp_num_collector_skips` count batch gaps and skipped events, and an inadequate MEMLOCK limit is now a warning rather than fatal. @rclaus2 [#329](https://github.com/slac-lcls/lcls2/pull/329)
- Fixed argument passing in Jungfrau geometry routines which throw hard errors after the move to NumPy v2.
- Fixed psana silently truncating XTC paths longer than 200 characters by storing filenames as Python lists instead of fixed-width NumPy arrays, and fixed its Logger wrapper's missing `exception()` method masking setup errors in `files=` mode. @monarin [#345](https://github.com/slac-lcls/lcls2/pull/345)
