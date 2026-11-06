# Changelog

All notable changes to this project will be documented in this file.
See [docs/CHANGELOG_GUIDE.md](docs/CHANGELOG_GUIDE.md) for how to add entries.

## [Unreleased]

### Added

- Added ePixUHR calibration and a CALIB mode for taking darks to the GPU DRP, with the Reader/Reducer performance work that runs five GPU DRPs on one node at 33 kHz, and per-stage metrics. @rclaus2 [#269](https://github.com/slac-lcls/lcls2/pull/269)

### Changed

- Changed the TEB, MEB and DRP to detect a reused or overwritten RDMA buffer slot when it happens and report the source, slot and pulse IDs; `-k eb_slot_check=0` turns this off. @rclaus2 [#289](https://github.com/slac-lcls/lcls2/pull/289)
- Changed control's BeginRun phase 1 timeout from 6 s to 12 s, since fetching calibration constants can take several seconds. @rclaus2 [#269](https://github.com/slac-lcls/lcls2/pull/269)
- Changed the DAQ and psana tests to run on every push to a PR. @mavaylon1 [#267](https://github.com/slac-lcls/lcls2/pull/267)

### Deprecated

### Removed

### Fixed

- Fixed psana SMD callbacks restarting their step and event counters at chunk boundaries, which could skip selected steps or repeat event-selection windows in serial and MPI processing. @monarin [#287](https://github.com/slac-lcls/lcls2/pull/287)
- Fixed parallel psana hanging silently when the MPI layout leaves an EventBuilder with no BigData ranks. `Communicators` now rejects such layouts up front with the numbers spelled out. @amandashack [#249](https://github.com/slac-lcls/lcls2/pull/249)
- Fixed a multi-lane DRP stalling silently when one lane stops delivering, as in #273. The reader now logs, once, which lanes arrived and which are missing for the stuck event, and counts it in the `drp_num_lane_stalls` metric. A partial event left by the stall no longer survives a Reset or an evtCounter restart, where it used to combine with the next event into a duplicate transition and abort the DRP; such events are counted in `drp_num_stale_events`. @rclaus2 [#282](https://github.com/slac-lcls/lcls2/pull/282)
- Fixed `XtcMonitorServer` and `ShmemClient` not saying why `mq_open` failed. They now name the exceeded system limit (e.g. `fs.mqueue.msg_max`) or other cause and how to fix it, and errors no longer appear ahead of earlier output in logs. @rclaus2 [#276](https://github.com/slac-lcls/lcls2/pull/276)
