# Changelog

All notable changes to this project will be documented in this file.
See [docs/CHANGELOG_GUIDE.md](docs/CHANGELOG_GUIDE.md) for how to add entries.

## [Unreleased]

### Added

- Added ePixUHR calibration and a CALIB mode for taking darks to the GPU DRP, with the Reader/Reducer performance work that runs five GPU DRPs on one node at 33 kHz, and per-stage metrics. @rclaus2 [#269](https://github.com/slac-lcls/lcls2/pull/269)

### Changed

- Changed control's BeginRun phase 1 timeout from 6 s to 12 s, since fetching calibration constants can take several seconds. @rclaus2 [#269](https://github.com/slac-lcls/lcls2/pull/269)
- Changed the DAQ and psana tests to run on every push to a PR. @mavaylon1 [#267](https://github.com/slac-lcls/lcls2/pull/267)

### Deprecated

### Removed

### Fixed

- Fixed parallel psana hanging silently when the MPI layout leaves an EventBuilder with no BigData ranks. `Communicators` now rejects such layouts up front with the numbers spelled out. @amandashack [#249](https://github.com/slac-lcls/lcls2/pull/249)
