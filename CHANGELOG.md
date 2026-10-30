# Changelog

All notable changes to this project will be documented in this file.
See [docs/CHANGELOG_GUIDE.md](docs/CHANGELOG_GUIDE.md) for how to add entries.

## [Unreleased]

### Added

### Changed

- Changed the DAQ and psana tests to run on every push to a PR. @mavaylon1 [#267](https://github.com/slac-lcls/lcls2/pull/267)

### Deprecated

### Removed

### Fixed

- Fixed `epix100_config`, `epixquad_config`, and `epixquad1kfps_config` so the PCIe firmware package is imported against its own copy of surf instead of the FEB package's copy. @ruck314 [#252](https://github.com/slac-lcls/lcls2/pull/252)
