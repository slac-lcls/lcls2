# Changelog

## [Unreleased]

### Added

- Stateful GPU tasks: `GpuTask(factory=...)` builds one `GpuTaskWorker` per
  CUDA execution stream with run-owned, budget-reserved state (`ctx.state`),
  `begin_step`/`end_step` hooks placed per stream at scope change, and keyed
  export through `ctx.emit` to a per-BD reducer and user sink. Phase 1 of the
  design in [#270](https://github.com/slac-lcls/lcls2/pull/270): step scans
  only ([#344](https://github.com/slac-lcls/lcls2/pull/344)). Example:
  `psana/psana/gpu/examples/step_cube.py`.

### Fixed

- Reduced psana GPU input submission overhead by validating group plans together,
  tracking occupied slots directly, and polling only input windows eligible for
  retirement. @monarin [#298](https://github.com/slac-lcls/lcls2/pull/298)
