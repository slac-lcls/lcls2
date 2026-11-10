# Changelog

## [Unreleased]

### Fixed

- Reduced psana GPU input submission overhead by validating group plans together,
  tracking occupied slots directly, and polling only input windows eligible for
  retirement. @monarin
