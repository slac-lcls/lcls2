# Offline validation before hardware

Full standard-library regression tooling suite: 103 tests PASS on Python 3.11.14.
Log: `/pscratch/sd/m/monarin/psana2-gpu/accounting-offline-03.log`
SHA256: `7e09001557eaec448c03253b2c9e41729286bec209308d740244685d42c2d6a4`.

Includes counter malformed/missing/reset/stale/duplicate-device rejection,
false cold/warm coverage and residency/data mismatch rejection, and a synthetic
10k prefix verifying 50k headers and rejecting accepted-hash mismatch. Existing
smoke, adapter, controller, input/runtime/site and cleanup tests remain passing.

Earlier run 01: 102 tests PASS. Run 02: one new test-fixture error attempting to
rewrite an exclusive-create metadata JSON; corrected the temporary fixture, then
reran all 103 tests. No production relaxation. No hardware run yet.
