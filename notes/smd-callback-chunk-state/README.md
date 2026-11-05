# Callback chunk-state regression

This branch implements the regression-test step only. Runtime callback behavior
is unchanged. The test asserts that selecting events by physical scan step and
event position gives the same timestamps for every internal chunk/batch size.

Run in a locally built psana environment with `xtcwriter`, `smdwriter`, and
`mpirun` on PATH:

```bash
python -m pytest psana/psana/tests/test_callback_chunk_state.py -v
python -m pytest psana/psana/tests/test_callback_chunk_state.py --runxfail -v --tb=short
```

The first command should currently report 8 passes and 16 strict expected
failures. The second intentionally exposes the unfixed bug as test failures.
Once the callback lifetime is fixed, remove the split-chunk xfail marks; strict
XPASS already makes the normal test run fail when the marks become obsolete.

The fixture generates two XTC streams containing three steps of ten events each,
with SlowUpdate transitions. A separate serial direct-file read of stream 0
supplies the timestamp oracle, bypassing SMD chunking and callback machinery.
The fixture checks step lengths and unique, ordered timestamps before testing
selection.

The 24 cases combine:

- Serial and four-rank MPI (one SMD0, one EB, two BD ranks).
- SMD chunk sizes 100 (whole-run control), 4, and 7 (split steps).
- Output batch sizes 1 and 3.
- All physical steps or physical step 1 only, selecting event indices `[2:5]`.

The event window follows the existing Jungfrau predicate `evskip <= ievt <
nrecs`: here `nrecs` is the upper index, not a count added to `evskip`. No
Jungfrau semantics are changed. The test callback drains unselected steps too,
so its trace can prove every input event was visited once in acquisition order.
MPI output is sorted across BD ranks without deduplication, retaining detection
of missing, extra, or duplicated selected events.

Only `CallbackSelectionMismatch` is expected to fail. Generator/reader errors,
lost input events in the callback trace, invalid fixture data, subprocess
failures, missing tools, and timeouts remain failures. Subprocess timeouts kill
the process group, including MPI workers. Only external logbook discovery is
stubbed; the actual SMD/EB/BD paths execute. Calibration loading and monitoring
are disabled for this synthetic test.

Failure messages include actual/expected timestamp lists and callback traces of
`[invocation, istep, ievt, timestamp]`. These show the reset directly and can be
used to evaluate the later framework fix. Multi-EB global state, pedestal
computation, and performance validation are outside this step.
