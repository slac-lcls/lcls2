# Validation at the implementation commit

Both required groups ran after committing
`7e2d5d3714b9ca1ef87cc9be8a1dc5879cbeed8b`, with no code/test edits during or
after the runs. HEAD is recorded beside each local log in core.sha/byhand.sha.

| Group | Result | Duration | Code SHA |
| --- | --- | --- | --- |
| `python -m pytest psana/psana/tests/ -v --tb=short` | 125 passed, 5 skipped, 2 deselected; exit 0; no xfails | 346.34 s | `7e2d5d3714b9ca1ef87cc9be8a1dc5879cbeed8b` |
| `python -m pytest psana/psana/tests/byhand_* -q --tb=short` | 4 passed; exit 0 | 117.31 s | `7e2d5d3714b9ca1ef87cc9be8a1dc5879cbeed8b` |

Core includes all 24 production regression cases (the 16 formerly marked xfail
are ordinary passes), 18 iterator tests and 17 stream integration/lifecycle
tests. No fixture, MPI, metadata, routing or timeout failures. The five skips
and two default slow-test deselections are existing availability/configuration
conditions. Compile, whitespace checks and pre-commit hooks passed.

Environment: `source setup_env.sh`, Python 3.9.20 (`ps_20241122`),
`PS_PARALLEL=mpi`, and
`PYTHONPATH=/tmp/smd-callback-step03/deps:/tmp/smd-callback-chunk-state/python:$PYTHONPATH`.
Current source Python is linked through the overlay; unchanged native psana
extensions come from the local install. The existing temporary psalg dgramCreate
build from step 2 is reused. Greenlet 3.2.5 is installed only in the temporary
deps directory. Tests ran with necessary MPI/shared-memory/network access.
The two groups used separate TEST_XTC_DIR paths core-data and byhand-data under
`/tmp/smd-callback-step03`.

| Log under `/tmp/smd-callback-step03/` | SHA256 |
| --- | --- |
| core.log | `d386e4fe8c80152b2393ee1e94752edb1aa8f12b8bce0135d020db611a85b4cb` |
| byhand.log | `00dc2b92e4e2934f5970ac7cdc2339849958933be86bd403fcb8d281dcd98ac8` |

Development-only runs before freezing code: original matrix 24 passed with
--runxfail (79.52 s); focused integration 54 passed (214.87 s); five additional
lifecycle checks passed (16.06 s). The post-commit core run above supersedes
these as final evidence.
