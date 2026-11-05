# Step 2 validation

Environment: `source setup_env.sh`, Python 3.9.20 (`ps_20241122`), current
checkout Python through `/tmp/smd-callback-chunk-state/python`, and existing
locally built native extensions from `install/lib/python3.9/site-packages`.
The new module was explicitly linked into the overlay. No native psana source
was changed. Test executables are the local xtcwriter/smdwriter and OpenMPI.

All final commands exited 0:

- `PS_PARALLEL=none python -m pytest psana/psana/tests/test_callback_iterator.py -v`:
  **18 passed**, 66.95 s. Includes close-before-start and repeated-close cases.
- `PS_PARALLEL=mpi TEST_XTC_DIR=/tmp/smd-callback-step02/core-data python -m pytest psana/psana/tests/ -v --tb=short`:
  **91 passed, 5 skipped, 2 deselected, 16 xfailed**, 268.41 s.
- `PS_PARALLEL=mpi TEST_XTC_DIR=/tmp/smd-callback-step02/byhand-data python -m pytest psana/psana/tests/byhand_* -q --tb=short`:
  **4 passed**, 114.45 s.
- Python compile and `git diff --check` passed.

Full-suite collection preceded addition of the close-before-start parameter;
the focused run above covers all 18 final cases. The core suite includes the
existing production regression matrix: its 16 strict expected failures remain
because step 2 intentionally does not connect production batching. Skips and
deselections are existing suite configuration/availability conditions.

The first core attempt stopped at collection because the environment lacked
`psalg.utils.dgramCreate`. Built the existing `psalg/psalg/utils/dgramCreate.pyx`
into the temporary overlay with Cython/setuptools and repository xtcdata
headers; no repository source was edited. Both required test groups then
passed using that repaired environment. MPI/shared-memory tests ran with the
necessary system access.

Artifact SHA256 (local, not committed):

| Log under `/tmp/smd-callback-step02/` | SHA256 |
| --- | --- |
| `focused-final.log` | `906dbcbb55c4282877d594d76bb7d1b99246fba9968d9d99b0cb1637aa19af6a` |
| `core-retry.log` | `c8ef85283e2c8c1347270201d6487c9bd4ca2ba7aeaae1e68e4bac0979ef318a` |
| `byhand.log` | `b982b8d767f3b1994781863894f3fa418968aefc9230c160fe6819e3f7302e4a` |

Initial collection failure: `core.log`. Temporary extension build evidence:
`legacy-build.log` and `legacy-build/setup.py` in the same artifact directory.
