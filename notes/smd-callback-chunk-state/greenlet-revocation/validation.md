# Validation

Both required test groups passed on implementation commit
`a64df64d9dffd975e9fbd1ed9e07d264bec5c63f`. No code or test changes followed
that commit during these runs.

| Check | Result |
| --- | --- |
| `python -m pytest psana/psana/tests/ -v --tb=short` | 125 passed, 5 skipped, 2 deselected; 351.51 s; exit 0 |
| `python -m pytest psana/psana/tests/byhand_* -q --tb=short` | 4 passed; 155.42 s; exit 0 |
| Focused callback tests, before commit | 59 passed; 219.51 s |
| Standalone delivery/cleanup driver | Passed with greenlet imports forbidden |
| Whitespace and commit hooks | Passed |

There are no xfails. All 24 production counter regressions, 18 iterator cases
and 17 stream cases passed. Five skips and two slow-test deselections are
existing availability/configuration conditions, as in step 3.

Environment: source setup_env.sh, Python 3.9.20 (ps_20241122),
PYTHONPATH=/tmp/smd-callback-chunk-state/python:$PYTHONPATH, PS_PARALLEL=mpi.
The source overlay follows this checkout and reuses unchanged local native
extensions and the earlier temporary psalg dgramCreate build. The greenlet
installation directory from step 3 is excluded. Both logs begin with an
assertion that greenlet is absent from import discovery. The groups use
separate TEST_XTC_DIR directories under /tmp/smd-callback-revocation and ran
with MPI/shared-memory/network access.

Final logs and full HEAD records are under /tmp/smd-callback-revocation:

| Log | SHA256 |
| --- | --- |
| core.log | `cdcbac08017a1379a2667692b0575045976a64eaf566ed827ede56eead348edb` |
| byhand.log | `848e581740a3e711633e58b64e9ee1912b7f32ab82ca57d03ee9d4be513050fa` |
