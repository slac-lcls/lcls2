Review the exact commit 7ddd3f4fd0b8273ba32ec67ad978d1ffe8801005 against master
9ab5c193b4277e44e3e8ef9563d830c8943fede3. No commands, edits, MCP, delegation or
nested review loop. Read-only source inspection; return structured findings with
stable IDs PRCB-R1-NNN and blocking flags. This is the user's requested
implementation/review loop for a clean PR, not approval to merge.

Scope: preserve SMD callback generator state across chunks in serial/MPI,
per-physical-step counters reset normally, callback invoked once perrun/perEB,
no greenlet. This is the final previously-reviewed synchronous fix ported to
current master. Large experiment notes stay on a separate audit branch, not in
this PR. There are15changed files: seven core Python files (callback_iterator.py,
callback_stream.py, eventbuilder_manager.py, events.py, node.py, serial_ds.py,
smdreader_manager.py), six run/test_callback_*.py, CHANGELOG.md and the existing
.github/workflows/run_psana_tests.yaml. Diff can be inferred from scope and
source; no shell tool. New master MPI-layout validation in node.py must remain.

Inspect correctness, transitions, ownership across input refill, lifecycle/error
handling, destination routing, max_events, early callback return/close, per-run
reset and unchanged no-callback paths. Inspect tests' independent oracle,
timeouts, isolation and new CI enrollment. Review native EventBuilder APIs and
RunSmallData/Step semantics as needed. Keep the known deferred limitation
separate: synchronous callback cannot deliver transitions until yield/return,
and pending transition memory is unbounded during a long no-yield call. #284
tracks that limitation, deliberately deferred by user. Output batches are bounded
but pending queue is not; top callback_iterator docstring now explicitly says so
(resolves prior low-priority REV1-01). No new dependency or unrelated fix requested.

Coordinator fresh-build/test evidence, all completed on the exact implementation:
- Native build from this worktree via build_all.sh -c -j8,177targets completed,
  isolated /tmp/smd-callback-pr/worktree/install. No reuse of old native build.
- Python3.9.20 ps_20241122, PATH isolated install/bin first, PYTHONPATH only
  install/lib/python3.9/site-packages. All seven changed installed Python modules
  byte-identical to checkout. greenlet unavailable. PS_PARALLEL=mpi for groups.
- python -m pytest psana/psana/tests/ -v --tb=short:142passed,5skipped,2deselected,
  355.56s,exit0. All59 new callback tests pass (24counter+18iterator+17stream).
  Skips:calib_pickle_roundtrip,hexanode,timetool,serial_loop_callback,
  serial_steps_w_ts_filter. Two slow tests deselected by existing pytest.ini.
- python -m pytest psana/psana/tests/byhand_* -q --tb=short:4passed,126.70s,exit0.
  Separate TEST_XTC_DIR roots, no production/test edits after freezing commit.
- Whitespace and all commit hooks passed. Hook removed existing trailing spaces
  on the edited workflow shell line; no other unrelated changes.
- New CI includes all three callback test files in explicit psana test list;
  OMPI_MCA_rmaps_base_oversubscribe=1 allows four-rank tests on small runners.

Real-data verification on current master (not the prior old71378658baseline):
exact CLI jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1
--nrecs50 --nrecs1 50 --stepnum2 (actual argv has separate flag/value tokens),
32panels, fresh isolated working directories. Driver invokes real do_main with
observational callback/raw instrumentation. Only access modifications: readonly
detector_name_short(add_shortname=False), DB post/put/delete blocked, local disk
file discovery, literal ./work1 suppressing central startup log. No DBwrite
helpers called. Current-master detached Python uses same new native libraries;
fixed uses new install directly. Fresh direct-SMDfile oracle3stepsx1000unique
timestamps bypasses callback/chunking. Master-default effective1000:4chunks,
4callbacks,0raw events,exit1 with existing ievt UnboundLocalError/no gates.
Master10000:1chunk/1callback,50raw; fixeddefault1000:3chunks/1callback,50raw.
Both successful: exact step2[:50] timestamps, full shape32x512x1024, two finite
uint16gate arrays exactly equal pixel-for-pixel. Imported Python hashes verified
against each checkout; native hashes same. Artifacts /tmp/smd-callback-pr/jf,
logs core.log/byhand.log/build.log and oracle.json; may be inaccessible outside
restricted worktree. Treat runtime results as coordinator evidence, not an
independently executed check. No unrelated application ievt fix in scope.

PR description will follow existing .github/pull_request_template.md: motivation,
test commands above, realdata results, no-greenlet/deferred#284 limitation,
validation/audit links and checklist. CHANGELOG entry under Unreleased/Fixed
has author @monarin; actual PR number/link added after GitHub allocates it.
No Fix#284: that issue must stay open. RemoteCI checked after creation; don't
claim green before it runs. Additional GitHub approval required after final push.
