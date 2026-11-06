Independent review of step 3, round 1.
Base: 1aff72ca844f17153fe75377f88a2ca47e32a3a9.
Reviewed implementation: 7e2d5d3714b9ca1ef87cc9be8a1dc5879cbeed8b.

The user requested production integration of the previously reviewed persistent
callback iterator and wants the implementation/review loop for this session.
The user explicitly approved sending code/context to the configured Claude
service. Do not delegate, execute commands/tests, edit files or start a review
loop. Use only the restricted file-reading tools. Report stable IDs and this SHA.

Read notes/smd-callback-chunk-state/step-03/scope.md and README.md. Changed files:
- pyproject.toml (greenlet runtime dependency)
- psana/psana/psexp/callback_stream.py (new bounded same-thread batch stream)
- psana/psana/psexp/callback_iterator.py (drain/cleanup integration)
- psana/psana/psexp/{smdreader_manager,eventbuilder_manager,node,events,serial_ds}.py
- psana/psana/tests/{test_callback_chunk_state,run_callback_chunk_state}.py
- psana/psana/tests/{test_callback_stream,run_callback_stream}.py (new)
- step-03 scope, README and initial status records.

Acceptance: actual serial and MPI production paths must keep one callback
invocation across SMD chunks; exact selections/counters must be invariant under
chunk and batch sizes. Bound pending output even for callbacks that never yield
selected events. Preserve event ownership, scan/EPICS transitions, routing and
cleanup, handle normal callback early return through upstream termination, and
reset per run. All 16 original xfails must become ordinary passes. No-callback
batching must keep working. State remains per EB rank; this is not global
multi-EB scheduling or a Jungfrau numerical/performance change.

Architecture to examine critically: greenlet suspends the callback's Python
stack inside transition append and selected-event serialization. Callback,
source reads and MPI calls remain on the caller thread; initial ContextVars
are copied. Output bounds are selected batch_size, 1000 records, 8 MiB payload
(one oversized record allowed). Serialization happens before retaining records;
the packet footer adds bounded overhead. One destination per output batch avoids
mixing default and explicit routing in the existing node send logic. Transition
history propagation stays with StepHistory. The README explicitly excludes
user-held events, upstream chunks, EnvStore and existing MPI StepHistory from
the new output bound; do not misread this as a globally memory-bounded psana.

Callback return stops event selection, then drains input transitions; it is
not immediate live-acquisition cancellation. The MPI source closes by receiving
the SMD0 protocol terminator. Serial run-generator cleanup/terminate closes
suspended callbacks. A new BeginRun resets the serial callback stream. Follow
these paths in current source, and flag any correctness/lifetime issues.

Applicable psana guidance: source is authoritative; serial exp/dir path is
SmdReaderManager -> callback stream -> Events/EventManager. MPI is SMD0 ->
EventBuilderNode -> callback stream -> BigDataNode -> Events/EventManager.
Direct-file oracle bypasses callback machinery. Both core pytest groups are
required at the reviewed code SHA, including byhand_* MPI tests. No native
psana extension sources changed; local source Python overlays use existing
locally built extensions. Greenlet 3.2.5 was installed only into a temporary
test directory, not the shared environment. Dependency declared >=3.1,<4.

Step-2 non-blocking follow-ups: CBI-R1-001 asks final-SHA core validation;
CBI-R1-002 asks preserving callback errors if cleanup also fails (implemented
and tested); CBI-R1-003 asks accounting for parsed Dgram retention (transitions
now serialize immediately; README states bounds and limitations). They are not
previous step-3 blocking findings. Evaluate the complete final change anew.

Coordinator-generated validation will follow below. The reviewer does not
execute tests. Logs outside the restricted worktree are a declared limitation,
not evidence that the reviewer independently ran or inspected them. Initial
status.json in the frozen commit precedes post-commit core runs; use the exact
SHA evidence supplied below for final results.

FINAL validation at exactly 7e2d5d3714b9ca1ef87cc9be8a1dc5879cbeed8b:
- `python -m pytest psana/psana/tests/ -v --tb=short`: exit 0,
  125 passed, 5 skipped, 2 deselected, no xfails, 346.34 seconds.
- `python -m pytest psana/psana/tests/byhand_* -q --tb=short`: exit 0,
  4 passed, 117.31 seconds.
- Both executed after commit, with HEAD recorded in core.sha/byhand.sha and no
  code/test edits during or after either run. Core includes all 24 production
  regressions, all 18 iterator cases, and all 17 new streaming/lifecycle cases.
- Pre-commit hooks, Python compile and whitespace checks passed.
- Environment: source setup_env.sh; Python 3.9.20 ps_20241122;
  PYTHONPATH=/tmp/smd-callback-step03/deps:/tmp/smd-callback-chunk-state/python
  prepended to local environment, PS_PARALLEL=mpi; TEST_XTC_DIR separately set
  to /tmp/smd-callback-step03/core-data and /tmp/smd-callback-step03/byhand-data.
  Core initial collection needed no repair in step 3; the temporary existing
  psalg.utils.dgramCreate build from step 2 remains in that overlay.
- /tmp/smd-callback-step03/core.log SHA256:
  d386e4fe8c80152b2393ee1e94752edb1aa8f12b8bce0135d020db611a85b4cb
- /tmp/smd-callback-step03/byhand.log SHA256:
  00dc2b92e4e2934f5970ac7cdc2339849958933be86bd403fcb8d281dcd98ac8

The five skips and two default slow-test deselections are existing availability
or test-configuration conditions, not callback failures. No validation blocker.
