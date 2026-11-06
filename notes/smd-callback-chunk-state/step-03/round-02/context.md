Independent review, attempt 2. Attempt 1 timed out at 900 seconds without any
findings or verdict. No code changed; do not assume earlier approval.

Review commit 7e2d5d3714b9ca1ef87cc9be8a1dc5879cbeed8b against base
1aff72ca844f17153fe75377f88a2ca47e32a3a9. Please keep the review focused on
changed production code and give a concise structured result. Do not spend the
review window reconstructing unrelated psana subsystems. No commands, tests,
edits, delegation or nested review loop. Inspect source using allowed read tools.

User requested step 3: wire one persistent callback/run across SMD chunks into
serial and MPI production batching, bound callback output even for reject-all
callbacks, preserve transitions/destinations/metadata, and finish the upstream
protocol after callback early return. All 16 former regressions must pass.
Local state is per EB rank, not globally coordinated multi-EB counters.

Read these changed files first:
1. psana/psana/psexp/callback_stream.py (new, ~150 lines): greenlet pauses the
   callback stack from deep transition append. Same thread, copied ContextVars.
   Serializes pending records; count/bytes/event bounds, single oversized record
   exception; one output destination per batch. Explicit close/callback errors.
2. psana/psana/psexp/callback_iterator.py: persistent input from step 2, now drains
   remaining transitions after normal return and preserves primary errors.
3. psana/psana/psexp/smdreader_manager.py: callback branch in __next__,
   _callback_builders, close, and reset at BeginRun in get_next_dgrams.
4. psana/psana/psexp/eventbuilder_manager.py: new CallbackEventBuilderManager.
5. psana/psana/psexp/node.py: EventBuilderNode.start now gets _batch_managers,
   new _batch_managers supplies a run-wide MPI source for callbacks, closes by
   draining to empty protocol message. start_broadcast uses the same input.
   Other routing/StepHistory machinery remains existing; inspect only as needed.
6. psana/psana/psexp/serial_ds.py runs finally closes callbacks; events.py closes
   on serial termination. pyproject.toml declares greenlet>=3.1,<4.

Relevant pre-existing APIs if needed: RunSmallData.steps/events in run.py
(around line 802); Step.events in step.py; native ProxyEvent.as_dgrams copies
payloads into bytearray-backed Dgrams, verified in step 2; gen_bytearray_batch
in eventbuilder.pyx defines the packet layout (the new stream avoids its typed
native recycler). SMD0 sends final missing step history then an empty terminator.

Tests: test_callback_chunk_state.py/run_callback_chunk_state.py cover the real
serial/single-EB MPI path, exact timestamps, one invocation and independent
chunk counts; test_callback_stream.py/run_callback_stream.py cover reject-all,
early return, metadata, mixed default/explicit destinations, tiny limits,
repeated runs, max_events, stress backpressure, close, and cleanup failure.
test_callback_iterator.py covers buffer ownership after deliberately poisoning
old chunks. Oracle is a direct-file DataSource, bypassing callback machinery.

Both psana-required test groups ran on EXACTLY this commit, with no later code
or test edits, current checkout Python overlay and unchanged local native libs:
- core pytest psana/psana/tests/: 125 passed, 5 existing skips, 2 default slow
  deselections, NO xfails; exit 0; 346.34 seconds.
- pytest psana/psana/tests/byhand_*: 4 passed; exit 0; 117.31 seconds.
- Compile, whitespace and commit hooks passed. Python3.9.20 ps_20241122;
  greenlet3.2.5 installed only into /tmp/smd-callback-step03/deps. Both groups
  used PS_PARALLEL=mpi and distinct TEST_XTC_DIR directories.
Logs unavailable in restricted checkout (supplied coordinator evidence):
/tmp/smd-callback-step03/core.log SHA256
d386e4fe8c80152b2393ee1e94752edb1aa8f12b8bce0135d020db611a85b4cb;
byhand.log SHA256 00dc2b92e4e2934f5970ac7cdc2339849958933be86bd403fcb8d281dcd98ac8.
Do not claim to execute tests or verify inaccessible log hashes.

Read step-03/README.md for the bounded-output scope: existing input buffers,
EnvStore/StepHistory caches and user-held events are outside the new bound.
Normal callback return drains through run end; it does not instantly cancel
live acquisition. Step-2 suggestions addressed: final-SHA validation, preserving
callback error over cleanup failure, and serializing parsed transition records.
Initial status.json in the frozen commit predates these completed test runs.

Report any blocking correctness/lifetime/integration issue, grounded in source.
Use stable IDs and the exact SHA. Keep the result concise; a long inventory of
every inspected line is unnecessary. This is the second of three allowed review
attempts, with a 15-minute limit per invocation.
