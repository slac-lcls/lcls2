Review step 2, round 1: base 516f25d4a2531d723cfb66f1b69db903f33c1a59,
implementation 60269c77796947d41f69107ff2758d78845be3ba.

The user requested step 2 of an agreed callback-lifetime fix plan and asked to
keep the implementation/review loop active throughout this session. Step 1
added production regressions. Step 2 adds the persistent input/lifetime
component. Step 3 will integrate production batching/refill/routing and bounded
buffers. Read notes/smd-callback-chunk-state/step-02/scope.md and README.md for
the exact contract and limitations. Do not interpret this component as a
completed production fix; existing CallbackBatchBuilder is intentionally
unchanged. Inspect whether the separation and advertised guarantees are sound.

Changed files (all new):
- psana/psana/psexp/callback_iterator.py
- psana/psana/tests/run_callback_iterator.py
- psana/psana/tests/test_callback_iterator.py
- notes/smd-callback-chunk-state/step-02/{scope.md,README.md,status.json,validation.md}

Acceptance: one callback generator and callback-side run per physical run on
one rank; chunk refill must not reset enumerate counters or end a Step; events
and transitions must remain valid after input buffers are recycled. Preserve
scan/EPICS metadata, selected-event routing destinations, cleanup and terminal
behavior, and fresh state for a second run. Verify tests exercise real native
EventBuilders over real generated XTC/SMD data with an independent file oracle.
Test source deliberately overwrites old buffers before refill and on close.

Important integration constraints to examine: native ProxyEvent objects borrow
chunk data; EB.events() supplies owned dgrams. The Python owned record is not
compatible with native proxy recycling. Selected events and pending non-L1
records will need a new serialization sink in step 3. This component does not
bound or drain the transition queue while a callback rejects all events inside
next(); production must do so during consumption. Early close closes its input
iterator but does not implement upstream MPI protocol draining. Multi-EB global
counts, Jungfrau numerical changes, production routing, and performance work
are out of scope for this commit. Flag issues in the component itself even if
production integration is deferred.

Applicable psana guidance: current source is authoritative. Read RunSmallData
in psexp/run.py, psexp/step.py, event.pyx and eventbuilder.pyx as necessary for
event ownership, iteration and serialization. Both full core pytest groups
are required for core implementation changes. Coordinator supplied evidence:
- Final focused tests: 18 passed (66.95 s).
- Full core suite: 91 passed, 5 skipped, 2 deselected, 16 xfailed (268.41 s).
  Collected before adding the before-start parameter; final focused run covers
  it. Production regressions remain strict expected failures intentionally.
- Longer byhand MPI suite: 4 passed (114.45 s).
- Python compile, whitespace checks and commit hooks passed.
- Initial core collection failed due to missing psalg.utils.dgramCreate;
  existing source extension was built into the temporary Python overlay before
  successful retry. No repository source changes for that repair.
- Source Python overlay with locally built native psana extensions; native
  sources were unchanged. See validation.md for commands, paths and log hashes.
  Logs are outside the restricted worktree; report that limitation rather than
  claiming to execute or independently verify the tests.

No unresolved step-2 findings. Step-1 optional chunk-count feedback is addressed
here by independently counting incoming builders, while asserting one callback
invocation. No shell, tests, edits or delegation by the reviewer. Maximum three
review rounds. Report stable finding IDs and the exact implementation SHA.
