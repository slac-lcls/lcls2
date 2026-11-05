Review round 1. User explicitly requested step 1 only of a callback bug plan,
on a separate branch, using the implementation/review loop. This is a regression
test commit, not the runtime fix. Review only the following change from base
71378658a54dcdd941c9137355a985fa4b637595 to
ccd93559debd18a58d8f87c3fb4b52287c7c96fd.

All changed files are new:
- psana/psana/tests/test_callback_chunk_state.py
- psana/psana/tests/run_callback_chunk_state.py
- notes/smd-callback-chunk-state/{README.md,scope.md,validation.md,status.json}

Read scope.md and validation.md in the detached checkout for the contract and
supplied test evidence. Check robustness, independent oracle, relevance to the
real callback path, xfail specificity, MPI isolation/shutdown, and exact-event
selection coverage. Inspect nearby runtime code as needed; it is unchanged.
The test drains unselected steps to allow a complete callback input trace; its
selection predicate matches Jungfrau's current upper-index bound.

Applicable psana guidance: source is authoritative; exp/dir tests exercise
SerialDataSource or MPIDataSource (one SMD0, one EB, two BDs); direct files use
SingleFileDataSource and bypass SMD callback machinery. Both full core pytest
groups are required when changing production psana core. This commit changes
only tests and task records; its required runtime check is the focused matrix,
not full-suite validation or a detector-data performance run.

Acceptance: varying SMD chunks 100/4/7 and batch sizes 1/3, serial and MPI,
all steps and step 1 only, skip 2 and use upper index 5, compare exact timestamps
against direct-file reference. Whole-run controls must pass. Split-chunk cases
must currently fail specifically on selection, be strict xfails under normal
pytest, and fail visibly with --runxfail. No setup/reader/MPI failures may hide
inside expected failures. Bug fixing, multi-EB global state, and deploying or
sending messages are out of scope.

Coordinator-generated validation (not executed by reviewer):
- 8 passed, 16 xfailed; exit 0; 77.83 seconds.
- --runxfail: 16 failed, 8 passed; exit 1; 88.96 seconds.
- All 16 failures were CallbackSelectionMismatch; trace integrity and fixture
  checks passed. See validation.md for an exact trace example and log hashes.
- Python 3.9 compile and git diff --check passed; commit hooks passed.
- Local source Python overlay plus existing locally built native extensions.
- No production-code edits; full core suites not run for this test-only scope.

No previous findings. Maximum three rounds. Do not run another review loop,
commands, shell, MPI, or tests. Do not modify files. Report stable finding IDs
and exact reviewed SHA. A pass must concern the test-only acceptance contract;
the underlying runtime bug is intentionally still present.
