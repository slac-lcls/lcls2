Review BOTH deliverables at a64df64d9dffd975e9fbd1ed9e07d264bec5c63f:
(1) removal of greenlet with persistent counters retained, and (2) the exact
GitHub issue payload to publish after review. Base is
570f954d96d23012fd14341405471582b164a728. This is round 1 of at most 3.
No edits, commands, external posting, delegation or nested review loop.

User explicitly requested revoking greenlet and tracking the long no-yield
case separately, with a standard label plus psana2-cpu label/title prefix.
Scope deliberately defers bounded pending transition memory and prompt delivery
during a non-yielding callback call. Do not apply the superseded step-3 requirement
to provide backpressure via stack suspension. Keep the original event-counter
fix: one callback per run, ievt continues across chunks within a physical step
and resets in the callback's next physical step. State is local to the EB rank.

Inspect these files first (the code change is small):
- psana/psana/psexp/callback_stream.py: ordinary generator replaces greenlet;
  serializes transitions into a deque during next(iterator), then emits ordered
  bounded batches. Preserve selected-event routing and cleanup/error behavior.
- psana/psana/psexp/callback_iterator.py: only documentation changed in this
  revocation; inspect implementation as needed to check lifecycle integration.
- pyproject.toml: removes greenlet requirement.
- psana/psana/tests/run_callback_stream.py and test_callback_stream.py:
  import guard forbids greenlet, stress now characterizes delayed delivery,
  finite delivery/routing/metadata/lifecycle coverage remains.
- notes/smd-callback-chunk-state/greenlet-revocation/{README.md,scope.md,
  issue.md,issue.json}: review exact title/body/labels and technical claims.
  Prior root/step-03 README records are explicitly marked historical/superseded.

Production integration from the previous reviewed commit is retained unchanged:
SmdReaderManager supplies per-run EventBuilders for serial; MPI uses
CallbackEventBuilderManager from EventBuilderNode._batch_managers. Source close
drains the upstream MPI terminator. Inspect these only as needed; no unrelated
subsystem audit is requested. Tests use real serial and one-EB MPI paths.

Issue publication evidence/contract: repository slac-lcls/lcls2 currently has
the standard bug label. gh label list found no psana2-cpu label; issue.json
specifies its proposed color/description. Create it only after this review,
then publish the exact reviewed title/body with bug and psana2-cpu. Searches
found no matching existing issue. Review the scope, example, acceptance criteria
and absence of unsupported production incident claims. Transitions are delayed,
not inherently lost; rejected L1 payloads need not all accumulate. No new
execution dependency or API change is pre-authorized as the eventual solution.
Issue body SHA256: 2116c8c3ba4f4c7eb45f722cc9c95b82ba790d90fd13c738d0837627fe632af3

Report a concise structured verdict covering BOTH code and issue payload.
Ground findings in inspected files and supplied test evidence. Required test
results follow below; do not claim to run tests or verify inaccessible logs.
The frozen status.json predates post-commit validation, so use this context's
final-SHA evidence. Unresolved findings: none from earlier revocation rounds.

FINAL-SHA VALIDATION (coordinator evidence)

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
