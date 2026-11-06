Independent review of b6628b0735a0723af5e2c9897d6b42605c943690, base
b59c2fd1aad012de195b14b62401bc966e6755c3. Scope is the new directory
notes/smd-callback-chunk-state/jf-exact-command only. No production edits.
User requested the exact Mikhail 50-frame full-detector command with the same
protections: unfixed default, unfixed PS_SMD_N_EVENTS=10000, another fixed build
at default. User retains implementation/review loop for the whole session.

Review scope.md, README.md, scripts, committed results/commands/per-case reports
and artifact manifest. Check that instrumentation preserves CLI/filter/array
semantics, safety measures match claims, source isolation is evidenced, negative
control is specific, and conclusions match results with appropriate limits.
The two source versions differ only in Python core/test files; the shared native
extensions are unchanged. This is explicitly a separate fixed checkout/overlay,
not a native rebuild or reproduction of Mikhail's complete installed version.
Pinned baseline71378658a is the same unfixed version as earlier experiments.

Coordinator executed all three commands, exit codes1,0,0; no timeouts. Unset
environment becomes1000 in both default cases due to _force_mfx_overrides in
psana/psana/datasource.py, not the generic get_smd_n_events default20000. Default
unfixed:4chunks/4callbacks,0events,UnboundLocalError ievt,0gates. Workaround:
1chunk/1callback,50events,2gates. Fixed:3chunks/1callback,50events,2gates. Actual
full raw shape32x512x1024. Both successful input sequences match physical step2
first50 oracle timestamps and both uint16 gate arrays are finite/exactly equal.
The fixed job stops after its50 frames, so it need not consume the transition
tail chunk observed by the all-empty failing job. Chunks' view_size counts
include transitions, so sizes1011/1012/1013 are not numbers of L1 events.

Read-only name lookups returned jungfrau_000005; HTTP post/put/delete helper
guards were never triggered. Literal -o ./work1 in separate fresh cwd suppresses
central startup log. Local disk discovery replaces logbook lookup; all cases
use the same ten SMD files. No deploy flags. Raw input data untouched.
Assertions in analyze.py passed, byte compilation passed, git diff --check and
commit hooks passed. No core test rerun required for these experiment-only notes;
earlier unchanged core had125passed5skipped2deselected and byhand4passed.

Raw evidence is /tmp/jf-exact-command/{unfixed-default,unfixed-10000,fixed-default}/
stdout.log and report.json; gates below respective work1/jungfrau/block_results.
Oracle /tmp/jf-callback-ab/oracle.json was generated earlier by direct SMD-file
DataSource iteration bypassing callback chunking (see prior jf-workflow-ab/prepare.py).
All raw paths/hash sizes in artifact-manifest.json. You may read raw text evidence
if accessible; cannot execute tests or independently load binary numpy arrays.
Distinguish coordinator runtime evidence from independently checked source.
Do not treat inaccessible raw evidence as an independently verified runtime pass.

No shell, edits, MCP, delegation or nested review loop. Return structured review
for the exact SHA, stable finding IDs JFEX-R1-NNN, explicit blocking flags, and
clear limitations. This is a scoped experiment review, not a new core redesign.
