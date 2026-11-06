Focused round2 review at 38010bfb13110f4bef0da3c8d235c466d06946a3.
Previously passed round1 code SHA7ddd3f4fd0b8273ba32ec67ad978d1ffe8801005.
Intervening fd2a469055e06039f393b7d45bf9fc8c26b73809 only appended PR287 link to
CHANGELOG.md (resolves PRCB-R1-002). Latest commit changes only test launcher
and CI env; all seven production modules unchanged. PR287 is open, not merged.

Read psana/psana/tests/test_callback_chunk_state.py run_driver and
.github/workflows/run_psana_tests.yaml; supporting test_callback_stream.py if
needed. This is a narrow correction review, not a full re-audit. No commands,
edits, MCP, delegation or nested review. Return structured verdict for exactSHA
and stable finding IDs PRCB-R2-NNN. Prior findings: legacy callback cleanup
PRCB-R1-001 deferred/out of scope, early-return synchronous drain PRCB-R1-003
documented and deferred with #284. None was blocking in round1.

Reason for correction: GitHub psana job128161793447/run42296144904 on fd2a46905
built successfully but reported20failed80passed3skipped. Every failure was an MPI
case rejected by mpirun before starting ranks: "There are not enough slots
available in the system to satisfy the4slots". CI conda uses Open MPI5.0.5; the
old OMPI_MCA_rmaps_base_oversubscribe=1 setting was visible in logs but ineffective.
All20newMPIcases failed for this launch reason. Serial cases and preexisting tests
passed. DAQ and changelog workflows passed on fd2a46905.

Correction: run_driver now constructs ['mpirun','--oversubscribe','-n','4',...]
for MPI, with a comment explaining small CI hosts. Serial unchanged. Remove the
ineffective workflow env block; keep all three regression test files enrolled.
Explicit flag is documented by Open MPI5 as alias for --map-by :oversubscribe:
https://docs.open-mpi.org/en/v5.0.5/man-openmpi/man1/mpirun.1.html
No topology, selection assertion, timeout or filtering changes. No new runtime
dependency, extra test, or production change.

Coordinator validates the actual test launcher under an explicit two-slot limit:
/tmp/smd-callback-pr/two-slots-bin/mpirun is a temporary wrapper that execs the
conda mpirun with --host localhost:2 followed by all original arguments.
Local runtime Open MPI5.0.6. With -n4 and no explicit flag, hostname probe fails
exit1; with the prior OMPI_MCA_rmaps_base_oversubscribe=1 it also fails exit1.
With --oversubscribe -n4, probe succeeds and prints all4rank hostnames.
Then PATH prefixed with wrapper and new install/bin, PYTHONPATH new install,
old env unset: python -m pytest psana/psana/tests/test_callback_chunk_state.py
psana/psana/tests/test_callback_stream.py -k mpi -v --tb=short runs all20affected
MPI cases under2slots. Exact result appended below before review invocation.

Original fresh-build fullcore142passed5skipped2deselected,byhand4passed and
protected realdata50frame32panel gate equivalence remain applicable to unchanged
production. No reason to rebuild/rerun unrelated groups for a test-only launcher
fix. GitHub CI on the correction runs separately and is not claimed green until
it completes. Raw coordinator logs under /tmp/smd-callback-pr may be inaccessible
to your restricted checkout; distinguish supplied evidence from independent
execution. Review source plus correction rationale, don't infer runtime success
from inaccessible files.

Completed affected-check result:20passed,21deselected,108.76s,exit0.
All20MPI cases exercised through two-slot wrapper, no timeouts. Git diff --check
and repository commit hooks passed. New CI is triggered by pushing38010bfb1.
