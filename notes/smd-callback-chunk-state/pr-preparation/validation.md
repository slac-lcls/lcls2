# Validation on current master

Implementation: 7ddd3f4fd0b8273ba32ec67ad978d1ffe8801005.
Base: 9ab5c193b4277e44e3e8ef9563d830c8943fede3.
Worktree: /tmp/smd-callback-pr/worktree, branch fix/smd-callback-persistence.

Fresh native build: source /sdf/home/m/monarin/lcls2/setup_env.sh, override
TESTRELDIR=/tmp/smd-callback-pr/worktree/install, then build_all.sh -c -j8.
177 build targets completed successfully. Installed Python production files
match source byte-for-byte (build-provenance.json). Test PATH puts this install's
bin first; PYTHONPATH is only this install's lib/python3.9/site-packages. Python
3.9.20 / ps_20241122. No greenlet import is available. The existing installation
was not modified. Build warning about unavailable UID12682's /lscratch directory
did not prevent compilation or installation.

| Check | Result |
| --- | --- |
| python -m pytest psana/psana/tests/ -v --tb=short | 142 passed, 5 skipped, 2 deselected; 355.56s; exit0 |
| python -m pytest psana/psana/tests/byhand_* -q --tb=short | 4 passed; 126.70s; exit0 |
| Current-master exact Jungfrau matrix | Expected failure/control/fixed results; all acceptance assertions passed |
| Whitespace and repository commit hooks | Passed |

Core includes all59 new callback cases (24 counter,18 iterator,17 delivery).
Five existing skips: calib pickle roundtrip, hexanode, timetool, serial loop
callback, serial step timestamp filtering. Two existing slow tests deselected
by pytest.ini. No xfails. Groups used independent TEST_XTC_DIR paths, PS_PARALLEL=mpi.

Real data: exact 50-frame full32-panel stage1 command from Mikhail, run49/step2.
Current-master Python control uses its detached checkout with the same new native
build; fixed case uses the freshly built install directly. No old native build
is involved. Current-master default:0events/exit1/ievt logging exception;
master10000 and fixed default:50correct events,2exactly equal full gate arrays.
Fresh direct-file oracle bypasses callback/chunk selection. Source SHA256 checks
bind observed imports to the respective checkouts. All use same10SMDstreams.
Readonly name lookup, blocked calibration HTTP mutations, isolated ./work1,
no deployment and no central startup log. No blocked mutation calls occurred.

Artifacts: /tmp/smd-callback-pr/{build.log,core.log,byhand.log,jf/,oracle.json}.
jf-results.json, jf-commands.json, per-case reports and jf-manifest.json preserve
compact results/provenance. Large binary output remains temporary storage.
CI workflow includes all three new callback test files. The initial environment
setting intended to permit MPI oversubscription was ineffective in Open MPI5;
the correction below supersedes that setting.

## CI launcher correction

PR287 initial head fd2a46905: changelog and DAQ checks passed, psana build passed,
but20MPI cases failed before launching because the runner exposed fewer than
4slots (80tests passed,3skipped). This exposed a test-launcher setup error.
Commit38010bfb13110f4bef0da3c8d235c466d06946a3 uses explicit
`mpirun --oversubscribe -n4` in the shared callback test helper and removes the
ineffective OMPI_MCA_rmaps_base_oversubscribe workflow setting. Production code
and all assertions are unchanged.

Open MPI5.0.6 local reproduction with `--host localhost:2`: both plain launch and
the old env setting reject4ranks; explicit option starts all4. All20affected MPI
cases then passed through this wrapper (21deselected,108.76s). Second independent
review passes with zero findings. Full core/byhand and realdata evidence above
remains applicable to unchanged production. Final GitHub results are recorded
in status.json and publication.json after the rerun completes.
