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
CI workflow now includes all three new callback test files and permits MPI
oversubscription on small runners. Remote CI is checked after PR creation.
