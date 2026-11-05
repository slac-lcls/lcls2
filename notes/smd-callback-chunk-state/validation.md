# Validation evidence

Executed 2026-11-05 on SDF. This is test-only work; no production core code was
changed and the full core suites were not run for this scoped regression step.

Environment: local `setup_env.sh`, conda `ps_20241122`, Python 3.9.20,
pytest 8.3.3, local installed xtcwriter/smdwriter and conda mpirun. Source Python
modules were loaded through `/tmp/smd-callback-chunk-state/python/psana`, a
symlink overlay of this checkout's `psana/psana`, with root native extensions
symlinked from `install/lib/python3.9/site-packages/psana`. The imported run.py
path was checked; installed native libraries were built on September 14, after
the last tracked changes to eventbuilder/smdreader/parallelreader (April 10).
No installation was modified.

Commands (from repository root, after setup):

```bash
export PYTHONPATH=/tmp/smd-callback-chunk-state/python:$PYTHONPATH
export PS_PARALLEL=none
python -m pytest psana/psana/tests/test_callback_chunk_state.py -v --tb=short -rx
python -m pytest psana/psana/tests/test_callback_chunk_state.py --runxfail -v --tb=short
```

- Whole-run serial controls: 4 passed, 20 deselected (8.89 seconds).
- Full normal matrix: **8 passed, 16 xfailed**, exit 0 (77.83 seconds).
- Full `--runxfail` matrix: **16 failed, 8 passed**, exit 1 (88.96 seconds).
  All 16 failures are `CallbackSelectionMismatch`, checked from the captured
  output. No fixture, subprocess, MPI, or trace-integrity failures occurred.
- Python 3.9 bytecode compilation and `git diff --check`: passed.

Representative failure (all steps, four-event chunks):

```text
E   test_callback_chunk_state.CallbackSelectionMismatch: execution='serial', chunk_size=4, batch_size=1, stepnum=None
E   expected timestamps: [4294967302, 4294967303, 4294967305, 4294967318, 4294967319, 4294967321, 4294967334, 4294967335, 4294967337]
E   actual timestamps: [4294967302, 4294967303, 4294967307, 4294967308, 4294967321, 4294967322, 4294967326, 4294967327, 4294967334, 4294967335, 4294967339, 4294967340]
E   callback trace [invocation, istep, ievt, timestamp]: [[1, 0, 0, 4294967300], [1, 0, 1, 4294967301], [1, 0, 2, 4294967302], [1, 0, 3, 4294967303], [2, 0, 0, 4294967305], [2, 0, 1, 4294967306], [2, 0, 2, 4294967307], [2, 0, 3, 4294967308], [3, 0, 0, 4294967310], [3, 0, 1, 4294967311], [3, 1, 0, 4294967316], [3, 1, 1, 4294967317], [4, 0, 0, 4294967318], [4, 0, 1, 4294967319], [4, 0, 2, 4294967321], [4, 0, 3, 4294967322], [5, 0, 0, 4294967323], [5, 0, 1, 4294967324], [5, 0, 2, 4294967326], [5, 0, 3, 4294967327], [6, 1, 0, 4294967332], [6, 1, 1, 4294967333], [6, 1, 2, 4294967334], [6, 1, 3, 4294967335], [7, 0, 0, 4294967337], [7, 0, 1, 4294967338], [7, 0, 2, 4294967339], [7, 0, 3, 4294967340], [8, 0, 0, 4294967342], [8, 0, 1, 4294967343]]
```

The expected selection has 9 events, while the actual selection has 12 and
includes incorrect positions. Serial and MPI have the same mismatch. The trace
shows a new invocation restarting `istep` and `ievt` while physical steps span
chunk boundaries. Seven-event chunks also reproduce the error; selecting step
1 loses one of the three expected events in that case.

Full logs are local artifacts (not needed to run the regression):

- `/tmp/smd-callback-chunk-state/control.log` (SHA256 `e6cd36d65a83636768ed31f9c3829b497008fe12f84f6bfedc5c9ca199aacfe3`)
- `/tmp/smd-callback-chunk-state/matrix.log` (SHA256 `62299f385f3b53f0f2e9d007902dc0b99a70ca8ed5103a83d4ed1f03389170ff`)
- `/tmp/smd-callback-chunk-state/runxfail.log` (SHA256 `dd8a5ce57a056c7752b38d4a1301a7c7abced873503583937a2410c62fa0195e`)
