# DAQ CI (`run_daq_tests.yaml`)

## Test groups

**"Run Tests" step:**
- `test_run_slurm_with_retries.py`, `test_sbatch_env_dump.py`, `test_sbatch_manager_output_path.py`, `test_daqmgr_spawn_console.py`, `test_daqmgr_strict_job_info.py`, `test_subproc.py` — despite the names, none touch real SLURM or spawn real subprocesses; all mocked/self-contained.
- `test_configdb.py` — always reports **skipped**, not passed or failed. It has an unconditional `pytest.skip()` to avoid a `pymongo` dependency that isn't in the CI environment. Included for completeness; expect a skip notice here, not coverage.
- `test_daqstat.py` — needs a display for Qt to construct a `QApplication`, even for `--help`. The step sets `QT_QPA_PLATFORM=offscreen` to handle this on the headless runner.

**Run test_rogue.py step:** 
- `test_rogue.py`. Just `import rogue`, but rogue has a history of broken/fragile boost bindings (per the test's own comment). Kept out of the main pytest list and given its own `timeout-minutes: 5`, so a hang or crash here doesn't take down or obscure the rest of the suite.

**Run test_shlex.py step:**
- `test_shlex.py`. Doesn't use pytest.

**Explicitly not included, and not tests** — two other `test_*.py`-named files exist elsewhere in `psdaq/` and should never be swept in by a broader glob: `psdaq/psdaq/configdb/test_merge.py` (demo script, not a real test) and `psdaq/psdaq/control_gui/test_QWZMQListener.py` (interactive helper that binds a live ZMQ socket at import time).

## Environment nuance: gdb is deliberately not in `.daq_20250402_r9.txt`

If you're wondering why gdb isn't in the pinned environment: `gdb-15.1`'s post-link script fails conda's environment-creation step outright on a fresh GitHub Actions runner (`CONDA_BACKUP_CXX: unbound variable`, inside `gxx_linux-64`'s deactivate script). Because a post-link script failure rolls back the *entire* conda environment, this isn't a "gdb is broken" footnote, rather it blocks the whole workflow before the build even starts.

This isn't the first time this exact package has caused this exact problem in this repo. It was already hit and removed once before, on a different (now-superseded) environment file. If you're adding or updating packages in `.daq_20250402_r9.txt`, know that gdb specifically doesn't work here, and don't be surprised if reintroducing it (or another package with a similar compiler-activation-touching post-link script) reproduces the same failure.
