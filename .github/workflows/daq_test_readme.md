# DAQ test triage for CI (`.github/workflows/run_daq_tests.yaml`)

Every `test_*.py` file in this directory, reviewed for whether it's safe to run in GitHub Actions. Checked against the actual CI environment file (`.daq_20250402.txt`) for which optional dependencies (`rogue`, `PyQt5`, `typer`, `pymongo`) are really present, not just assumed.

## Environment file: gdb removed (again)

`gdb-15.1-py39hc71013c_1` was removed from `.daq_20250402_r9.txt`. Installing it on a fresh GitHub Actions runner fails conda's environment creation step entirely:

```
LinkError: post-link script failed for package conda-forge::gdb-15.1-py39hc71013c_1
location of failed script: /usr/share/miniconda/envs/build-conda-env/bin/.gdb-post-link.sh
stderr: /usr/share/miniconda/envs/build-conda-env/etc/conda/deactivate.d/deactivate-gxx_linux-64.sh: line 68: CONDA_BACKUP_CXX: unbound variable
```

gdb's post-link script activates/deactivates the pinned C++ compiler (`gxx_linux-64-13.3.0`) as part of a self-check. `deactivate-gxx_linux-64.sh` tries to restore `$CXX` from `$CONDA_BACKUP_CXX`, but that backup variable was never set (the corresponding activate script never ran first in this context), so the reference is to an undefined variable. That script runs with bash's `nounset` (`set -u`) in effect, so the undefined reference is a hard error, and conda rolls back the *entire* environment-creation transaction — not just gdb.

This is a known class of bug in conda-forge's compiler activation/deactivation scripts (they aren't written defensively, e.g. `${CONDA_BACKUP_CXX:-}`) against being invoked before the matching activate script has run — see [conda/conda#3200](https://github.com/conda/conda/issues/3200) and [conda/conda#9966](https://github.com/conda/conda/issues/9966).

This isn't the first time this exact problem has hit this repo's CI. It was already found and fixed once on `.daq_20250402.txt`:

- `fce5929d9` — "Remove gdb from ci conda env."
- `0cfda798c` (#82) — "Remove gdb from ci conda env." (the change that stuck)

`.daq_20250402_r9.txt` (this branch's `first try` commit) is a separately regenerated lockfile, not derived from the already-fixed `.daq_20250402.txt`, so it silently reintroduced `gdb-15.1` and the same failure. None of the tests in `run_daq_tests.yaml` need a debugger, so gdb is dropped from this file the same way it was dropped before.

## Add to the pytest step — safe as-is

Currently included in `run_daq_tests.yaml`.

- `test_run_slurm_with_retries.py`, `test_sbatch_env_dump.py`, `test_sbatch_manager_output_path.py` — despite the names, none of these touch real SLURM. `test_run_slurm_with_retries.py` writes a fake `sbatch` shell script to a temp dir and puts it first on `PATH`, testing the Python retry logic only. The other two use `monkeypatch`/`tmp_path` fixtures and only check generated command strings or computed paths, no subprocess call to `sbatch`/`srun` ever happens.
- `test_daqmgr_spawn_console.py`, `test_daqmgr_strict_job_info.py` — fully mocked (`monkeypatch` on `Popen` and on `run_slurm_with_retries`), no real process spawning or SLURM calls.
- `test_subproc.py` — self-contained `asyncio` subprocess tests using `sys.executable`, no external DAQ dependencies.

## Add, but as its own separate step — not pytest, plain script

Not yet added to either workflow.

- `test_shlex.py` — has no `pytest.skip()`, no `import pytest` at all, and runs to completion fine via `python psdaq/psdaq/tests/test_shlex.py` (hits `if __name__ == '__main__': run()`). The logic (`fix_env_whitespace`) is real and exercised, but there are no `assert` statements anywhere, it just prints input/output pairs for a human to read. Needs to be run as a plain script step, not collected by pytest, and someone still has to eyeball the output to judge correctness.

## Don't add — currently dead, zero signal either way

Not included in either workflow.

- `test_configdb.py` — contains a real, well-written, `assert`-based test (`Test_CONFIGDB.test_one()`), but it's unreachable in *every* invocation mode: line 2 is an unconditional `pytest.skip("skip so we can avoid pymongo dependency", allow_module_level=True)`, which raises immediately regardless of whether the module is collected by pytest or run directly with `python test_configdb.py`. Execution never reaches the class definition, `run()`, or the `if __name__ == "__main__":` block. Confirmed `pymongo` isn't even in `.daq_20250402.txt`, so this would need both the skip line removed and `pymongo` added to the environment before it could ever run again. Adding it to CI as-is produces a skip notice, not coverage.

## Needs verification before trusting in CI — will actually execute, not skip

Not yet added to either workflow — each of these has an optional-dependency guard, but the dependency is confirmed present in `.daq_20250402.txt`, so the guard won't trigger and the real code path will run.

- `test_rogue.py` — just `import rogue`, but the test's own comment says: *"do this simple test since rogue has a rather complex boost dependency which we have gotten wrong in the past."* `rogue` is present in the environment file, so this will genuinely attempt the import on a fresh GitHub Actions runner, not skip. Worth running in isolation first.
- `test_daqstat.py` — gated with `pytest.importorskip("PyQt5")`. `PyQt5` is present, so it won't skip, it calls `daqstat.main()` with `-h`/`-v`/`--help`/`--version`. Headless GitHub runners sometimes need `QT_QPA_PLATFORM=offscreen` even for help-text-only invocations, depending on how the app constructs its `QApplication`. Worth a standalone run before trusting it in the full suite.
- `test_json2xtc.py` — a real test with real assertions, but it silently returns (reports as **passed**, not skipped) if the `json2xtc` executable isn't found on `PATH`: `if not exe_found: return`. Needs confirmation that `build_all.sh -d` actually builds and installs `json2xtc` before this can be trusted, otherwise it's a false green that tests nothing.

## Files outside this directory, checked but out of scope here

Two other `test_*.py`-named files exist elsewhere under `psdaq/` and were checked for completeness, neither belongs in this list:

- `psdaq/psdaq/configdb/test_merge.py` — same situation as `test_shlex.py` (no `test_*` functions, demo-only `main()`), and additionally has a latent bug (`trim()` references an undefined `old_keys` variable) that happens not to trigger given the specific inputs in its own demo call.
- `psdaq/psdaq/control_gui/test_QWZMQListener.py` — not a test at all. An interactive helper script with an infinite `while True:` loop under `__main__`, meant to manually feed fake ZMQ messages to something else. It also binds a ZMQ socket at module import time, outside any function, so even collecting it (without running the `__main__` block) does something. Should never be swept in by a broad glob pattern.
