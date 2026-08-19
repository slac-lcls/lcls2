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

## Status summary

All 11 `test_*.py` files in `psdaq/psdaq/tests/` are now wired into `run_daq_tests.yaml` in one of three forms: the main `pytest` step, an isolated `pytest` step of their own, or a plain-script step. Scope here was making sure each one *runs* on a GitHub-hosted runner, not fixing anything about what the test itself checks or how well it checks it — a few of these (`test_configdb.py`, `test_json2xtc.py`) have real quality issues in the test itself, noted below, that are left as-is.

None of this has been exercised by an actual CI run yet — the first real run will confirm whether `test_rogue.py` and `test_daqstat.py` behave as expected on a fresh runner (see their entries below for what to watch for).

## Add to the pytest step — safe as-is

Currently included in `run_daq_tests.yaml`, main `Run Tests` step.

- `test_run_slurm_with_retries.py`, `test_sbatch_env_dump.py`, `test_sbatch_manager_output_path.py` — despite the names, none of these touch real SLURM. `test_run_slurm_with_retries.py` writes a fake `sbatch` shell script to a temp dir and puts it first on `PATH`, testing the Python retry logic only. The other two use `monkeypatch`/`tmp_path` fixtures and only check generated command strings or computed paths, no subprocess call to `sbatch`/`srun` ever happens.
- `test_daqmgr_spawn_console.py`, `test_daqmgr_strict_job_info.py` — fully mocked (`monkeypatch` on `Popen` and on `run_slurm_with_retries`), no real process spawning or SLURM calls.
- `test_subproc.py` — self-contained `asyncio` subprocess tests using `sys.executable`, no external DAQ dependencies.

## Add to the pytest step — runs for real, caveats are on the test itself

Currently included in `run_daq_tests.yaml`, main `Run Tests` step. Unlike the group above, these actually exercise real code paths (their guard/skip conditions don't trigger in this environment), and each has a known wrinkle. None of these wrinkles are being fixed here — only that the test is able to run.

- `test_configdb.py` — always reports as **skipped**, not passed: line 2 is an unconditional `pytest.skip("skip so we can avoid pymongo dependency", allow_module_level=True)`, and `pymongo` isn't in `.daq_20250402_r9.txt`. Included for completeness; expect a skip notice, not coverage, until someone adds `pymongo` to the environment and removes that line — not something this CI setup is fixing.
- `test_json2xtc.py` — has real assertions, but silently returns (reports as **passed**, not skipped) if the `json2xtc` executable isn't found on `PATH`: `if not exe_found: return`. If `build_all.sh -d` doesn't actually build/install `json2xtc`, this is a false green that tests nothing. That's a test-quality issue in `test_json2xtc.py` itself (it should skip or fail loudly instead of silently passing) — not addressed here.
- `test_daqstat.py` — gated with `pytest.importorskip("PyQt5")`, which won't trigger since `PyQt5` is present, so `daqstat.main()` runs for real with `-h`/`-v`/`--help`/`--version`. Headless GitHub runners need a display for `QApplication` construction even for help-text-only invocations, so the `Run Tests` step now sets `QT_QPA_PLATFORM=offscreen` before the `pytest` call. This is an environment fix (in scope), not a test-code fix.

## Add, but isolated in its own step — not bundled into the main pytest run

Currently included in `run_daq_tests.yaml` as its own `Run test_rogue.py (isolated)` step, with a 5-minute `timeout-minutes` safety net.

- `test_rogue.py` — just `import rogue`, but the test's own comment says: *"do this simple test since rogue has a rather complex boost dependency which we have gotten wrong in the past."* `rogue` is present in the environment file, so this genuinely attempts the import on a fresh GitHub Actions runner. Split into its own step (rather than the main pytest list) so that if it hangs or crashes instead of failing cleanly, it doesn't take down or obscure the rest of the suite's results. If it fails cleanly, that's between the DAQ team and the `rogue` package, not something to fix here.

## Add, but as its own separate step — not pytest, plain script

Currently included in `run_daq_tests.yaml` as its own `Run test_shlex.py (manual review, not pytest)` step.

- `test_shlex.py` — has no `pytest.skip()`, no `import pytest` at all, and runs to completion fine via `python psdaq/psdaq/tests/test_shlex.py` (hits `if __name__ == '__main__': run()`). The logic (`fix_env_whitespace`) is real and exercised, but there are no `assert` statements anywhere, it just prints input/output pairs for a human to read. There's no pytest pass/fail signal here — the step runs and exits 0 as long as the script itself doesn't raise; someone still has to read the log output to judge correctness.

## Files outside this directory, checked but out of scope here

Two other `test_*.py`-named files exist elsewhere under `psdaq/` and were checked for completeness, neither belongs in this list:

- `psdaq/psdaq/configdb/test_merge.py` — same situation as `test_shlex.py` (no `test_*` functions, demo-only `main()`), and additionally has a latent bug (`trim()` references an undefined `old_keys` variable) that happens not to trigger given the specific inputs in its own demo call.
- `psdaq/psdaq/control_gui/test_QWZMQListener.py` — not a test at all. An interactive helper script with an infinite `while True:` loop under `__main__`, meant to manually feed fake ZMQ messages to something else. It also binds a ZMQ socket at module import time, outside any function, so even collecting it (without running the `__main__` block) does something. Should never be swept in by a broad glob pattern.
