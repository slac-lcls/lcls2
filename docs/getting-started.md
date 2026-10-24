# Getting started

This page collects the build and setup pointers that exist in the repository
(`README.md`, `build_all.sh`, `build_psana.sh`, the `setup_env*.sh` scripts and
the GitHub workflows). It does not replace them: when in doubt, read the script.

## Get the source

Unless a section says otherwise, the commands on this page run from the root
of a checkout. The `dev` version of this site is built from the `master`
branch:

```bash
git clone https://github.com/slac-lcls/lcls2.git
cd lcls2
```

## Choose a build

| You want | Script | What it builds | Install location |
|---|---|---|---|
| Analysis only (psana2) | `build_psana.sh` | xtcdata, psalg, psana (no psdaq) | `./install_psana` |
| Full repository (analysis, optionally DAQ) | `build_all.sh` | xtcdata, psalg, psana; psdaq with `-d` | `./install` (or `$TESTRELDIR`) |

Both use the root Meson project (`meson.build`, `meson_options.txt`).

## Analysis-only build (`build_psana.sh`)

From `README.md`: create the locked Conda environment that ships with the
repository, build, and activate the install.

```bash
conda create --prefix ./.conda-psana --file .daq_20250402_r9.txt
conda activate ./.conda-psana
./build_psana.sh --clean -j 8
source ./install_psana/activate.sh
python -c "import psana; import psana.dgram; print(psana.__file__)"
```

Useful options (from `./build_psana.sh --help`):

- `-p, --prefix DIR`: installation prefix (default `<repo>/install_psana`).
- `-t, --build-type TYPE`: `debug`, `debugoptimized` (default), `release`, `minsize` or `plain`.
- `--build-dir DIR`: Meson build directory (default `<repo>/builddir_psana`).
- `--python-only`: refresh installed Python files and entry points from an existing native build, without compiling.
- `--with-cuda`: allow nvcc detection and the CUDA subprojects (off by default).
- `--perlmutter-setup FILE`: after a successful build, write a sourceable runtime setup script (see `README.md` for the shared NERSC Perlmutter procedure).

## Full build (`build_all.sh`)

`build_all.sh` refuses to run unless `ENV_TYPE` is set, which the setup scripts do:

| Script | Sets `ENV_TYPE` | Notes |
|---|---|---|
| `setup_env_ana.sh` | `ana` | S3DF paths under `/sdf/group/lcls/ds/ana/`; activates `ps_20241122`; sets `PATH`, `PYTHONPATH` and `TESTRELDIR` to `./install`. |
| `setup_env_daq.sh` | `daq` | S3DF paths; activates `daq_20250402_r9`; `build_all.sh` then also builds psdaq. |

```bash
source setup_env_ana.sh   # or setup_env_daq.sh for a DAQ build
./build_all.sh            # -c compile only, -f force clean, -d build DAQ, -j N jobs
```

`README.md` still says `source setup_env.sh`; that script was removed from the
repository on 2026-10-21.

`build_all.sh` runs `meson setup`/`meson compile`/`meson install` into
`builddir`, then `uv pip install .` into `$INSTDIR` (and the same for `psdaq/`
when the DAQ build is on). The `-d` flag turns the DAQ build **on**, as the
comment in `.github/workflows/run_psana_tests.yaml` points out.

The setup scripts contain site-specific paths (SLAC S3DF). Outside S3DF, use
the analysis-only build above.

## Calibration access from outside SLAC

From `README.md`: `LCLS_CALIB_HTTP` is the base URL of the calibration web
service, for example `https://pswww.slac.stanford.edu/ws`. Do not append
`/calib_ws/`; psana adds that path itself. See
Calibration constants.

## Run the tests

```bash
pytest psana/psana/tests/
```

`psana/psana/tests/pytest.ini` deselects tests marked `slow` by default. The
subset of tests that runs in CI is listed in
`.github/workflows/run_psana_tests.yaml`; several other tests need LCLS data
paths or the calibration database. Small xtc2 files used by the tests are in
`psana/psana/tests/test_data/` and `psana/psana/tests/*.xtc2`.

## Synthetic test data

Several examples on this site use `DataSource(exp="xpptut15", run=14, dir=xtc_dir)`.
This is a small synthetic run that the tests write with the `xtcwriter` and
`smdwriter` tools (both installed by the build, see
`xtcdata/xtcdata/app/meson.build`, and on `PATH` after
`source ./install_psana/activate.sh`). To make it yourself, run
`psana/psana/tests/setup_input_files.py` from an empty directory:

```bash
mkdir ~/psana-tutorial && cd ~/psana-tutorial
python /path/to/lcls2/psana/psana/tests/setup_input_files.py
ls .tmp .tmp/smalldata
```

It creates the directory `.tmp` (the script fails if `.tmp` already exists)
with two big-data files, `xpptut15-r0014-s000-c000.xtc2` and
`xpptut15-r0014-s001-c000.xtc2`, and their small-data files in
`.tmp/smalldata/`. In the examples, set `xtc_dir = ".tmp"` and run them from
the same directory; the tests do the same
(`xtc_dir = os.path.join(os.environ.get('TEST_XTC_DIR', os.getcwd()), '.tmp')`
in `psana/psana/tests/ds.py`).

## Wheels and PyPI

`.github/workflows/release.yml` builds wheels with cibuildwheel on tag pushes
and has a PyPI publish step. psdaq is not part of the wheel (comment in
`pyproject.toml`). On 2026-10-24 the PyPI project
[`psana`](https://pypi.org/project/psana/) listed version 4.3.4. This is the
same version as in this repository's `pyproject.toml`, and PyPI lists the same
dependencies as that file. There were wheels for CPython 3.11, 3.12 and 3.13
on Linux x86_64 (manylinux_2_28). Installing it
with `pip install psana` was not tested here; note that the `mpi4py`
dependency needs an MPI library on the system (comment in `pyproject.toml`).
The wheel can be older or newer than the `dev` version of this site.

## Build this documentation

Run these from the repository root (the griffe extension paths in
`mkdocs.yml` are relative to it):

```bash
pip install -r docs/requirements.txt     # plus doxygen (e.g. apt-get install doxygen)
doxygen Doxyfile                         # C++ API -> docs/api/cpp/ (git-ignored)
mkdocs build --strict                    # or: mkdocs serve
```

`mkdocs.yml` sets `strict: true`, and the "C++ API" navigation entry points at
`docs/api/cpp/index.html`, so run Doxygen first. The site is published per
version with [mike](https://github.com/jimporter/mike) by
`.github/workflows/build_versioned_mkdocs.yml`.

A documentation change should touch only comments and docstrings in source
files. To check a branch against `master`:

```bash
python docs/tools/check_comments_only.py origin/master
```
