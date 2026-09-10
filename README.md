# LCLS II development repository [![Build Status](https://travis-ci.org/slac-lcls/lcls2.svg?branch=master)](https://travis-ci.org/slac-lcls/lcls2)

## Build instructions:

```bash

# build all packages in the repository and install them in ./install, option to choose build type
# most developers can eliminate all the arguments to build_all.sh
source setup_env.sh
./build_all.sh

Possible flags:
-c Compile only - does not make entry points
-f Force clean - recompiles all targets
-d Build daq (automatically set on psbuild)

```

To run the psana automated tests run "pytest psana/psana/tests/" in your git root directory.

You can read the above build_all.sh script to see how to build individual packages.

### Analysis-only psana build

For a local psana development install without psdaq, create and activate the
repository's locked Linux environment (or use an equivalent environment that
provides Meson, Ninja, Cython, NumPy, Hatchling, RapidJSON, and psana's runtime
dependencies), then run:

```bash
conda create --prefix ./.conda-psana --file .daq_20250402_r9.txt
conda activate ./.conda-psana
./build_psana.sh --clean -j 8
source ./install_psana/activate.sh
python -c "import psana; import psana.dgram; print(psana.__file__)"
```

`build_psana.sh` uses the root Meson project, installs xtcdata, psalg, and
psana into `./install_psana`, creates all Python command-line entry points,
and verifies the installed package. Run `./build_psana.sh --help` for custom
prefixes, incremental Python-only refreshes, and CUDA builds.

For calibration access from an off-site host, `LCLS_CALIB_HTTP` is a service
base URL; for example, set it to `https://pswww.slac.stanford.edu/ws`. Do not
append `/calib_ws/`, because psana adds that path itself.
