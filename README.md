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
repository's locked Conda environment, then run:

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
prefixes, incremental Python-only refreshes, CUDA builds, and generation of a
Perlmutter runtime setup script.

#### Shared Perlmutter build and activation

Build on a Perlmutter login node. Start from a fresh `master` checkout and an
activated shared build environment, then select versioned build and install
directories. The `--perlmutter-setup` option writes a runtime script only after
the build and installed-package verification succeed:

```bash
base=/global/cfs/cdirs/lcls/users/psana2_build
module load conda/Miniforge3-24.11.3-0 gcc-native/13.2
conda activate "$base/envs/psana-build"

cd "$base/lcls2"
git switch master
git pull --ff-only
commit=$(git rev-parse --short=12 HEAD)

./build_psana.sh --clean -j 8 \
    --prefix "$base/releases/$commit" \
    --build-dir "$base/build/$commit" \
    --perlmutter-setup "$base/setup_env_perlmutter.sh"
```

The generated setup script records the Conda environment and install prefix
used for that successful build. All users of the shared installation can then
activate the same runtime with one command:

```bash
source /global/cfs/cdirs/lcls/users/psana2_build/setup_env_perlmutter.sh
python -c "import psana; print(psana.__file__)"
```

For safety, `--clean` accepts custom targets only below the documented
`$base/releases` and `$base/build` directories. The checkout-local
`install_psana` and `builddir_psana` defaults are also accepted.

This is separate from the repository's `setup_env.sh`, which prepares the
SLAC/full-repository build environment. Regenerating the Perlmutter script with
a newly verified versioned prefix promotes that release for subsequent users;
existing shells remain on the release they already activated.

For calibration access from an off-site host, `LCLS_CALIB_HTTP` is a service
base URL; for example, set it to `https://pswww.slac.stanford.edu/ws`. Do not
append `/calib_ws/`, because psana adds that path itself.
