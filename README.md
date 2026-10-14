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
