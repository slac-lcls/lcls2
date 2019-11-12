# LCLS II development repository [![Build Status](https://travis-ci.org/slac-lcls/lcls2.svg?branch=master)](https://travis-ci.org/slac-lcls/lcls2)

## Build instructions:
### Note: build on psbuild-rhel7 where redhat gcc7 compilers are installed

```bash

# build all packages in the repository and install them in ./install, option to choose build type
# most developers can eliminate all the arguments to build_all.sh
source setup_env.sh
./build_all.sh -c {Release, Debug, RelWithDebInfo} -p {develop, install}
```

To run the psana automated tests run "pytest psana/psana/tests/" in your git root directory.

You can read the above build_all.sh script to see how to build individual packages.

