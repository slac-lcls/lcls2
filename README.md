# LCLS II development repository
## Build instructions
### Note: build on psbuild-rhel6 where redhat gcc6 compilers are installed
```bash

# repository consists of seperate packages: xtcdata, psdaq, drp and psana

# build all packages in the repository and install them in ./install, option to choose build type
./build_all.sh -c {Release, Debug, RelWithDebInfo} -p {develop, install}
```

To run the psana automated tests run "nosetests psana/psana/tests" in your git root directory.

You can read the above build_all.sh script to see how to build individual packages.
