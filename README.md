# LCLS II development repository
## Build instructions
## Note: build on psbuild-rhel6 where redhat gcc6 compilers are installed
```bash

# repository consists of seperate packages: xtcdata, psdaq, drp and psana
# all packages depend on xtcdata

source setup_env.sh
./build_all.sh
```


You can read the above build_all.sh script to see how to build individual packages.  You can change between optimize/debug builds by running cmake with the following:
```bash
cmake -DCMAKE_BUILD_TYPE={Debug, RelWithDebInfo, Release} ..
```
