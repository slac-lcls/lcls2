
- Project repo: https://github.com/slac-lcls/lcls2/tree/master/xtc_vol
- Link to README: https://github.com/slac-lcls/lcls2/blob/master/xtc_vol/README.md

## Code documentations
For detailed code documentation in html pages, please download this repo and start with `xtc_vol/docs/html/index.html`.

## Dependencies and CMakeLists.txt settings:
- CMake: version 3.10 or newer.
- XTC: CMakeLists assumes it's installed to `/usr/local`, or you can change the find_library line in "Dependency: XTC" section in CMakeLists.txt.
- MPI: Needed by HDF5, any reasonably new version should work. This repo has been tested with OpenMPI.
	- CMakeLists assumes it's installed to `/usr/local`, or you can change the "include_directories" line in "Dependency: MPI" section in CMakeLists.txt.
- HDF5: Any recent versions of HDF5 develop branch should work. 
	- This repo has been tested with HDF5 develop branch, commit e02e86b78d01c4815ae8c6f13c6d7862701873b1. 
	- CMakeLists assumes it's installed to `../../hdf5_build/hdf5`, or you can change the `set(HDF5_HOME ...)` line in "Dependecy: HDF5" section in CMakeLists.txt. 
	- [Download HDF5](https://bitbucket.hdfgroup.org/projects/HDFFV/repos/hdf5/browse)

## Build XTC_VOL library
To build XTC2_VOL project, please follow these steps:
- Go to LCLS2 directory, assume we have the lcls2 repo at `/home/my_name/lcls2`
- In the source directory (`/home/my_name/lcls2/xtc_vol/`) edit CMakeList.txt
        Find `set HDF5_HOME` line, and change it to your HDF5 install location. It should be the one contains `include/`, `lib/`, `bin/` directories, such as `/home/my_name/hdf5_build/hdf5`.
- Create a build directory, such as `xtc_vol/build`
- `cd xtc_vol/build # currently in xtc_vol/`
- `cmake ..         # currently in build/`
- `make             # currently in build/`

You will see libh5xtc.so (Linux) or libh5xtc.dylib (OSX), that is the file to be loaded by HDF5 runtime.

## Run the demo
To run the demo or any apps with the VOL connector, following environment variables are needed:

    export HDF5_PLUGIN_PATH=/home/my_name/lcls2/xtc_vol/build
    export HDF5_VOL_CONNECTOR="xtc_vol under_vol=0;under_info={};"

After build the xtc_vol library and set environment variables, HDF5 can read xtc2 files. Currently, both xtc2 file and corresponding smd file are needed. 
Use a HDF5 tool h5ls for demostration on sample files (data.xtc and data.smd.xtc2), run following command line to show the HDF5 hierarchy of the xtc2 file:
    
        $HDF5_HOME/bin/h5ls data.xtc2
and h5ls will show the xtc2 data in the HDF5 virtual view. 

    Configure                Group
    BeginRun                 Group
    BeginStep                Group
    Enable                   Group
    L1Accept                 Group
    Disable                  Group
    EndStep                  Group
    EndRun                   Group
    
With **-r** option to show the hierarchy.    

    /                        Group
    //Configure              Group
    //Configure/runinfo_runinfo_runinfo Group
    //Configure/runinfo_runinfo_runinfo/0 Group
    //Configure/runinfo_runinfo_runinfo/0/expt Group
    //Configure/runinfo_runinfo_runinfo/0/runnum Group
    ......
    //BeginRun               Group
    //BeginRun/runinfo_runinfo_runinfo Group
    //BeginRun/runinfo_runinfo_runinfo/0 Group
    //BeginRun/runinfo_runinfo_runinfo/0/timestamps Dataset {1}
    //BeginRun/runinfo_runinfo_runinfo/0/expt Dataset {1}
    //BeginRun/runinfo_runinfo_runinfo/0/runnum Dataset {SCALAR}
    ......
    //L1Accept               Group
    //L1Accept/xpphsd_hsd_raw Group
    //L1Accept/xpphsd_hsd_raw/0 Group
    //L1Accept/xpphsd_hsd_raw/0/timestamps Dataset {1}
    //L1Accept/xpphsd_hsd_raw/0/floatPgp Dataset {SCALAR}
    //L1Accept/xpphsd_hsd_raw/1/timestamps Dataset {1}
    //L1Accept/xpphsd_hsd_raw/1/floatPgp Dataset {SCALAR}
    ......

With **-v** option for more details of metadata.

	......
    //L1Accept/xpphsd_hsd_raw Group
        Location:  3:16-103-0-80
        Links:     1
    //L1Accept/xpphsd_hsd_raw/0 Group
        Location:  3:9-103-0-73
        Links:     1
    //L1Accept/xpphsd_hsd_raw/0/timestamps Dataset {1/1}
        Location:  3:25-103-0-89
        Links:     1
        Storage:   8 logical bytes, 0 allocated bytes
        Type:      native double
    //L1Accept/xpphsd_hsd_raw/0/floatPgp Dataset {SCALAR}
        Location:  3:39-103-0-103
        Links:     1
        Storage:   8 logical bytes, 0 allocated bytes
        Type:      native double
    //L1Accept/xpphsd_hsd_raw/0/array0Pgp Dataset {3/3, 3/3}
        Location:  3:135-103-0-199
        Links:     1
        Storage:   36 logical bytes, 0 allocated bytes
        Type:      native float
    //L1Accept/xpphsd_hsd_raw/0/intPgp Dataset {SCALAR}
        Location:  3:35-103-0-99
        Links:     1
        Storage:   8 logical bytes, 0 allocated bytes
        Type:      native long
    ......
