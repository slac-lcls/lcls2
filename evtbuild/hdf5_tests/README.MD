## HDF5 high-frequency data testing scripts


This folder contains two testings scripts for measuring the read/write performance of HDF5. A dataset is grown by appending to the dataset one element at a time. The read/write performance is measured by accessing one element at a time. The tests are repeated with varying the chunk size.


The test cases are 
1. Small amounts of data: `HighFreqHDFWrite.cc, HighFreqHDFRead.cc`
   - 2 bytes are added to the dataset at a time
   - The addition is repeated one million times
   - The chunk size is varied in powers of two from one to 2^19


2. Variable length images: `VarLenHDFWrite.cc, VarLenHDFRead.cc`
   - 1 MB images are added to the dataset one at a time
   - The addition is repeated ten thousand times
   - The chunk size is varied in powers of two from one to 2^13


The scripts can be called with `HFbuild.sh` or `VLbuild.sh`. The write path for the .h5 file must be set in these scripts.  

A text file of the results will be written out to the test_results/ directory. The results can be plotted with matplotlib using test_results/plot_HF_HDF5.py.


### Test results comparing HF to VL
```
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./VLWrite /nvme1n1/vldata.h5 10000 1024 1000000
1024                10000               1000000             7058                10000.49            0.25                1416.90             1.42                
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./HFWrite /nvme1n1/vldata.h5 10000 1024 1000000
1024                10000               1000000             13835               10240.00            1.02                740.15              0.72                
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./VLWrite /nvme1n1/vldata.h5 10000 1024 1000000
1024                10000               1000000             7247                10000.49            0.25                1379.95             1.38                
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./HFWrite /nvme1n1/vldata.h5 10000 1024 1000000
1024                10000               1000000             12964               10240.00            1.02                789.88              0.77                
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./HFWrite /nvme1n1/vldata.h5 10000 16 1000000
16                  10000               1000000             9191                10000.03            1.00                1088.02             1.09                
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./HFWrite /nvme1n1/vldata.h5 10000 16 1000000
16                  10000               1000000             9143                10000.03            1.00                1093.74             1.09                
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./VLWrite /nvme1n1/vldata.h5 10000 16 1000000
16                  10000               1000000             12230               10000.51            0.25                817.70              0.82                
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./VLWrite /nvme1n1/vldata.h5 10000 16 1000000
16                  10000               1000000             15442               10000.51            0.25                647.62              0.65                
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./HFWrite /nvme1n1/vldata.h5 10000 16 1000000
16                  10000               1000000             8252                10000.03            1.00                1211.83             1.21                
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./HFWrite /nvme1n1/vldata.h5 10000 16 1000000
16                  10000               1000000             8958                10000.03            1.00                1116.32             1.12                
(ps-0.0.6) drp-tst-oss10:~/lcls2/evtbuild/hdf5_tests$ ./VLWrite /nvme1n1/vldata.h5 10000 16 1000000
16                  10000               1000000             7401                10000.51            0.25                1351.24             1.35 
``` 
