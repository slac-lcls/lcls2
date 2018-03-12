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
