#!/bin/bash
 
path='/nvme1n1/vldata.h5'
rm -f $path
# h5c++ -o VLWrite VarLenHDFWrite.cc
./VLWrite $path | tee -a "test_results/VL_1M.txt"
