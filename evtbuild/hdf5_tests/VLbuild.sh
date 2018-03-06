#!/bin/bash

rm -f /nvme1n1/vldata.h5  
# h5c++ -o VLWrite VarLenHDFWrite.cc
./VLWrite /nvme1n1/vldata.h5 | tee -a "test_results/VL_1M.txt"
