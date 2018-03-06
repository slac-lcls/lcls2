#!/bin/bash

rm -f /nvme1n1/hfdata.h5  
#h5c++ -o HFWrite HighFreqHDFWrite.cc
./HFWrite /nvme1n1/hfdata.h5 | tee -a "test_results/2bytes_HF.txt"


