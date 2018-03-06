#!/bin/bash

path='/nvme1n1/hfdata.h5'
rm -f $path
#h5c++ -o HFWrite HighFreqHDFWrite.cc
./HFWrite $path | tee -a "test_results/HF_2bytes.txt"


