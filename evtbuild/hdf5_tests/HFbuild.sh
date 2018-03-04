#!/bin/bash

rm -f /nvme1n1/hfdata.h5  
#h5c++ -o HFWrite HighFreqHDFWrite.cc
#./HFWrite
./HFbuild.sh | tee -a "test_results/1M_evts_chunk_varied_2_ints.txt"


