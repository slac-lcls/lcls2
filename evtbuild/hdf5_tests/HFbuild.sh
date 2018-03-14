#!/bin/bash

path='/nvme1n1/hfdata.h5'
#h5c++ -o HFWrite HighFreqHDFWrite.cc
#h5c++ -o HFRead HighFreqHDFRead.cc

loop_limit=10000
log_limit=$(echo "l($loop_limit)/l(2)" | bc -l)
it_limit=$(echo "($log_limit+0.5)/1" | bc )

num_bytes=1000000

x=0
while [ $x -lt $it_limit ]
do
    rm -f $path
    chunks=$(echo "2^$x" | bc)
    ./HFWrite $path $loop_limit $chunks $num_bytes| tee -a "test_results/HF_1M2_write.txt"
    echo 3 | sudo tee /proc/sys/vm/drop_caches
    ./HFRead $path $x| tee -a "test_results/HF_1M2_read.txt"
    x=$(( $x+1 ))
done
