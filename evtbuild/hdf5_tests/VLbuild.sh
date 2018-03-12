#!/bin/bash
 
path='/nvme1n1/vldata.h5'
# h5c++ -o VLWrite VarLenHDFWrite.cc
# h5c++ -o VLRead VarLenHDFRead.cc

loop_limit=10000
log_limit=$(echo "l($loop_limit)/l(2)" | bc -l)
it_limit=$(echo "($log_limit+0.5)/1" | bc )

num_bytes=1000000

x=0
while [ $x -lt $it_limit ]
do
    rm -f $path
    chunks=$(echo "2^$x" | bc)
    ./VLWrite $path $loop_limit $chunk $num_bytes| tee -a "test_results/VL_1M_write.txt"
    echo 3 | sudo tee /proc/sys/vm/drop_caches
    ./VLRead $path $x | tee -a "test_results/VL_1M_read.txt"
    x=$(( $x+1 ))
done
