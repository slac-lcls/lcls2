#!/bin/bash

rm -f smalldata_highfreq.h5  
h5c++ -o HFWrite HighFreqHDFWrite.cc
./HFWrite


