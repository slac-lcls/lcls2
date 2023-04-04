traingenerator -s 14000 -b 28 -n 32000 -r 2 -N 1 -d "burst" -t 910000 >& /tmp/beam.py
periodicgenerator -p 910000 9100 -s 0 0 -d '1Hz' '100Hz' --repeat 3 >& /tmp/codes.py 
seqprogram --seq 0:/tmp/codes.py 3:/tmp/beam.py --pv DAQ:NEH:XPM:3 
