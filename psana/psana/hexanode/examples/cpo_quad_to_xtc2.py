import dgramCreate as dc
import numpy as np
import os
import h5py

config = {}
detname = 'tmo_hexanode'
dettype = 'hexanode'
serial_number = '1234'
namesid = 0

nameinfo = dc.nameinfo(detname,dettype,serial_number,namesid)
alg = dc.alg('raw',[0,0,1])

cydgram = dc.CyDgram()

fname = 'hexanode.xtc2'

f = open(fname,'wb')
h5f = h5py.File('hexanode.h5')
waveforms = h5f['waveforms']
times = h5f['times']
for nevt,(wfs,times) in enumerate(zip(waveforms,times)):
    my_data = {
        'waveforms': wfs,
        'times': times
    }

    cydgram.addDet(nameinfo, alg, my_data)
    timestamp = nevt
    pulseid = nevt
    if (nevt==0): transitionid = 2  # Configure
    else:         transitionid = 12 # L1Accept
    xtc_bytes = cydgram.get(timestamp,pulseid,transitionid)
    f.write(xtc_bytes)
f.close()
