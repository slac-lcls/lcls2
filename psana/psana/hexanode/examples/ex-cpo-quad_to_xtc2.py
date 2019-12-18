
# command to run
# on psana node
# cd .../lcls2
# . setup_env.sh
# python lcls2/psana/psana/hexanode/examples/ex-cpo-quad_to_xtc2.py

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

#dirtmp = './'
dirtmp = '/reg/data/ana03/scratch/dubrovin/'
ifname = dirtmp + 'hexanode.h5'
ofname = dirtmp + 'hexanode.xtc2'
print('Input file: %s\nOutput file: %s' % (ifname,ofname))

f = open(ofname,'wb')
h5f = h5py.File(ifname)
waveforms = h5f['waveforms']
times = h5f['times']
for nevt,(wfs,times) in enumerate(zip(waveforms,times)):
    my_data = {
        'waveforms': wfs,
        'times': times
    }

    if nevt<10\
    or nevt<50 and not nevt%10\
    or not nevt%100 : print('Event %3d'%nevt)

    cydgram.addDet(nameinfo, alg, my_data)
    timestamp = nevt
    pulseid = nevt
    if (nevt==0): transitionid = 2  # Configure
    else:         transitionid = 12 # L1Accept
    xtc_bytes = cydgram.get(timestamp,transitionid)
    #xtc_bytes = cydgram.get(timestamp,pulseid, transitionid)
    f.write(xtc_bytes)
f.close()

print('DO NOT FORGET TO MOVE FILE TO /reg/g/psdm/detector/data2_test/xtc/')
