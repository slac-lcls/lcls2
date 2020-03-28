#!/usr/bin/env python

"""Reds Acqiris data from hdf5 file created by previous script and saves data and runinfo in xtc2 file.
"""

# command to run
# on psana node
# cd .../lcls2
# . setup_env.sh
# python lcls2/psana/psana/hexanode/examples/ex-07-acqiris-h5-to-xtc2.py

#----------

def usage() :
    return '\nUsage:'\
      + '\n  in LCLS2 environment after'\
      + '\n  cd .../lcls2; . setup_env.sh'\
      + '\n  lcls2/psana/psana/hexanode/examples/ex-07-acqiris-h5-to-xtc2.py'\
      + '\n    or with positional arguments:'\
      + '\n  lcls2/psana/psana/hexanode/examples/ex-07-acqiris-h5-to-xtc2.py <IFNAME> <OFNAME> <DIRTMP> <DETNAME> <EXPNAME> <RUNNUM> <DETTYPE> <SERNUM> <NAMESID>'\
      + '\n  lcls2/psana/psana/hexanode/examples/ex-07-acqiris-h5-to-xtc2.py acqiris_data.h5 acqiris_data.xtc2 /reg/data/ana03/scratch/dubrovin/ tmo_quadanode amox27716 100 hexanode 1234 0'\
      + '\n'

#----------

import sys

nargs = len(sys.argv)

IFNAME  = 'acqiris_data.h5'   if nargs <= 1 else sys.argv[1]
OFNAME  = 'acqiris_data.xtc2' if nargs <= 2 else sys.argv[2]
DIRTMP  = './'                if nargs <= 3 else sys.argv[3] # '/reg/data/ana03/scratch/dubrovin/'
DETNAME = 'tmo_quadanode'     if nargs <= 4 else sys.argv[4]
EXPNAME = 'amox27716'         if nargs <= 5 else sys.argv[5]
RUNNUM  = 100                 if nargs <= 6 else int(sys.argv[6])
DETTYPE = 'hexanode'          if nargs <= 7 else sys.argv[7]
SERNUM  = '1234'              if nargs <= 8 else sys.argv[8]
NAMESID = 0                   if nargs <= 9 else int(sys.argv[9])

FNAME_HDF5 = DIRTMP + IFNAME
FNAME_XTC2 = DIRTMP + OFNAME

def convert_hdf5_to_xtc2_with_runinfo() :

    import dgramCreate as dc
    import numpy as np
    import os
    import h5py
    
    nameinfo = dc.nameinfo(DETNAME, DETTYPE, SERNUM, NAMESID)
    alg = dc.alg('raw', [0,0,1])
    
    cydgram = dc.CyDgram()
    
    #---------- for runinfo
    
    runinfo_detname = 'runinfo'
    runinfo_dettype = 'runinfo'
    runinfo_detid   = ''
    runinfo_namesid = 1
    runinfo_nameinfo = dc.nameinfo(runinfo_detname, runinfo_dettype,
                                   runinfo_detid, runinfo_namesid)
    runinfo_alg = dc.alg('runinfo', [0,0,1])
    runinfo_data = {'expt': EXPNAME, 'runnum': RUNNUM}
    
    #----------
    
    ifname = FNAME_HDF5
    ofname = FNAME_XTC2
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
    
        #---------- for runinfo
        if nevt<2 : cydgram.addDet(runinfo_nameinfo, runinfo_alg, runinfo_data)

        cydgram.addDet(nameinfo, alg, my_data)
        timestamp = nevt
        pulseid = nevt
        if   (nevt==0): transitionid = 2  # Configure
        elif (nevt==1): transitionid = 4  # BeginRun
        else:           transitionid = 12 # L1Accept
        xtc_bytes = cydgram.get(timestamp, transitionid)
        #xtc_bytes = cydgram.get(timestamp, pulseid, transitionid)
        f.write(xtc_bytes)
    f.close()

    print('DO NOT FORGET TO MOVE FILE TO /reg/g/psdm/detector/data2_test/xtc/')

#----------

def test_xtc2_runinfo() :
    from psana.pyalgos.generic.NDArrUtils import print_ndarr

    from psana import DataSource
    ds = DataSource(files=FNAME_XTC2)
    orun = next(ds.runs())

    print('\ntest_xtc2_runinfo expt: %s  runnum: %d' % (orun.expt, orun.runnum))

    det = orun.Detector(DETNAME)

    for nev,evt in enumerate(orun.events()):
        if nev>100 : break
        print('Event %d'%nev)
        if getattr(det, 'raw', None) is None :
            print('    --- ev:%d det has no attribute raw...'%nev)
            continue
        print_ndarr(det.raw.times(evt),     '  times : ', last=4)
        print_ndarr(det.raw.waveforms(evt), '  wforms: ', last=4)

#----------

if __name__ == "__main__":
    convert_hdf5_to_xtc2_with_runinfo()
    test_xtc2_runinfo()
    print(usage())

#----------
