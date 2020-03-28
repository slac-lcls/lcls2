#!/usr/bin/env python

"""Reds detector data from hdf5 file created by previous script and saves data and runinfo in xtc2 file.
"""

# command to run
# on psana node
# cd .../lcls2
# . setup_env.sh
# python lcls2/psana/psana/hexanode/examples/conv-02-h5-to-xtc2.py

#----------
import os
import sys
from psana.pyalgos.generic.Utils import do_print #, get_login, 

def usage() :
    scrname = sys.argv[0]
    return '\nUsage:'\
      + '\n  in LCLS2 environment after'\
      + '\n  cd .../lcls2; . setup_env.sh'\
      + '\n  %s' % scrname\
      + '\n    or with positional arguments:'\
      + '\n  %s <IFNAME> <OFNAME> <DIRTMP> <EXPNAME> <RUNNUM> <DETNAME> <DETTYPE> <SERNUM> <NAMESID>' % scrname\
      + '\n  %s amox23616-r0104-e400-xtcav.h5 data-amox23616-r0104-e000400-xtcav.xtc2 /reg/data/ana03/scratch/dubrovin/ amox23616 104 xtcav camera 1234 0' % scrname\
      + '\n  %s amox23616-r0131-e200-xtcav.h5 data-amox23616-r0131-e000200-xtcav.xtc2 /reg/data/ana03/scratch/dubrovin/ amox23616 131 xtcav camera 1234 0' % scrname\
      + '\n  %s amox23616-r0137-e100-xtcav.h5 data-amox23616-r0137-e000100-xtcav.xtc2 /reg/data/ana03/scratch/dubrovin/ amox23616 137 xtcav camera 1234 0' % scrname\
      + '\n'

#----------
#exp=amox23616:run=104

nargs = len(sys.argv)

IFNAME  = 'tmp-data.h5'       if nargs <= 1 else sys.argv[1]
OFNAME  = 'tmp-data.xtc2'     if nargs <= 2 else sys.argv[2]
DIRTMP  = './'                if nargs <= 3 else sys.argv[3] # '/reg/data/ana03/scratch/dubrovin/'
EXPNAME = 'amox23616'         if nargs <= 4 else sys.argv[4]
RUNNUM  = 104                 if nargs <= 5 else int(sys.argv[5])
DETNAME = 'xtcav'             if nargs <= 6 else sys.argv[6]
DETTYPE = 'camera'            if nargs <= 7 else sys.argv[7]
SERNUM  = '1234'              if nargs <= 8 else sys.argv[8]
NAMESID = 0                   if nargs <= 9 else int(sys.argv[9])

FNAME_HDF5 = os.path.join(DIRTMP,IFNAME)
FNAME_XTC2 = os.path.join(DIRTMP,OFNAME)

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
    #waveforms = h5f['waveforms']
    #times = h5f['times']
    raw = h5f['raw']
    #for nevt,(wfs,times) in enumerate(zip(waveforms,times)):
    for nevt,nda in enumerate(raw):
        my_data = {
            'array': nda
        }
            #'times': times
    
        if do_print(nevt) : print('Event %3d'%nevt, ' nda.shape:', nda.shape)
    
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

#----------

def test_xtc2_runinfo() :
    from psana.pyalgos.generic.NDArrUtils import print_ndarr

    from psana import DataSource
    ds = DataSource(files=FNAME_XTC2)
    orun = next(ds.runs())

    print('\ntest_xtc2_runinfo expt: %s  runnum: %d' % (orun.expt, orun.runnum))

    det = orun.Detector(DETNAME)
    #print('dir(det)', dir(det))

    for nev,evt in enumerate(orun.events()):
        if nev>10 : break
        print('Event %d'%nev, end='')
        print_ndarr(det.raw.array(evt), '  raw:')
        #print_ndarr(det.raw.raw(evt), '  raw:')

        #print('XXXXX', evt._dgrams[0].xtcav[0].raw.array)
        #print('XXXXXXX det.raw', str(det.raw))
        #print_ndarr(det.raw.times(evt),     '  times : ', last=4)
        #print_ndarr(det.raw.waveforms(evt), '  wforms: ', last=4)

#----------

if __name__ == "__main__":
    convert_hdf5_to_xtc2_with_runinfo()
    test_xtc2_runinfo()
    print(usage())
    print('DO NOT FORGET TO MOVE FILE TO /reg/g/psdm/detector/data2_test/xtc/')

#----------
