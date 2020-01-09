
# command to run
# on psana node
# cd .../lcls2
# . setup_env.sh
# python lcls2/psana/psana/hexanode/examples/ex-cpo-quad_to_xtc2.py
#----------

#DIRTMP =  '/reg/data/ana03/scratch/dubrovin/'
DIRTMP =  './'
FNAME_HDF5 = DIRTMP + 'hexanode.h5'
FNAME_XTC2 = DIRTMP + 'hexanode.xtc2'
DETNAME = 'tmo_hexanode'

def convert_hdf5_to_xtc2_with_runinfo() :

    import dgramCreate as dc
    import numpy as np
    import os
    import h5py
    
    dettype = 'hexanode'
    serial_number = '1234'
    namesid = 0
    
    nameinfo = dc.nameinfo(DETNAME,dettype,serial_number,namesid)
    alg = dc.alg('raw',[0,0,1])
    
    cydgram = dc.CyDgram()
    
    #---------- for runinfo
    
    runinfo_detname = 'runinfo'
    runinfo_dettype = 'runinfo'
    runinfo_detid = ''
    runinfo_namesid = 1
    runinfo_nameinfo = dc.nameinfo(runinfo_detname,runinfo_dettype,
                                   runinfo_detid,runinfo_namesid)
    runinfo_alg = dc.alg('runinfo',[0,0,1])
    runinfo_data = {
    'expt': 'amox27716',
    'runnum': 100
    }
    
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
        if nev>10 : break
        print('Event %d'%nev)
        print_ndarr(det.raw.times(evt),     '  times : ', last=4)
        print_ndarr(det.raw.waveforms(evt), '  wforms: ', last=4)

#----------

if __name__ == "__main__":
    convert_hdf5_to_xtc2_with_runinfo()
    test_xtc2_runinfo()

#----------
