#!/usr/bin/env python

"""Reds detector data from hdf5 file created by previous script and saves data and runinfo in xtc2 file.
"""
# command to run
# on psana node
# cd .../lcls2
# . setup_env.sh
# python lcls2/psana/psana/hexanode/examples/conv-12-h5-to-xtc2.py

#----------
import os
import sys
from psana.pyalgos.generic.Utils import do_print #, get_login, 
from psana.pyalgos.generic.NDArrUtils import print_ndarr

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

EVSKIP  = 10

FNAME_HDF5 = os.path.join(DIRTMP,IFNAME)
FNAME_XTC2 = os.path.join(DIRTMP,OFNAME)

def convert_hdf5_to_xtc2_with_runinfo() :

    import dgramCreate as dc # .../psana/peakFinder/dgramCreate.pyx (Lord, why it is here?)
    import numpy as np
    import os
    import h5py
    
    #xxxxxx_nameinfo = dc.nameinfo(DETNAME, DETTYPE, SERNUM, NAMESID)
    #---- raw
    raw_nameinfo = dc.nameinfo(DETNAME, DETTYPE, SERNUM, NAMESID)
    raw_alg = dc.alg('raw', [0,0,1])

    #---- runinfo
    runinfo_nameinfo = dc.nameinfo('runinfo', 'runinfo', '11', 1)
    runinfo_alg = dc.alg('runinfo', [0,0,1])

    #---- ebeam
    ebeam_nameinfo = dc.nameinfo('ebeam', 'ebeam', '22', 2)
    ebeam_alg = dc.alg('valsebm', [0,0,1])

    #---- eventid
    eventid_nameinfo = dc.nameinfo('eventid', 'eventid', '33', 3)
    eventid_alg = dc.alg('valseid', [0,0,1])

    #---- gasdetector
    gasdet_nameinfo = dc.nameinfo('gasdetector', 'gasdetector', '44', 4)
    gasdet_alg = dc.alg('valsgd', [0,0,1])

    #---- xtcav
    xtcav_nameinfo = dc.nameinfo('xtcav_pars', 'xtcav_pars', '55', 5)
    xtcav_alg = dc.alg('xtcav_pars', [0,0,1])

    #----------
    
    cydgram = dc.CyDgram()

    ifname = FNAME_HDF5
    ofname = FNAME_XTC2
    print('Input file: %s\nOutput file: %s' % (ifname,ofname))
    
    f = open(ofname,'wb')
    h5f = h5py.File(ifname, 'r')
    eventid = h5f['EventId']
    ebeam   = h5f['EBeam']
    gasdet  = h5f['GasDetector']
    xtcav   = h5f['XTCAV']
    raw     = h5f['raw']

    for nevt,nda in enumerate(raw):

        #if do_print(nevt) : print('Event %3d'%nevt, ' nda.shape:', nda.shape)

        print('Event %3d'%nevt, ' nda.shape:', nda.shape)

        if nevt<EVSKIP :
           print('  - skip event')
           continue

        print('  HDF5 exp        :', eventid['experiment'][nevt].decode())
        print('  HDF5 run        :', eventid['run'][nevt])
        print('  HDF5 time       :', eventid['time'][nevt])
        print('  HDF5 charge     :', ebeam['Charge'][nevt])
        print('  HDF5 dumpcharge :', ebeam['DumpCharge'][nevt])
        print('  HDF5 f_11_ENRC  :', gasdet['f_11_ENRC'][nevt])
        print('  HDF5 Beam_energy:', xtcav['XTCAV_Beam_energy_dump_GeV'][nevt])
        print_ndarr(nda, '  HDF5 raw:')

        #---------- for runinfo
        if nevt<2 : 
            runinfo_data = {'expt': EXPNAME, 'runnum': RUNNUM}
            cydgram.addDet(runinfo_nameinfo, runinfo_alg, runinfo_data)

        cydgram.addDet(ebeam_nameinfo, ebeam_alg, {\
          'Charge'    : ebeam['Charge'    ][nevt],\
          'DumpCharge': ebeam['DumpCharge'][nevt],\
          'XTCAVAmpl' : ebeam['XTCAVAmpl' ][nevt],\
          'XTCAVPhase': ebeam['XTCAVPhase'][nevt],\
          'PkCurrBC2' : ebeam['PkCurrBC2' ][nevt],\
          'L3Energy'  : ebeam['L3Energy'  ][nevt],\
        })

        cydgram.addDet(eventid_nameinfo, eventid_alg, {\
          'experiment': eventid['experiment'][nevt].decode(),\
          'run'       : eventid['run'       ][nevt],\
          'fiducials' : eventid['fiducials' ][nevt],\
          'time'      : eventid['time'      ][nevt],\
        })

        cydgram.addDet(gasdet_nameinfo, gasdet_alg, {\
          'f_11_ENRC': gasdet['f_11_ENRC'][nevt],\
          'f_12_ENRC': gasdet['f_12_ENRC'][nevt],\
          'f_21_ENRC': gasdet['f_21_ENRC'][nevt],\
          'f_22_ENRC': gasdet['f_22_ENRC'][nevt],\
          'f_63_ENRC': gasdet['f_63_ENRC'][nevt],\
          'f_64_ENRC': gasdet['f_64_ENRC'][nevt],\
        })

        cydgram.addDet(xtcav_nameinfo, xtcav_alg, {\
          'XTCAV_Analysis_Version'      : xtcav['XTCAV_Analysis_Version'      ][nevt],\
          'XTCAV_ROI_sizeX'             : xtcav['XTCAV_ROI_sizeX'             ][nevt],\
          'XTCAV_ROI_sizeY'             : xtcav['XTCAV_ROI_sizeY'             ][nevt],\
          'XTCAV_ROI_startX'            : xtcav['XTCAV_ROI_startX'            ][nevt],\
          'XTCAV_ROI_startY'            : xtcav['XTCAV_ROI_startY'            ][nevt],\
          'XTCAV_calib_umPerPx'         : xtcav['XTCAV_calib_umPerPx'         ][nevt],\
          'OTRS:DMP1:695:RESOLUTION'    : xtcav['OTRS:DMP1:695:RESOLUTION'    ][nevt],\
          'XTCAV_strength_par_S'        : xtcav['XTCAV_strength_par_S'        ][nevt],\
          'OTRS:DMP1:695:TCAL_X'        : xtcav['OTRS:DMP1:695:TCAL_X'        ][nevt],\
          'XTCAV_Amp_Des_calib_MV'      : xtcav['XTCAV_Amp_Des_calib_MV'      ][nevt],\
          'SIOC:SYS0:ML01:AO214'        : xtcav['SIOC:SYS0:ML01:AO214'        ][nevt],\
          'XTCAV_Phas_Des_calib_deg'    : xtcav['XTCAV_Phas_Des_calib_deg'    ][nevt],\
          'SIOC:SYS0:ML01:AO215'        : xtcav['SIOC:SYS0:ML01:AO215'        ][nevt],\
          'XTCAV_Beam_energy_dump_GeV'  : xtcav['XTCAV_Beam_energy_dump_GeV'  ][nevt],\
          'REFS:DMP1:400:EDES'          : xtcav['REFS:DMP1:400:EDES'          ][nevt],\
          'XTCAV_calib_disp_posToEnergy': xtcav['XTCAV_calib_disp_posToEnergy'][nevt],\
          'SIOC:SYS0:ML01:AO216'        : xtcav['SIOC:SYS0:ML01:AO216'        ][nevt],\
        })

        cydgram.addDet(raw_nameinfo, raw_alg, {'array': nda})

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

    os.system('detnames -r %s' % FNAME_XTC2)

    from psana import DataSource
    ds = DataSource(files=FNAME_XTC2)
    orun = next(ds.runs())

    print('\ntest_xtc2_runinfo expt: %s  runnum: %d' % (orun.expt, orun.runnum))

    det = orun.Detector(DETNAME)
    oeid= orun.Detector('eventid')
    oeb = orun.Detector('ebeam')
    ogd = orun.Detector('gasdetector')
    #print('dir(det)', dir(det))

    for nev,evt in enumerate(orun.events()):
        if nev>10 : break
        print('Event %d'%nev, end='')
        #print('=== dir(orun):\n', dir(orun))

        print('\n  XXXX - DIRECT TEST')
        dg0 = evt._dgrams[0]
        print('  XXXX dir(dg0):\n', dir(dg0))

        o1 = getattr(dg0, 'eventid', None)
        print('  XXXX valseid.run       :', o1[0].valseid.run        if o1 is not None else 'None')
        print('  XXXX valseid.experiment:', o1[0].valseid.experiment if o1 is not None else 'None')
        print('  XXXX valseid.fiducials :', o1[0].valseid.fiducials  if o1 is not None else 'None')
        print('  XXXX valseid.time      :', o1[0].valseid.time       if o1 is not None else 'None')

        o2 = getattr(dg0, 'ebeam', None)
        print('  XXXX valsebm.Charge    :', o2[0].valsebm.Charge     if o2 is not None else 'None')
        print('  XXXX valsebm.DumpCharge:', o2[0].valsebm.DumpCharge if o2 is not None else 'None')

        o3 = getattr(dg0, 'gasdetector', None)
        print('  XXXX valsgd.f_11_ENRC  :', o3[0].valsgd.f_11_ENRC   if o3 is not None else 'None')

        o4 = getattr(dg0, 'xtcav', None)
        print_ndarr(o4[0].raw.array, '  XXXX raw:')


        print('\n  YYYY - DETECTOR TEST')
        #print('  dir(oeb):', dir(oeb))
        print('  YYYY Charge    :', oeb.valsebm.Charge(evt))
        print('  YYYY DumpCharge:', oeb.valsebm.DumpCharge(evt))
        print('  YYYY fiducials :', oeid.valseid.fiducials(evt))
        print('  YYYY time      :', oeid.valseid.time(evt))
        print('  YYYY f_11_ENRC :', ogd.valsgd.f_11_ENRC(evt))
        print_ndarr(det.raw.array(evt), '  YYYY raw:')

#----------

if __name__ == "__main__":
    convert_hdf5_to_xtc2_with_runinfo()
    test_xtc2_runinfo()
    print(usage())
    print('DO NOT FORGET TO MOVE FILE TO /reg/g/psdm/detector/data2_test/xtc/')

#----------
