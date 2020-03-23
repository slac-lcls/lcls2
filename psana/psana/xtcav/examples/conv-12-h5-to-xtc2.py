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



def time_stamp(t_sec_nsec) :
    t_sec, t_nsec = t_sec_nsec
    return (t_sec << 32) | t_nsec

#----------
#exp=amox23616:run=104

nargs = len(sys.argv)

IFNAME  = 'tmp-data.h5'       if nargs <= 1 else sys.argv[1]
OFNAME  = 'tmp-data.xtc2'     if nargs <= 2 else sys.argv[2]
DIRTMP  = './'                if nargs <= 3 else sys.argv[3] # '/reg/data/ana03/scratch/dubrovin/'
EXPNAME = 'amox23616'         if nargs <= 4 else sys.argv[4]
RUNNUM  = 131                 if nargs <= 5 else int(sys.argv[5])
DETNAME = 'xtcav'             if nargs <= 6 else sys.argv[6]
DETTYPE = 'camera'            if nargs <= 7 else sys.argv[7]
SERNUM  = '1234'              if nargs <= 8 else sys.argv[8]
NAMESID = 0                   if nargs <= 9 else int(sys.argv[9])

EVSKIP  = 0
EVENTS  = 10 + EVSKIP

FNAME_HDF5 = os.path.join(DIRTMP,IFNAME)
FNAME_XTC2 = os.path.join(DIRTMP,OFNAME)

RAW       = 101
RUNINFO   = 102
EBEAM     = 103
EVENTID   = 104
GASDET    = 105
XTCAVPARS = 106

LIST_SAVE = (RAW, RUNINFO, EBEAM, EVENTID, GASDET, XTCAVPARS)
#LIST_SAVE = (EBEAM, EVENTID, GASDET, XTCAVPARS)
#LIST_SAVE = (RAW, GASDET)

def convert_hdf5_to_xtc2_with_runinfo() :

    import dgramCreate as dc # .../psana/peakFinder/dgramCreate.pyx (Lord, why it is here?)
    import numpy as np
    import os
    import h5py
    
    if RAW in LIST_SAVE :
        #xxxxxx_nameinfo = dc.nameinfo(DETNAME, DETTYPE, SERNUM, NAMESID)
        #---- raw
        #raw_nameinfo = dc.nameinfo(DETNAME, DETTYPE, SERNUM, NAMESID)
        raw_nameinfo = dc.nameinfo('xtcav', 'camera', '1234', 0)
        raw_alg = dc.alg('raw', [0,0,1])
    
    if RUNINFO in LIST_SAVE :
        #---- runinfo
        runinfo_nameinfo = dc.nameinfo('runinfo', 'runinfo', '11', 1)
        runinfo_alg = dc.alg('runinfo', [0,0,1])
    
    if EBEAM in LIST_SAVE :
        #---- ebeam
        ebeam_nameinfo = dc.nameinfo('ebeam', 'ebeam', '22', 2)
        ebeam_alg = dc.alg('valsebm', [0,0,1])
    
    if EVENTID in LIST_SAVE :
        #---- eventid
        eventid_nameinfo = dc.nameinfo('eventid', 'eventid', '33', 3)
        eventid_alg = dc.alg('valseid', [0,0,1])
    
    if GASDET in LIST_SAVE :
        #---- gasdetector
        gasdet_nameinfo = dc.nameinfo('gasdetector', 'gasdetector', '44', 4)
        gasdet_alg = dc.alg('valsgd', [0,0,1])
    
    if XTCAVPARS in LIST_SAVE :
        #---- xtcav
        xtcav_nameinfo = dc.nameinfo('xtcavpars', 'xtcavpars', '55', 5)
        xtcav_alg = dc.alg('valsxtp', [0,0,1])

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

        #if nevt<EVSKIP :
        #   print('  - skip event')
        #   continue

        print('       -> RUNINFO    EXPNAME    :', EXPNAME)
        print('       -> RUNINFO    RUNNUM     :', RUNNUM)
        print('  HDF5 -> EVENTID    exp        :', eventid['experiment'][nevt].decode())
        print('  HDF5 -> EVENTID    run        :', eventid['run'       ][nevt])
        print('  HDF5 -> EVENTID    time       :', eventid['time'      ][nevt])
        print('  HDF5 -> EBEAM      charge     :', ebeam['Charge'      ][nevt])
        print('  HDF5 -> EBEAM      dumpcharge :', ebeam['DumpCharge'  ][nevt])
        print('  HDF5 -> GASDET     f_11_ENRC  :', gasdet['f_11_ENRC'  ][nevt])
        print('  HDF5 -> XTCAVPARS  Beam_energy:', xtcav['XTCAV_Beam_energy_dump_GeV'][nevt])
        print_ndarr(nda, '  HDF5 RAW raw:')

        if RUNINFO in LIST_SAVE :
            #---------- for runinfo
            if nevt<2 : 
                runinfo_data = {'expt': EXPNAME, 'runnum': RUNNUM}
                cydgram.addDet(runinfo_nameinfo, runinfo_alg, runinfo_data)
    
        if EBEAM in LIST_SAVE :
          #if ebeam['Charge'][nevt] > 0 :
            cydgram.addDet(ebeam_nameinfo, ebeam_alg, {\
              'Charge'    : ebeam['Charge'    ][nevt],\
              'DumpCharge': ebeam['DumpCharge'][nevt],\
              'XTCAVAmpl' : ebeam['XTCAVAmpl' ][nevt],\
              'XTCAVPhase': ebeam['XTCAVPhase'][nevt],\
              'PkCurrBC2' : ebeam['PkCurrBC2' ][nevt],\
              'L3Energy'  : ebeam['L3Energy'  ][nevt],\
            })
    
        if EVENTID in LIST_SAVE :
            cydgram.addDet(eventid_nameinfo, eventid_alg, {\
              'experiment': eventid['experiment'][nevt].decode(),\
              'run'       : eventid['run'       ][nevt],\
              'fiducials' : eventid['fiducials' ][nevt],\
              'time'      : eventid['time'      ][nevt],\
            })
    
        if GASDET in LIST_SAVE :
          #if gasdet['f_11_ENRC'][nevt] > 0 :
            cydgram.addDet(gasdet_nameinfo, gasdet_alg, {\
              'f_11_ENRC': gasdet['f_11_ENRC'][nevt],\
              'f_12_ENRC': gasdet['f_12_ENRC'][nevt],\
              'f_21_ENRC': gasdet['f_21_ENRC'][nevt],\
              'f_22_ENRC': gasdet['f_22_ENRC'][nevt],\
              'f_63_ENRC': gasdet['f_63_ENRC'][nevt],\
              'f_64_ENRC': gasdet['f_64_ENRC'][nevt],\
            })
    
        if XTCAVPARS in LIST_SAVE :
            cydgram.addDet(xtcav_nameinfo, xtcav_alg, {\
              'XTCAV_Analysis_Version'      : xtcav['XTCAV_Analysis_Version'      ][nevt],\
              'XTCAV_ROI_sizeX'             : xtcav['XTCAV_ROI_sizeX'             ][nevt],\
              'XTCAV_ROI_sizeY'             : xtcav['XTCAV_ROI_sizeY'             ][nevt],\
              'XTCAV_ROI_startX'            : xtcav['XTCAV_ROI_startX'            ][nevt],\
              'XTCAV_ROI_startY'            : xtcav['XTCAV_ROI_startY'            ][nevt],\
              'XTCAV_calib_umPerPx'         : xtcav['XTCAV_calib_umPerPx'         ][nevt],\
              'OTRS_DMP1_695_RESOLUTION'    : xtcav['OTRS:DMP1:695:RESOLUTION'    ][nevt],\
              'XTCAV_strength_par_S'        : xtcav['XTCAV_strength_par_S'        ][nevt],\
              'OTRS_DMP1_695_TCAL_X'        : xtcav['OTRS:DMP1:695:TCAL_X'        ][nevt],\
              'XTCAV_Amp_Des_calib_MV'      : xtcav['XTCAV_Amp_Des_calib_MV'      ][nevt],\
              'SIOC_SYS0_ML01_AO214'        : xtcav['SIOC:SYS0:ML01:AO214'        ][nevt],\
              'XTCAV_Phas_Des_calib_deg'    : xtcav['XTCAV_Phas_Des_calib_deg'    ][nevt],\
              'SIOC_SYS0_ML01_AO215'        : xtcav['SIOC:SYS0:ML01:AO215'        ][nevt],\
              'XTCAV_Beam_energy_dump_GeV'  : xtcav['XTCAV_Beam_energy_dump_GeV'  ][nevt],\
              'REFS_DMP1_400_EDES'          : xtcav['REFS:DMP1:400:EDES'          ][nevt],\
              'XTCAV_calib_disp_posToEnergy': xtcav['XTCAV_calib_disp_posToEnergy'][nevt],\
              'SIOC_SYS0_ML01_AO216'        : xtcav['SIOC:SYS0:ML01:AO216'        ][nevt],\
            })

        if RAW in LIST_SAVE :
            cydgram.addDet(raw_nameinfo, raw_alg, {'array': nda})

        t_sec_nsec = eventid['time'][nevt]
        t_sec, t_nsec = t_sec_nsec
        #timestamp = nevt
        #timestamp = t_sec_nsec #time_stamp(t_sec_nsec)
        timestamp = t_sec + nevt #time_stamp(t_sec_nsec)

        transitionid = None
        if   (nevt==0): transitionid = 2  # Configure
        elif (nevt==1): transitionid = 4  # BeginRun
        else:           transitionid = 12 # L1Accept
        xtc_bytes = cydgram.get(timestamp, transitionid)
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
    oxp = orun.Detector('xtcavpars')
    #print('dir(det)', dir(det))

    for nev,evt in enumerate(orun.events()):
        if nev>EVENTS : break
        print('Event %d'%nev, end='')
        #print('=== dir(orun):\n', dir(orun))

        print('\n  XXXX - DIRECT TEST')
        dg0 = evt._dgrams[0]
        print('  XXXX dir(dg0):\n', dir(dg0))

        if EVENTID in LIST_SAVE :
          o1 = getattr(dg0, 'eventid', None)
          print('  XXXX valseid.run       :', o1[0].valseid.run        if o1 is not None else 'None')
          print('  XXXX valseid.experiment:', o1[0].valseid.experiment if o1 is not None else 'None')
          print('  XXXX valseid.fiducials :', o1[0].valseid.fiducials  if o1 is not None else 'None')
          print('  XXXX valseid.time      :', o1[0].valseid.time       if o1 is not None else 'None')

        if EBEAM in LIST_SAVE :
          o2 = getattr(dg0, 'ebeam', None)
          print('  XXXX valsebm.Charge    :', o2[0].valsebm.Charge     if o2 is not None else 'None')
          print('  XXXX valsebm.DumpCharge:', o2[0].valsebm.DumpCharge if o2 is not None else 'None')

        if GASDET in LIST_SAVE :
          o3 = getattr(dg0, 'gasdetector', None)
          print('  XXXX valsgd.f_11_ENRC  :', o3[0].valsgd.f_11_ENRC   if o3 is not None else 'None')

        if XTCAVPARS in LIST_SAVE :
          o4 = getattr(dg0, 'xtcavpars', None)
          print('  XXXX valsgd.XTCAV_Beam_energy_dump_GeV  :', o4[0].valsxtp.XTCAV_Beam_energy_dump_GeV if o4 is not None else 'None')

        if RAW in LIST_SAVE :
          o5 = getattr(dg0, 'xtcav', None)
          print_ndarr(o5[0].raw.array, '  XXXX raw:')

        print('\n  YYYY - DETECTOR TEST')
        #print('  dir(oeb):', dir(oeb))

        if EBEAM in LIST_SAVE :
          print('  YYYY Charge    :', oeb.valsebm.Charge(evt))
          print('  YYYY DumpCharge:', oeb.valsebm.DumpCharge(evt))

        if EVENTID in LIST_SAVE :
          print('  YYYY fiducials :', oeid.valseid.fiducials(evt))
          print('  YYYY time      :', oeid.valseid.time(evt))

        if GASDET in LIST_SAVE :
          o = getattr(ogd, 'valsgd', None)
          print('  YYYY f_11_ENRC :', ogd.valsgd.f_11_ENRC(evt) if o is not None else 'None')

        if XTCAVPARS in LIST_SAVE :
          print('  YYYY XTCAV_Beam_energy_dump_GeV :', oxp.valsxtp.XTCAV_Beam_energy_dump_GeV(evt))

        if RAW in LIST_SAVE :
          print_ndarr(det.raw.array(evt), '  YYYY raw:')

#----------

if __name__ == "__main__":
    convert_hdf5_to_xtc2_with_runinfo()
    test_xtc2_runinfo()
    print(usage())
    print('DO NOT FORGET TO MOVE FILE TO /reg/g/psdm/detector/data2_test/xtc/')

#----------
