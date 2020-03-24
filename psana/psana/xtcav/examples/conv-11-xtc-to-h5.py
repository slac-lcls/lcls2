#!/usr/bin/env python

"""Reads LCLS data and saves smd in hdf5 file.
"""

# should be run on one of psana nodes
# ssh -Y pslogin
# ssh -Y psana
# which sees data like
# /reg/d/psdm/amo/amox23616/xtc/e1114-r0104-s00-c00.xtc or exp=amox23616:run=104
#
# source /reg/g/psdm/etc/psconda.sh
# event_keys -d exp=amox23616:run=104 -m3
# in LCLS1 environment after 

#----------

import sys
import h5py
import _psana
from psana import *
from pyimgalgos.GlobalUtils import print_ndarr

def usage() :
    scrname = sys.argv[0]
    return '\nUsage:'\
      + '\n  in LCLS1 environment after'\
      + '\n    source /reg/g/psdm/etc/psconda.sh'\
      + '\n  run script with positional arguments:'\
      + '\n    %s <EVENTS> <OFNAME> <DSNAME> <DETNAME>' % scrname\
      + '\n    %s 400 /reg/data/ana03/scratch/dubrovin/amox23616-r0104-e400-xtcav.h5 exp=amox23616:run=104 xtcav' % scrname\
      + '\n    %s 200 /reg/data/ana03/scratch/dubrovin/amox23616-r0131-e200-xtcav.h5 exp=amox23616:run=131 xtcav' % scrname\
      + '\n    %s 100 /reg/data/ana03/scratch/dubrovin/amox23616-r0137-e100-xtcav.h5 exp=amox23616:run=137 xtcav' % scrname\
      + '\n'

#--------------------

def do_print(nev) :
    return nev<10\
       or (nev<50 and (not nev%10))\
       or (nev<500 and (not nev%100))\
       or not nev%1000

def print_psobj_methods(o, ptrn='ebeam') :
   dir_o = dir(o)
   print ptrn, dir_o
   for k in dir_o : 
       if ptrn in k :
           v = getattr(o, k, 'N/A')()
           print '  %24s :' % k, v, type(v)

def print_group_xtcav_values(evt) :
   print 'XTCAV_Analysis_Version      :', Detector('XTCAV_Analysis_Version')(evt)
   print 'XTCAV_ROI_sizeX             :', Detector('XTCAV_ROI_sizeX')(evt)
   print 'XTCAV_ROI_sizeY             :', Detector('XTCAV_ROI_sizeY')(evt)
   print 'XTCAV_ROI_startX            :', Detector('XTCAV_ROI_startX')(evt)
   print 'XTCAV_ROI_startY            :', Detector('XTCAV_ROI_startY')(evt)
   print 'XTCAV_calib_umPerPx         :', Detector('XTCAV_calib_umPerPx')(evt)
   print 'OTRS:DMP1:695:RESOLUTION    :', Detector('OTRS:DMP1:695:RESOLUTION')(evt)
   print 'XTCAV_strength_par_S        :', Detector('XTCAV_strength_par_S')(evt)
   print 'OTRS:DMP1:695:TCAL_X        :', Detector('OTRS:DMP1:695:TCAL_X')(evt)
   print 'XTCAV_Amp_Des_calib_MV      :', Detector('XTCAV_Amp_Des_calib_MV')(evt)
   print 'SIOC:SYS0:ML01:AO214        :', Detector('SIOC:SYS0:ML01:AO214')(evt)
   print 'XTCAV_Phas_Des_calib_deg    :', Detector('XTCAV_Phas_Des_calib_deg')(evt)
   print 'SIOC:SYS0:ML01:AO215        :', Detector('SIOC:SYS0:ML01:AO215')(evt)
   print 'XTCAV_Beam_energy_dump_GeV  :', Detector('XTCAV_Beam_energy_dump_GeV')(evt)
   print 'REFS:DMP1:400:EDES          :', Detector('REFS:DMP1:400:EDES')(evt)
   print 'XTCAV_calib_disp_posToEnergy:', Detector('XTCAV_calib_disp_posToEnergy')(evt)
   print 'SIOC:SYS0:ML01:AO216        :', Detector('SIOC:SYS0:ML01:AO216')(evt)
   #print ':', Detector('')(evt)

#----------

nargs = len(sys.argv)

EVENTS  = 20                          if nargs <= 1 else int(sys.argv[1])
OFNAME  = 'tmp-data.h5'               if nargs <= 2 else sys.argv[2]
DSNAME  = 'exp=amox23616:run=131:smd' if nargs <= 3 else sys.argv[3] 
DETNAME = 'xtcav'                     if nargs <= 4 else sys.argv[4]
EXPNAME = 'amox23616'                 if nargs <= 5 else sys.argv[5]

print 'Input dataset: %s\nNumber of events: %d\nOutput file: %s\nDetector name: %s' % (DSNAME, EVENTS, OFNAME, DETNAME)

###ds = MPIDataSource(DSNAME)
ds = DataSource(DSNAME)
det = Detector(DETNAME)
evrdet0 = Detector('NoDetector.0:Evr.0')
evrdet1 = Detector('NoDetector.0:Evr.1')
#gasdet = Detector('FEEGasDetEnergy')

###smldata = ds.small_data(OFNAME, gather_interval=100)

#chsize = 2048
chsize = 100
h5out = h5py.File(OFNAME, 'w')

#'damageMask', 'ebeamCharge', 'ebeamDumpCharge', 'ebeamEnergyBC1', 'ebeamEnergyBC2', 
#'ebeamL3Energy', 'ebeamLTU250', 'ebeamLTU450', 'ebeamLTUAngX', 'ebeamLTUAngY', 
#'ebeamLTUPosX', 'ebeamLTUPosY', 'ebeamPhotonEnergy', 'ebeamPkCurrBC1', 'ebeamPkCurrBC2', 
#'ebeamUndAngX', 'ebeamUndAngY', 'ebeamUndPosX', 'ebeamUndPosY', 'ebeamXTCAVAmpl', 'ebeamXTCAVPhase'
ebeam_gr = h5out.create_group('EBeam')
ebeam_Charge     = ebeam_gr.create_dataset('Charge',     (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
ebeam_DumpCharge = ebeam_gr.create_dataset('DumpCharge', (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
ebeam_XTCAVAmpl  = ebeam_gr.create_dataset('XTCAVAmpl',  (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
ebeam_XTCAVPhase = ebeam_gr.create_dataset('XTCAVPhase', (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
ebeam_PkCurrBC2  = ebeam_gr.create_dataset('PkCurrBC2',  (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
ebeam_L3Energy   = ebeam_gr.create_dataset('L3Energy',   (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))

#'fiducials', 'idxtime', 'run', 'ticks', 'time', 'vector'
eventid_gr = h5out.create_group('EventId')
eventid_exp       = eventid_gr.create_dataset('experiment',(EVENTS+1,),  dtype='S10')
eventid_run       = eventid_gr.create_dataset('run',       (EVENTS+1,),  dtype='i4', chunks=(chsize,), maxshape=(None,))
eventid_fiducials = eventid_gr.create_dataset('fiducials', (EVENTS+1,),  dtype='i4', chunks=(chsize,), maxshape=(None,))
eventid_time      = eventid_gr.create_dataset('time',      (EVENTS+1,2), dtype='i8', chunks=(chsize,2), maxshape=(None,2))

# 'f_11_ENRC', 'f_12_ENRC', 'f_21_ENRC', 'f_22_ENRC', 'f_63_ENRC', 'f_64_ENRC'
gasdet_gr = h5out.create_group('GasDetector')
gasdet_f_11_ENRC = gasdet_gr.create_dataset('f_11_ENRC', (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
gasdet_f_12_ENRC = gasdet_gr.create_dataset('f_12_ENRC', (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
gasdet_f_21_ENRC = gasdet_gr.create_dataset('f_21_ENRC', (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
gasdet_f_22_ENRC = gasdet_gr.create_dataset('f_22_ENRC', (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
gasdet_f_63_ENRC = gasdet_gr.create_dataset('f_63_ENRC', (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
gasdet_f_64_ENRC = gasdet_gr.create_dataset('f_64_ENRC', (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))

xtcav_gr = h5out.create_group('XTCAV')
xtcav_XTCAV_Analysis_Version       = xtcav_gr.create_dataset('XTCAV_Analysis_Version',       (EVENTS+1,), dtype='i4', chunks=(chsize,), maxshape=(None,))
xtcav_XTCAV_ROI_sizeX              = xtcav_gr.create_dataset('XTCAV_ROI_sizeX',              (EVENTS+1,), dtype='i4', chunks=(chsize,), maxshape=(None,))
xtcav_XTCAV_ROI_sizeY              = xtcav_gr.create_dataset('XTCAV_ROI_sizeY',              (EVENTS+1,), dtype='i4', chunks=(chsize,), maxshape=(None,))
xtcav_XTCAV_ROI_startX             = xtcav_gr.create_dataset('XTCAV_ROI_startX',             (EVENTS+1,), dtype='i4', chunks=(chsize,), maxshape=(None,))
xtcav_XTCAV_ROI_startY             = xtcav_gr.create_dataset('XTCAV_ROI_startY',             (EVENTS+1,), dtype='i4', chunks=(chsize,), maxshape=(None,))
xtcav_XTCAV_calib_umPerPx          = xtcav_gr.create_dataset('XTCAV_calib_umPerPx',          (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_OTRS_DMP1_695_RESOLUTION     = xtcav_gr.create_dataset('OTRS:DMP1:695:RESOLUTION',     (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_XTCAV_strength_par_S         = xtcav_gr.create_dataset('XTCAV_strength_par_S',         (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_OTRS_DMP1_695_TCAL_X         = xtcav_gr.create_dataset('OTRS:DMP1:695:TCAL_X',         (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_XTCAV_Amp_Des_calib_MV       = xtcav_gr.create_dataset('XTCAV_Amp_Des_calib_MV',       (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_SIOC_SYS0_ML01_AO214         = xtcav_gr.create_dataset('SIOC:SYS0:ML01:AO214',         (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_XTCAV_Phas_Des_calib_deg     = xtcav_gr.create_dataset('XTCAV_Phas_Des_calib_deg',     (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_SIOC_SYS0_ML01_AO215         = xtcav_gr.create_dataset('SIOC:SYS0:ML01:AO215',         (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_XTCAV_Beam_energy_dump_GeV   = xtcav_gr.create_dataset('XTCAV_Beam_energy_dump_GeV',   (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_REFS_DMP1_400_EDES           = xtcav_gr.create_dataset('REFS:DMP1:400:EDES',           (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_XTCAV_calib_disp_posToEnergy = xtcav_gr.create_dataset('XTCAV_calib_disp_posToEnergy', (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))
xtcav_SIOC_SYS0_ML01_AO216         = xtcav_gr.create_dataset('SIOC:SYS0:ML01:AO216',         (EVENTS+1,), dtype='f8', chunks=(chsize,), maxshape=(None,))

raw_ds = h5out.create_dataset('raw', (EVENTS+1,1024,1024), dtype='u2', chunks=(chsize,1024,1024), maxshape=(None,1024,1024))
  
for nevt,evt in enumerate(ds.events()):
   #wfs = det.waveform(evt)
   #times = det.wftime(evt)
   raw = det.raw(evt)

   eventId = evt.get(EventId)
   ebeam   = evt.get(Bld.BldDataEBeamV7, Source('EBeam'))
   gasdet  = evt.get(Bld.BldDataFEEGasDetEnergyV1, Source('FEEGasDetEnergy'))
   #epics   = evt.get(Epics.ConfigV1, Source('EpicsArch.0:NoDevice.0'))
   #evr0    = evt.get(_psana.EvrData.DataV4, Source('NoDetector.0:Evr.0'))
   #evr1    = evt.get(EvrData.DataV4, Source('NoDetector.0:Evr.1'))

   if do_print(nevt) : print 'ev: %3d'%nevt, ' det.raw().shape:', raw.shape, ' ebeam is None: %s' % (ebeam is None)
   if raw is None:
      print '  ev: %3d'%nevt, ' raw: None'
      continue

   if ebeam is None : continue

   #print_psobj_methods(ebeam, 'ebeam')
   #print_psobj_methods(eventId, 'eventId')
   #print_psobj_methods(gasdet, 'gasdet')
   #print_psobj_methods(epics, 'epics')
   #print_psobj_methods(evr0, 'evr0')
   #print_psobj_methods(evr1, 'evr1')
   #print '  time:', getattr(eventId, 'time', None)()

   #print 'fifoEvents   :', getattr(evr0, 'fifoEvents', None)()
   #print 'numFifoEvents:', getattr(evr0, 'numFifoEvents', None)()
   #print 'present      :', getattr(evr0, 'present', None)(0)

   #print 'evrdet0:', evrdet0(evt)
   #print 'evrdet1:', evrdet1(evt)
   #print_group_xtcav_values(evt)

   #ebeam_Charge    .resize((nevt+1,))
   #ebeam_DumpCharge.resize((nevt+1,))

   ebeam_Charge[nevt]     = getattr(ebeam, 'ebeamCharge', None)()
   ebeam_DumpCharge[nevt] = getattr(ebeam, 'ebeamDumpCharge', None)()
   ebeam_XTCAVAmpl[nevt]  = getattr(ebeam, 'ebeamXTCAVAmpl', None)()
   ebeam_XTCAVPhase[nevt] = getattr(ebeam, 'ebeamXTCAVPhase', None)()
   ebeam_PkCurrBC2[nevt]  = getattr(ebeam, 'ebeamPkCurrBC2', None)()
   ebeam_L3Energy[nevt]   = getattr(ebeam, 'ebeamL3Energy', None)()

   eventid_exp[nevt]      = EXPNAME
   eventid_run[nevt]      = getattr(eventId, 'run', None)()
   eventid_time[nevt]     = getattr(eventId, 'time', None)()
   eventid_fiducials[nevt]= getattr(eventId, 'fiducials', None)()

   gasdet_f_11_ENRC[nevt] = getattr(gasdet, 'f_11_ENRC', None)()
   gasdet_f_12_ENRC[nevt] = getattr(gasdet, 'f_12_ENRC', None)()
   gasdet_f_21_ENRC[nevt] = getattr(gasdet, 'f_21_ENRC', None)()
   gasdet_f_22_ENRC[nevt] = getattr(gasdet, 'f_22_ENRC', None)()
   gasdet_f_63_ENRC[nevt] = getattr(gasdet, 'f_63_ENRC', None)()
   gasdet_f_64_ENRC[nevt] = getattr(gasdet, 'f_64_ENRC', None)()

   xtcav_XTCAV_Analysis_Version[nevt]       = Detector('XTCAV_Analysis_Version')(evt)
   xtcav_XTCAV_ROI_sizeX[nevt]              = Detector('XTCAV_ROI_sizeX')(evt)
   xtcav_XTCAV_ROI_sizeY[nevt]              = Detector('XTCAV_ROI_sizeY')(evt)
   xtcav_XTCAV_ROI_startX[nevt]             = Detector('XTCAV_ROI_startX')(evt)
   xtcav_XTCAV_ROI_startY[nevt]             = Detector('XTCAV_ROI_startY')(evt)
   xtcav_XTCAV_calib_umPerPx[nevt]          = Detector('XTCAV_calib_umPerPx')(evt)
   xtcav_OTRS_DMP1_695_RESOLUTION[nevt]     = Detector('OTRS:DMP1:695:RESOLUTION')(evt)
   xtcav_XTCAV_strength_par_S[nevt]         = Detector('XTCAV_strength_par_S')(evt)
   xtcav_OTRS_DMP1_695_TCAL_X[nevt]         = Detector('OTRS:DMP1:695:TCAL_X')(evt)
   xtcav_XTCAV_Amp_Des_calib_MV[nevt]       = Detector('XTCAV_Amp_Des_calib_MV')(evt)
   xtcav_SIOC_SYS0_ML01_AO214[nevt]         = Detector('SIOC:SYS0:ML01:AO214')(evt)
   xtcav_XTCAV_Phas_Des_calib_deg[nevt]     = Detector('XTCAV_Phas_Des_calib_deg')(evt)
   xtcav_SIOC_SYS0_ML01_AO215[nevt]         = Detector('XTCAV_Beam_energy_dump_GeV')(evt)
   xtcav_XTCAV_Beam_energy_dump_GeV[nevt]   = Detector('REFS:DMP1:400:EDES')(evt)
   xtcav_REFS_DMP1_400_EDES[nevt]           = Detector('REFS:DMP1:400:EDES')(evt)
   xtcav_XTCAV_calib_disp_posToEnergy[nevt] = Detector('XTCAV_calib_disp_posToEnergy')(evt)
   xtcav_SIOC_SYS0_ML01_AO216[nevt]         = Detector('SIOC:SYS0:ML01:AO216')(evt)

   raw_ds[nevt] = raw

   ###smldata.event(raw=raw) # waveforms=wfs,times=times)

   if not(nevt<EVENTS): break

print 'End of event loop, ev: %3d'%nevt
print_ndarr(raw, 'raw')

###smldata.save()
h5out.close()
print usage()
print 'saved file %s' % OFNAME

#----------
