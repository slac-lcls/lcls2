
# should be run on one of psana nodes
# ssh -Y pslogin
# ssh -Y psana
# which sees data like
# /reg/d/psdm/amo/amox27716/xtc/e1122-r0100-s00-c00.xtc
# in LCLS1 environment after 
# source /reg/g/psdm/etc/psconda.sh
# command to run:
# python lcls2/psana/psana/hexanode/examples/ex-cpo-quad_save_lcls1.py

from psana import *

EVENTS = 100

dsname = 'exp=amox27716:run=100:smd'
ofname = 'hexanode.h5'
#ofname = '/reg/data/ana03/scratch/dubrovin/hexanode.h5'
print 'Input dataset: %s\nOutput file: %s' % (dsname, ofname)

dsource = MPIDataSource(dsname)
acq = Detector('ACQ1')

smldata = dsource.small_data(ofname,gather_interval=100)

for nevt,evt in enumerate(dsource.events()):
   wfs = acq.waveform(evt)
   times = acq.wftime(evt)
   print 'ev: %3d'%nevt,
   if wfs is None:
      print 'wfs: None'
      continue
   print 'shapes of wfs:', wfs.shape, ' times:', times.shape
   smldata.event(waveforms=wfs,times=times)

   if nevt>EVENTS: break

smldata.save()
