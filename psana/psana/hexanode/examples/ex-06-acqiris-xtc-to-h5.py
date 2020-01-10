#!/usr/bin/env python

"""Reads LCLS Acqiris data and saves waveform and time arrays in hdf5 file.
"""

# should be run on one of psana nodes
# ssh -Y pslogin
# ssh -Y psana
# which sees data like
# /reg/d/psdm/amo/amox27716/xtc/e1122-r0100-s00-c00.xtc or exp=amox27716:run=100
#
# source /reg/g/psdm/etc/psconda.sh
# event_keys -d exp=amox27716:run=100 -m3
# in LCLS1 environment after 

#----------

def usage() :
    return '\nUsage:'\
      + '\n  in LCLS1 environment after'\
      + '\n  source /reg/g/psdm/etc/psconda.sh'\
      + '\n  python lcls2/psana/psana/hexanode/examples/ex-06-acqiris-xtc-to-h5.py'\
      + '\n    or with positional arguments:'\
      + '\n  python lcls2/psana/psana/hexanode/examples/ex-06-acqiris-xtc-to-h5.py <EVENTS> <OFNAME> <DSNAME> <DETNAME>'\
      + '\n  lcls2/psana/psana/hexanode/examples/ex-06-acqiris-xtc-to-h5.py 100 /reg/data/ana03/scratch/dubrovin/acqiris_data.h5 exp=amox27716:run=100:smd ACQ1'\
      + '\n'

#----------

import sys
from psana import *

nargs = len(sys.argv)

EVENTS  = 10                          if nargs <= 1 else int(sys.argv[1])
OFNAME  = 'acqiris_data.h5'           if nargs <= 2 else sys.argv[2]
DSNAME  = 'exp=amox27716:run=100:smd' if nargs <= 3 else sys.argv[3] 
DETNAME = 'ACQ1'                      if nargs <= 4 else sys.argv[4] # AmoEndstation.0:Acqiris.1

print 'Input dataset: %s\nNumber of events: %d\nOutput file: %s\nDetector name: %s' % (DSNAME, EVENTS, OFNAME, DETNAME)

dsource = MPIDataSource(DSNAME)
acq = Detector(DETNAME)

smldata = dsource.small_data(OFNAME, gather_interval=100)

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

print usage()

#----------
