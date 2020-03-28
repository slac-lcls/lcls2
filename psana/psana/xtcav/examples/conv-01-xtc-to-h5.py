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
from psana import *

def usage() :
    scrname = sys.argv[0]
    return '\nUsage:'\
      + '\n  in LCLS1 environment after'\
      + '\n    source /reg/g/psdm/etc/psconda.sh'\
      + '\n  run script with positional arguments:'\
      + '\n    %s <EVENTS> <OFNAME> <DSNAME> <DETNAME>' % scrname\
      + '\n    %s 400 /reg/data/ana03/scratch/dubrovin/amox23616-r0104-e400-xtcav.h5 exp=amox23616:run=104:smd xtcav' % scrname\
      + '\n    %s 200 /reg/data/ana03/scratch/dubrovin/amox23616-r0131-e200-xtcav.h5 exp=amox23616:run=131:smd xtcav' % scrname\
      + '\n    %s 100 /reg/data/ana03/scratch/dubrovin/amox23616-r0137-e100-xtcav.h5 exp=amox23616:run=137:smd xtcav' % scrname\
      + '\n'

#--------------------

def do_print(nev) :
    return nev<10\
       or (nev<50 and (not nev%10))\
       or (nev<500 and (not nev%100))\
       or not nev%1000

#----------

nargs = len(sys.argv)

EVENTS  = 10                          if nargs <= 1 else int(sys.argv[1])
OFNAME  = 'tmp-data.h5'               if nargs <= 2 else sys.argv[2]
DSNAME  = 'exp=amox23616:run=104:smd' if nargs <= 3 else sys.argv[3] 
DETNAME = 'xtcav'                     if nargs <= 4 else sys.argv[4] # XrayTransportDiagnostic.0:Opal1000.0

print 'Input dataset: %s\nNumber of events: %d\nOutput file: %s\nDetector name: %s' % (DSNAME, EVENTS, OFNAME, DETNAME)

dsource = MPIDataSource(DSNAME)
det = Detector(DETNAME)

smldata = dsource.small_data(OFNAME, gather_interval=100)

for nevt,evt in enumerate(dsource.events()):
   #wfs = det.waveform(evt)
   #times = det.wftime(evt)
   raw = det.raw(evt)
   if do_print(nevt) : print 'ev: %3d'%nevt, ' det.raw().shape:', raw.shape
   if raw is None:
      print '  ev: %3d'%nevt, ' raw: None'
      continue
   smldata.event(raw=raw) # waveforms=wfs,times=times)
   if not(nevt<EVENTS): break

print 'End of event loop, ev: %3d'%nevt, ' det.raw().shape:', raw.shape

smldata.save()

print usage()

#----------
