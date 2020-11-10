#!/usr/bin/env python
"""Test access to detector raw data.
"""

import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]

usage = 'Usage > ./%s <test-name [1,2,3,...]> <file-name.xtc2> <detector-name> <max-number-of-evants>' % SCRNAME
print(usage)

fname0 = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005.xtc2'
#fname1 = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005-seg1and3.xtc2'

tname   = sys.argv[1] if len(sys.argv)>1 else '1'
dsname  = sys.argv[2] if len(sys.argv)>2 else fname0
detname = sys.argv[3] if len(sys.argv)>3 else 'epix10k2M'
evtmax  = int(sys.argv[4]) if len(sys.argv)>4 else 10

if tname == '1':

    from psana import DataSource
    ds = DataSource(files=dsname)

    #orun = next(ds.runs())
    for orun in ds.runs():
      det = orun.Detector(detname)
      for evnum,evt in enumerate(orun.events()):
        print('%s\nEvent %04d' % (50*'_',evnum))
        raw = det.raw.raw(evt)
        for segment,panel in raw.items():
            print(segment,panel.shape)
        if evnum > evtmax: break
else: 
    print('NON-RECOGNIZED TEST NAME\n%s' % usage)

print('END OF %s\n%s'%(SCRNAME,50*'-'))
