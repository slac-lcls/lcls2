#!/usr/bin/env python
import logging
#logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s]: %(message)s', level=logging.INFO)

import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]
USAGE = '\nUsage:'\
      + '\n  python %s <test-name>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - print usage'\
      + '\n    1 - issue_2020_11_09 - cpo something about epix10k2M/quad raw'\
      + '\n    2 - issue_2020_11_24 - cpo opal access to consatants for the same detector but other experiment'\

TNAME = sys.argv[1] if len(sys.argv)>1 else '0'


def issue_2020_11_09():
    from psana import DataSource
    ds = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005.xtc2')

    orun = next(ds.runs())
    #for orun in ds.runs():
    det = orun.Detector('epix10k2M')
    for evnum,evt in enumerate(orun.events()):
        print('%s\nEvent %04d' % (50*'_',evnum))
        raw = det.raw.raw(evt)
        print('type(raw)',type(raw))
        #for segment,panel in raw.items():
        for panel in raw:
            print(panel.shape)
    print(50*'-')


def issue_2020_11_24():
    from psana import DataSource
    import numpy as np
    ds = DataSource(exp='tmolw0618',run=52)
    myrun = next(ds.runs())
    det = myrun.Detector('tmoopal')
    print(det.calibconst)


#if __name__ == "__main__":

if   TNAME in ('1',): issue_2020_11_09()
elif TNAME in ('2',): issue_2020_11_24()
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)
