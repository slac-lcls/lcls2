#!/usr/bin/env python

import sys
import logging
SCRNAME = sys.argv[0].rsplit('/')[-1]

STRLOGLEV = sys.argv[2] if len(sys.argv)>2 else 'INFO'
INTLOGLEV = logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)


def issue_2022_01_21():
    """O'Grady, Paul Christopher <cpo@slac.stanford.edu> Wed 1/19/2022 9:08 PM
       Hi Mikhail,
       I took exp=tmoc00318,run=8 that has some epix100 data.  Some caveats, however.
       There are only 4 events. The detector name/type is wrong.
       There is no good detector interface yet, but I kludged one that you can see in
       ~cpo/git/lcls2/psana/psana/detector/ in the files epix100.py and one added line at the bottom of detectors.py.
       With that the script below works for me.
       chris
    """
    from psana import DataSource
    ds = DataSource(exp='tmoc00318',run=8)
    myrun = next(ds.runs())
    det = myrun.Detector('epix100hw')
    for nevt,evt in enumerate(myrun.events()):
        print('det.raw.raw(evt).shape  :', det.raw.raw(evt).shape)
        print('det.raw.calib(evt).shape:', det.raw.calib(evt).shape)
        print('det.raw.image(evt).shape:', det.raw.image(evt).shape)


def issue_2022_01_26():
    """The same as issue_2022_01_21 but for run 10, print ndarray, access constants.
    """
    from psana.pyalgos.generic.NDArrUtils import info_ndarr
    from psana import DataSource
    ds = DataSource(exp='tmoc00318',run=10)
    myrun = next(ds.runs())
    det = myrun.Detector('epix100')

    print('dir(det.raw):', dir(det.raw))
    print()
    print(info_ndarr(det.raw._pedestals(),   'det.raw._pedestals()  '))
    print(info_ndarr(det.raw._gain(),        'det.raw._gain()'))
    print(info_ndarr(det.raw._rms(),         'det.raw._rms()'))
    print(info_ndarr(det.raw._status(),      'det.raw._status()'))
    print(info_ndarr(det.raw._mask_calib(),  'det.raw._mask_calib()'))
    print(info_ndarr(det.raw._mask_from_status(),  'det.raw._mask_from_status()'))
    print(info_ndarr(det.raw._mask_edges(),  'det.raw._mask_edges()'))
    print(info_ndarr(det.raw._common_mode(), 'det.raw._common_mode()'))
    #print(info_ndarr(det.raw.,   'det.raw.'))
    print(info_ndarr(det.raw._pixel_coords(do_tilt=True, cframe=0), 'det.raw._pixel_coords(...)'))

    print()

    for nevt,evt in enumerate(myrun.events()):
        if nevt>10: break
        print(info_ndarr(det.raw.raw(evt),   'det.raw.raw(evt)  '))
        print(info_ndarr(det.raw.calib(evt), 'det.raw.calib(evt)'))


USAGE = '\nUsage:'\
      + '\n  python %s <test-name> <loglevel-e.g.-DEBUG-or-INFO>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - print usage'\
      + '\n    1 - issue_2022_01_21 - cpo epix100hw'\
      + '\n    2 - issue_2022_01_26 - cpo epix100'\


TNAME = sys.argv[1] if len(sys.argv)>1 else '0'

if   TNAME in  ('1',): issue_2022_01_21()
elif TNAME in  ('2',): issue_2022_01_26()
elif TNAME in  ('3',): issue_2022_01_dd()
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)

#if __name__ == "__main__":
