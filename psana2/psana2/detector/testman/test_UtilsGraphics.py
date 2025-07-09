#!/usr/bin/env python

import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]
STRLOGLEV = sys.argv[2] if len(sys.argv)>2 else 'INFO'

import logging
INTLOGLEV = logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV) #logging.DEBUG)

import psana2.detector.UtilsGraphics as ug
import psana2.pyalgos.generic.NDArrGenerators as ag


def test_01(func):
   """
   """
   flimg = None
   for i in range(10):
       arr = img = ag.random_standard(shape=(256,256), mu=200, sigma=25, dtype=float)
       #amin, amax = ug.arr_median_limits(arr, nneg=1, npos=3)
       #if flimg is None: flimg = ug.fleximage(img, arr=arr, nneg=1, npos=3)
       if flimg is None: flimg = func(img, nneg=3, npos=3)
       else: flimg.update(img)#, arr=arr)
       flimg.axtitle(title='image %d' % i)
       ug.gr.show(mode='NO HOLD')
   ug.gr.show()


USAGE = '\nUsage:'\
      + '\n  python %s <test-name> <loglevel-e.g.-DEBUG-or-INFO>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - print usage'\
      + '\n    1 - test fleximage'\
      + '\n    2 - test fleximagespec'\
      + '\n    3 - test flexhist'\


TNAME = sys.argv[1] if len(sys.argv)>1 else '0'
if   TNAME in  ('1',): test_01(ug.fleximage)
elif TNAME in  ('2',): test_01(ug.fleximagespec)
elif TNAME in  ('3',): test_01(ug.flexhist)
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)

