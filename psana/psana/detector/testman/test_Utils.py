#!/usr/bin/env python

import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]
STRLOGLEV = sys.argv[2] if len(sys.argv)>2 else 'INFO'

import logging
INTLOGLEV = logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV) #logging.DEBUG)

import psana.detector.Utils as ut


def test_save_log_record_at_start():
   """save_log_record_at_start(dirrepo, procname, dirmode=0o777, filemode=0o666, tsfmt='%Y-%m-%dT%H:%M:%S%z', scrname='nondef_scrname')
   """
   dirrepo = '/a/b/c/d'
   procname = 'test_procname'
   scrname = 'test_scrname'
   ut.save_log_record_at_start(dirrepo, procname, scrname=scrname)


USAGE = '\nUsage:'\
      + '\n  python %s <test-name> <loglevel-e.g.-DEBUG-or-INFO>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - print usage'\
      + '\n    1 - test save_log_record_at_start'\


TNAME = sys.argv[1] if len(sys.argv)>1 else '0'
if   TNAME in  ('1',): test_save_log_record_at_start()
elif TNAME in  ('2',): test_save_log_record_at_start()
else:
    print(USAGE)
    sys.exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

sys.exit('END OF TEST %s'%TNAME)

