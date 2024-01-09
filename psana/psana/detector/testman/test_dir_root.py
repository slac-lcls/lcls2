#!/usr/bin/env python

"""test of variables in psana/detector/dir_root.py"""

import sys

#import inspect
#self = sys.modules[__name__] # __name__ = '__main__
#s = inspect.getsource(self) # source code of the module
#docstr = self.__doc__ # doc-string of the module
#print(sys._getframe().f_code.co_name)

SCRNAME = sys.argv[0].rsplit('/')[-1]
print("""\n\n%s - %s\n""" % (SCRNAME, sys.modules[__name__].__doc__))

from psana.detector.dir_root import *
print('HOSTNAME', HOSTNAME)
print('DIR_ROOT', DIR_ROOT)
print('DIR_LOG_AT_START', DIR_LOG_AT_START)
print('DIR_REPO', DIR_REPO)
print('DIR_DATA_TEST', DIR_DATA_TEST)
print('DIR_DATA', DIR_DATA)
print('DIR_FFB', DIR_FFB)

assert DIR_ROOT == '/sdf/group/lcls/ds/ana/'
assert DIR_LOG_AT_START == '/sdf/group/lcls/ds/ana/detector/logs/atstart/'
assert DIR_REPO == '/sdf/group/lcls/ds/ana/detector/calib2/constants'
assert DIR_DATA_TEST == '/sdf/group/lcls/ds/ana/detector/data2_test'
assert DIR_DATA == '/sdf/data/lcls/ds/'
assert DIR_FFB == '/sdf/data/lcls/drpsrcf/ffb'

sys.exit('END OF %s' % SCRNAME)
# EOF
